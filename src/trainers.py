from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
class Trainer(ABC):
    """
    Abstract base class for training models.

    This class defines the interface that all trainer classes should implement.
    """

    @abstractmethod
    def train_epoch(self, model, train_loader, optimizer, device):
        """
        Train the model for one epoch.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device to train on (CPU or GPU).

        Returns:
            loss of the epoch
            
        """
        pass
    
    @abstractmethod
    def get_features(self, model, train_loader, val_loader, device):
        """
        Get final embedings from train and val samples

        Args:
            model (torch.nn.Module): The features extractor model.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            device (torch.device): Device to train on (CPU or GPU).

        Returns:
            train_features (torch.tensor), test_features (torch.tensor)
            
        """
        pass

    @abstractmethod
    def save_checkpoint(self, model, file_path):
        """
        Save a checkpoint of the model.

        Args:
            model (torch.nn.Module): The model to save.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load_checkpoint(self, model, file_path):
        """
        Load a checkpoint of the model.

        Args:
            model (torch.nn.Module): The model to load.
            file_path (str): Path to the checkpoint file.

        Returns:
            int: The epoch at which the checkpoint was saved.
        """
        pass


class SIM_CLR_Trainer(Trainer):

    """
    X. Hao, Z. Feng, R. Liu, S. Yang, L. Jiao, and R. Luo, 
    "Contrastive self-supervised clustering for specific emitter identification,” IEEE Internet of 
    Things Journal, vol. 10, no. 23, pp. 20 803–20 818, 2023.
    """
    
    def __init__(self, augs):
        """
        Init the ContrastiveTrainer object 

        Args:
            augs: list of augimentations applied to the data samples
        """
        self.augs = augs

    def similratiry(self, a, b, type = 'cosine'):
        """
        Calculates the simmiliraty function between batches of embedings

        Args:
            a (torch.tensor): the first batch of embeings of the shape (batch_size, n_features)
            b (torch.tensor): the second batch of embeings of the shape (batch_size, n_features)
            type (str): type of similratiry function

        Returns: 
            torch.tensor: similratiry matrix between samlples
            
        """
        #(batch_size, n_features)
        
        if type == 'cosine':
            
            #(batch_size,)
            
            a_norm = torch.sqrt((a**2).sum(axis=-1))
            b_norm = torch.sqrt((b**2).sum(axis=-1))

            #(batch_size_1, batch_size_2)

            sims = a @ b.T / (a_norm[:, None] * b_norm[None, :])

            return sims
                

    def loss_sim_clr(self, p_samples, q_samples):
        """
        Calculates the simmiliraty function between batches of embedings

        Args:
            a (torch.tensor): the first batch of embeings of the shape (batch_size, n_features)
            b (torch.tensor): the second batch of embeings of the shape (batch_size, n_features)
            type (str): type of similratiry function
            
        Returns: 
            torch.tensor: simmilarity loss
            
        
        """
        # (batch_size, batch_size)
        
        p_p_sim = self.similratiry(p_samples, p_samples)
        q_q_sim = self.similratiry(q_samples, q_samples)
        p_q_sim = self.similratiry(p_samples, q_samples)

        # contrastive loss
        
        l_p = - torch.diag(p_q_sim) + torch.log(
            torch.exp(p_p_sim).sum(axis=-1) +
            torch.exp(p_q_sim).sum(axis=-1)
            - torch.tensor([np.exp(1)] * p_p_sim.shape[1], device = p_p_sim.device)
        )

        l_q = - torch.diag(p_q_sim) + torch.log(
            torch.exp(q_q_sim).sum(axis=-1) +
            torch.exp(p_q_sim).sum(axis=0)
            - torch.tensor([np.exp(1)] * p_p_sim.shape[1], device = p_p_sim.device)
        )

        
        loss = torch.mean(l_p + l_q) / 2

        return loss
    
        
    def train_epoch(self, model, mlp_instance, mlp_cluster, train_loader, val_loader, optimizer, device='cpu', scheduler=None):
        """
        Train the features extractor and mlp heads for one epoch.

        Args:
            model (torch.nn.Module): features extracor.
            mlp_instance (torch.nn.Module): mlp head for instance loss.
            mlp_cluster (torch.nn.Module): mlp head for cluster loss.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device to train on (CPU or GPU).
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.

        Returns:
            loss of the epoch
        """
        
        running_loss = 0
        c = 0
        
        model = model.to(device)
        model.train()

        mlp_instance.to(device)
        mlp_instance.train()

        mlp_cluster.to(device)
        mlp_cluster.train()
        
        for inputs, target in train_loader:

            inputs, target = inputs.to(device), target.to(device)

            # (batch_size, f_size)
            optimizer.zero_grad()
            c += 1
                
            # apply augs
            
            p_aug, q_aug = np.random.choice(self.augs, size = 2, replace = False)
            
            batch_p = p_aug(inputs)

            batch_q = q_aug(inputs)

            # instance and cluster representations
            
            p_features, _ = model(batch_p)
            q_features, _ = model(batch_q)
            
            # (batch size, num_features)
            
            p_instance, q_instance = mlp_instance(p_features), mlp_instance(q_features)

            # (batch size, num_clusters)
            
            p_cluster, q_cluster = mlp_cluster(p_features), mlp_cluster(q_features)

            # instance and cluster loss

            instance_loss = self.loss_sim_clr(p_instance, q_instance)

            p_cluster_probas = p_cluster.sum(axis=0) / torch.sum(p_cluster)

            q_cluster_probas = q_cluster.sum(axis=0) / torch.sum(q_cluster)

            cluster_loss = self.loss_sim_clr(p_cluster.T, q_cluster.T) #+\
            
            #torch.sum(
            #    torch.log(p_cluster_probas) * p_cluster_probas + 
            #    torch.log(q_cluster_probas) * q_cluster_probas
            #)

            loss = instance_loss + cluster_loss

            # gradient step
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        return running_loss / c

    def get_features(self, model, train_loader, val_loader, device = 'cpu',  mlp = None):
        """
        Get final embedings from train and val samples

        Args:
            model (torch.nn.Module): features extracor.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            device (torch.device): Device to train on (CPU or GPU).
            mlp (torch.nn.Module): mlp head upon feature extracor, default is None which means no mpl is applied.
            
        Returns:
            train_features (torch.tensor), test_features (torch.tensor)
        """
        train_features = []
        test_features = []

        model = model.to(device)
        model.eval()

        if not (mlp is None):
            mlp = mlp.to(device)
            mlp.eval()
        
        for inputs, target in train_loader:

            # (batch_size, f_size)
            
            inputs, target = inputs.to(device), target.to(device)

            features,_ = model(inputs)
            
            if not (mlp is None):
                features = mlp(features)
                
            train_features.append(features.cpu().detach())

        for inputs, target in val_loader:

            # (batch_size, f_size)
            
            inputs, target = inputs.to(device), target.to(device)

            features,_ = model(inputs)

            if not (mlp is None):
                features = mlp(features)
                
            test_features.append(features.cpu().detach())

        return torch.cat(train_features), torch.cat(test_features)


    def save_checkpoint(self, model, mlp_instance, mlp_cluster, file_path):
        """
        Save a checkpoint of the model.

        Args:
            model (torch.nn.Module): features extracor.
            mlp_instance (torch.nn.Module): mlp head for instance loss.
            mlp_cluster (torch.nn.Module): mlp head for cluster loss.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        torch.save({
            'model_state_dict': model.state_dict(),
            'mlp_instance_state_dict': mlp_instance.state_dict(),
            'mlp_cluster_state_dict': mlp_cluster.state_dict(),
            }, file_path)

    def load_checkpoint(self, model, mlp_instance, mlp_cluster, file_path):
        """
        Load a checkpoint of the model.

        Args:
            model (torch.nn.Module): features extracor.
            mlp_instance (torch.nn.Module): mlp head for instance loss.
            mlp_cluster (torch.nn.Module): mlp head for cluster loss.
            optimizer (torch.optim.Optimizer): The optimizer.
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """
        checkpoint = torch.load(file_path, weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        mlp_instance.load_state_dict(checkpoint['mlp_instance_state_dict'])
        mlp_cluster.load_state_dict(checkpoint['mlp_cluster_state_dict'])
            