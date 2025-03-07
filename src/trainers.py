from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tqdm
import pickle

class Trainer(ABC):
    """
    Abstract base class for training models.

    This class defines the interface that all trainer classes should implement.
    """

    @abstractmethod
    def train_epoch():
        """
        Train the model for one epoch.
        """
        pass
    
    @abstractmethod
    def get_features():
        """
        Get final embedings of the data.
        """
        pass

    @abstractmethod
    def save_checkpoint():
        """
        Save a checkpoint of the model.
        """
        pass

    @abstractmethod
    def load_checkpoint():
        """
        Load a checkpoint of the model.
        """
        pass


class SIM_CLR_Trainer(Trainer):

    """
    The implementation of
    ''X. Hao, Z. Feng, R. Liu, S. Yang, L. Jiao, and R. Luo, 
    "Contrastive self-supervised clustering for specific emitter identification,” IEEE Internet of 
    Things Journal, vol. 10, no. 23, pp. 20 803–20 818, 2023.''
    with learnable augmentations.

    DataLoader dataset objects must have the flag "return_indices",
    which enebles or removes sample indices: (inputs, device_ids, ids) or (inputs, device_ids).

    Features excractor must return two outputs: (features, smth) where smth is not used.

    """
    
    def __init__(self, hard_postitves_mining=False, hard_negatives_mining=False):
        """
        Init the ContrastiveTrainer object.
        
        Args:
            augs (nn.ModuleList): list of augimentations applied to the data samples.
            hard_postitves_mining (bool): remove 1/4 positives with the hieghst similarity, defalut False.
            hard_negatives_mining (bool): remove 1/4 negatives with the lowest similarity, defalut False.
            
        """
        self.sehard_postitves_mining = hard_postitves_mining
        self.hard_negatives_mining = hard_negatives_mining
        
    def similarity(self, a, b, type = 'cosine'):
        """
        Calculates the similarity function between batches of embedings.

        Args:
            a (torch.tensor): the first batch of embeings of the shape (batch_size, n_features).
            b (torch.tensor): the second batch of embeings of the shape (batch_size, n_features).
            type (str): type of similratiry function.

        Returns: 
            torch.tensor: similarity  matrix between samlples.
            
        """
        #(batch_size, n_features)
        
        if type == 'cosine':
            
            #(batch_size,)
            
            a_norm = torch.sqrt((a**2).sum(axis=-1))
            b_norm = torch.sqrt((b**2).sum(axis=-1))

            #(batch_size_1, batch_size_2)

            sims = a @ b.T / (a_norm[:, None] * b_norm[None, :])

            return sims

        if type == 'l_2':

            #(batch_size_1, batch_size_2, features)

            diff = (a[:, None, :] - b[None, :, :])

            diff = torch.sqrt(torch.mean(diff**2, -1))

            return diff
                

    def compute_loss(self, p_samples: torch.Tensor, q_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss.

        Args:
            p_samples (torch.Tensor): Positive samples.
            q_samples (torch.Tensor): Query samples.

        Returns:
            torch.Tensor: Computed loss.
        """
        loss_f = torch.nn.CrossEntropyLoss()

        # Compute similarity matrices, (batch_size, batch_size)
        p_p_sim = self.similarity(p_samples, p_samples)
        q_q_sim = self.similarity(q_samples, q_samples)
        p_q_sim = self.similarity(p_samples, q_samples)

        # Remove the diagonal elements from similarity matrices, (batch_size, batch_size-1)
        n = p_samples.size(0)
        p_p_sim = p_p_sim.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        q_q_sim = q_q_sim.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        
        if self.hard_negatives_mining:
            # Sort and select top 75% similar negatives
            p_p_sim, _ = torch.sort(p_p_sim, dim=-1, descending=True)
            p_p_sim = p_p_sim[:, :(3 * p_p_sim.shape[1]) // 4]

            q_q_sim, _ = torch.sort(q_q_sim, dim=-1, descending=True)
            q_q_sim = q_q_sim[:, :(3 * q_q_sim.shape[1]) // 4]
        
        # Concatenate similarities for cross-entropy loss
        p_sims = torch.cat([p_q_sim, p_p_sim], 1)
        q_sims = torch.cat([p_q_sim, q_q_sim], 1)
        
        labels = torch.arange(p_sims.shape[0], device = p_sims.device, dtype = torch.long)
        
        if self.sehard_postitves_mining:
            # Sort and select top 75% disimilar positives
            _, indices = torch.sort(torch.diag(p_q_sim),  descending=True)
            indices = indices[len(indices) // 4:]

            p_sims = p_sims[indices]
            q_sims = q_sims[indices]
            labels = labels[indices]

        # Compute loss
        l_p = loss_f(p_sims, labels)
        l_q = loss_f(q_sims, labels)

        loss = torch.mean(l_p + l_q) / 2

        return loss
    
        
    def train_epoch(self, model: torch.nn.Module, mlp_instance: torch.nn.Module, mlp_cluster: torch.nn.Module, 
                    augs: torch.nn.ModuleList, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
                    device: torch.device = torch.device('cpu'), epoch_type: str = 'features extractor', 
                    cluster_loss: bool = True, scheduler=None) -> float:
        """
        Train the feature extractor and MLP heads for one epoch.

        Args:
            model (torch.nn.Module): Feature extractor, output must be (features, smth), where smth is not used.
            mlp_instance (torch.nn.Module): MLP head for instance loss.
            mlp_cluster (torch.nn.Module): MLP head for cluster loss.
            augs (torch.nn.ModuleList): List of augmentations.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            optimizer (torch.optim.Optimizer): Optimizer. The last parameter group must be augmentations.
            device (torch.device): Device to train on (cpu or cuda).
            epoch_type (str): Type of epoch ('features extractor' or other).
            cluster_loss (bool): Whether to compute cluster loss.
            scheduler: Learning rate scheduler.

        Returns:
            float: Average loss for the epoch.
        """
        
        running_loss = 0
        c = 0
        
        model = model.to(device)
        model.train()

        mlp_instance.to(device)
        mlp_instance.train()

        mlp_cluster.to(device)
        mlp_cluster.train()

        augs.to(device)
        augs.train()

        train_loader.dataset.return_indices = False

        for batch_inputs, batch_device_ids in train_loader:
            optimizer.zero_grad()

            # (batch_size, f_size)
            batch_inputs= batch_inputs.to(device)
                
            # apply augs
            p_aug, q_aug = np.random.choice(augs, size = 2, replace = False)
            batch_p = p_aug(batch_inputs)
            batch_q = q_aug(batch_inputs)

            # instance and cluster representations
            p_features, _ = model(batch_p)
            q_features, _ = model(batch_q)
            
            # (batch size, num_features)
            p_instance, q_instance = mlp_instance(p_features), mlp_instance(q_features)

            # (batch size, num_clusters)
            p_cluster, q_cluster = mlp_cluster(p_features), mlp_cluster(q_features)

            # instance and cluster loss
            loss = self.compute_loss(p_instance, q_instance)

            if cluster_loss:
                loss += self.compute_loss(p_cluster.T, q_cluster.T)

            if epoch_type == 'features extractor':

                aug_lr = optimizer.param_groups[-1]['lr']
                optimizer.param_groups[-1]['lr'] = 0
                
                (loss).backward()
                optimizer.step()

                optimizer.param_groups[-1]['lr'] = aug_lr
                
            elif epoch_type == 'augs':

                feat_lrs = []

                for group in optimizer.param_groups[:-1]:
                    feat_lrs.append(group['lr'])
                    group['lr'] = 0
                
                (-loss).backward()
                optimizer.step()
            

                for lr, group in zip(feat_lrs, optimizer.param_groups[:-1]):
                    group['lr'] = lr
            
            running_loss += (loss).item()
            c += 1

        if scheduler:
            scheduler.step()

        return running_loss / c

    def get_features(self, model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device = torch.device('cpu'), mlp: torch.nn.Module = None) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            model (torch.nn.Module): Feature extractor, output must be (features, smth), where smth is not used.
            loader (torch.utils.data.DataLoader): DataLoader. The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to use (cpu or cuda).
            mlp (torch.nn.Module): MLP head upon feature extractor, default is None which means no MLP is applied.
            
        Returns:
            torch.Tensor: Concatenated features from all batches.
        """
        all_features = []
        
        model = model.to(device)
        model.eval()

        if mlp is not None:
            mlp = mlp.to(device)
            mlp.eval()

        loader.dataset.return_indices = False
        
        with torch.no_grad():
            for inputs, target in tqdm.tqdm(loader):
                inputs, target = inputs.to(device), target.to(device)

                features, _ = model(inputs)
                
                if mlp is not None:
                    features = mlp(features)
                    
                all_features.append(features.cpu())

        return torch.cat(all_features)


    def save_checkpoint(self, model: torch.nn.Module, mlp_instance: torch.nn.Module, mlp_cluster: torch.nn.Module, 
                        augs: torch.nn.Module, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            model (torch.nn.Module): Feature extractor.
            mlp_instance (torch.nn.Module): MLP head for instance loss.
            mlp_cluster (torch.nn.Module): MLP head for cluster loss.
            augs (torch.nn.ModuleList): List of augmentations.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'mlp_instance_state_dict': mlp_instance.state_dict(),
            'mlp_cluster_state_dict': mlp_cluster.state_dict(),
            'augs_state_dict': [aug.state_dict() for aug in augs]
        }
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, model: torch.nn.Module, mlp_instance: torch.nn.Module, mlp_cluster: torch.nn.Module, 
                        augs: torch.nn.ModuleList, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            model (torch.nn.Module): Feature extractor.
            mlp_instance (torch.nn.Module): MLP head for instance loss.
            mlp_cluster (torch.nn.Module): MLP head for cluster loss.
            augs (torch.nn.ModuleList): List of augmentations.
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """

        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        mlp_instance.load_state_dict(checkpoint['mlp_instance_state_dict'])
        mlp_cluster.load_state_dict(checkpoint['mlp_cluster_state_dict'])
        for aug, state_dict in zip(augs, checkpoint['augs_state_dict']):
            aug.load_state_dict(state_dict)
        

class AE_Trainer(Trainer):
    
    """
    Classical Auto Encoder
    """
    
    def __init__(self, noise_std: float = 0):
        """
        Args:
           noise_std (float): Amount of normal noise applied to the signal.
        """
        self.noise_std = noise_std

    def train_epoch(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device = torch.device('cpu'), scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> float:
        """
        Train the feature extractor for one epoch.

        Args:
            model (torch.nn.Module): Feature extractor. Must return two outputs: (features, reconstructed).
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device to train on (CPU or CUDA).
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Default is None.

        Returns:
            float: Loss of the epoch.
        """
        running_loss = 0
        c = 0
        
        model = model.to(device)
        model.train()

        train_loader.dataset.return_indices = False
        
        for inputs, target in train_loader:

            optimizer.zero_grad()

            inputs, target = inputs.to(device), target.to(device)

            # Apply noise to inputs
            x = (inputs + torch.randn(inputs.shape, device=inputs.device) * self.noise_std) / (1 + self.noise_std)

            _, reconstructed = model(x)

            loss = torch.mean((reconstructed - inputs)**2)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            c += 1
            
        return running_loss / c
        
    def get_features(self, model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            model (torch.nn.Module): Feature extractor.
            loader (torch.utils.data.DataLoader): DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to use (CPU or CUDA).
            
        Returns:
            torch.Tensor: Concatenated features from all batches.
        """
        
        model = model.to(device)
        model.eval()

        all_features = []

        loader.dataset.return_indices = False

        for inputs, target in loader:

            inputs, target = inputs.to(device), target.to(device)

            _, features = model(inputs)

            all_features.append(features.detach().cpu())

        return torch.cat(all_features)

    def save_checkpoint(self, model: torch.nn.Module, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            model (torch.nn.Module): Feature extractor.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        torch.save({
            'model_state_dict': model.state_dict(),
            }, file_path)

    def load_checkpoint(self, model: torch.nn.Module, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            model (torch.nn.Module): Feature extractor.
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """
        checkpoint = torch.load(file_path, weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
    


class PCA_Trainer(Trainer):

    def __init__(self):
        pass

    def train_epoch(self, pca_extractor: PCA, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train the PCA extractor.

        Args:
            pca_extractor (PCA): PCA extractor from sklearn.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).

        Returns:
            float: Explained variance ratio.
        """
        features_train = []

        train_loader.dataset.return_indices = False

        for inputs, _ in train_loader:

            features_train.append(inputs.reshape(inputs.shape[0], -1).detach().cpu())

        features_train = torch.cat(features_train)
        
        pca_extractor.fit(features_train)

        return pca_extractor.explained_variance_ratio_.sum()
        
    def get_features(self, pca_extractor: PCA, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            pca_extractor (PCA): PCA extractor from sklearn.
            loader (torch.utils.data.DataLoader): DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).

        Returns:
            torch.Tensor: Transformed features.
        """
        
        all_features = []

        loader.dataset.return_indices = False

        for inputs, _ in loader:

            all_features.append(inputs.reshape(inputs.shape[0], -1).detach().cpu())

        all_features = torch.cat(all_features)
        
        return torch.tensor( pca_extractor.transform(all_features))
        
    def save_checkpoint(self, pca_extractor: PCA, file_path: str) -> None:
        """
        Save a checkpoint of the PCA extractor.

        Args:
            pca_extractor (PCA): PCA extractor from sklearn.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """

        with open(file_path, 'wb') as f:
            pickle.dump(pca_extractor, f)

       
    def load_checkpoint(self, file_path: str) -> PCA:
        """
        Load a checkpoint of the PCA extractor.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            PCA: Loaded PCA extractor.
        """
        
        with open(file_path, 'rb') as f:
            pca_extractor = pickle.load(f)
            
        return pca_extractor
       


class Deep_Clustering_Trainer(Trainer):

    """
    Implementation of deep clustering algorithm.

    Dataloaders datasets objects must have the flag "return_indices" which makes dataloader return samples indices at the end of the touple: (inputs, deivce_ids, ids).
    """

    def __init__(self, clusters_update_interval: int, n_clusters: int = 10):
        """
        Args:
            clusters_update_interval (int): Number of epochs between clusters updates.
            n_clusters (int): Number of clusters.
        """

        # counter for clusters update
        self.cur_counter = 0

        # interval between clusters updates
        self.clusters_update_interval = clusters_update_interval

        # loss function
        self.loss_func = nn.CrossEntropyLoss()

        # number of clusters
        self.n_clusters = n_clusters

        # softmax
        self.softmax = nn.Softmax(dim=-1)

        # pseudo labels and sample indices
        self.ids = None
        self.p_labels = None

    def get_features(self, model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, 
                     type: str = 'features', return_indices: bool = False) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            model (torch.nn.Module): Feature extractor. Must return two outputs: (features, scores).
            loader (torch.utils.data.DataLoader): DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to use (cpu or cuda).
            type (str): Type of output ('features' or 'scores').
            return_indices (bool): Whether to return indices.

        Returns:
            torch.Tensor: Features or scores.
        """
        
        model = model.to(device)
        model.eval()

        features_list = []
        ids_list = []

        loader.dataset.return_indices = True

        with torch.no_grad():
            for inputs, _, ids in loader:

                inputs = inputs.to(device)

                features, scores = model(inputs)

                if type == 'features':

                    features_list.append(features.cpu())

                elif type == 'scores':

                    features_list.append(
                        self.softmax(scores.cpu())
                    )
                
                ids_list.append(ids)
            
        if return_indices:

            return torch.cat(features_list), torch.cat(ids_list)

        else:

            return torch.cat(features_list)
        

    def _update_labels(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.device):
        """
        Recalculate pseudo labels.

        Args:
            model (torch.nn.Module): Feature extractor. Must return two outputs: (features, scores).
            train_loader (torch.utils.data.DataLoader): Train DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to use (CPU or CUDA).
        """
        
        train_features, ids = self.get_features(model, train_loader, device, return_indices=True)

        self.ids = ids
        
        pca = PCA(20)
        train_features_reduced = pca.fit_transform(train_features, return_indices=True)

        kmeans = KMeans(self.number_of_clustres)
        self.p_labels =  torch.tensor(kmeans.fit_predict(train_features_reduced), dtype=torch.long)

    def train_epoch(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
                    device: torch.device = torch.device('cpu'), scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> float:
        """
        Train the model for one epoch.

        Args:
            model (torch.nn.Module): Feature extractor. Must return two outputs: (features, scores).
            train_loader (torch.utils.data.DataLoader): Train DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device to train on (CPU or CUDA).
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Default is None.

        Returns:
            float: Average loss for the epoch.
        """
        
        running_loss = 0
        c = 0

        if self.cur_counter % self.clustres_update_interval == 0:
            self.cur_counter = 1
            self._update_labels(model, train_loader, device)

        self.cur_counter += 1

        train_loader.dataset.return_indices = True
        
        model = model.to(device)
        model.train()
 
        for inputs, _, ids in train_loader:

            optimizer.zero_grad()
            
            inputs = inputs.to(device)

            p_labels = self.p_labels[self.ids == ids].to(device)

            _, scores = model(inputs)

            loss = self.loss_func(
                scores, p_labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            c += 1
        
        if scheduler:
            scheduler.step()

        return running_loss / c

    def save_checkpoint(self, model: nn.Module, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            model (torch.nn.Module): Feature extractor.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """

        torch.save({'model_state_dict': model.state_dict()}, file_path)

    def load_checkpoint(self, model: nn.Module, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            model (torch.nn.Module): Feature extractor.
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """

        checkpoint = torch.load(file_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])