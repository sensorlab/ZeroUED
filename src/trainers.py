from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.metrics import get_supervised_metrics_features
import tqdm
import pickle
import math
import src.metrics as metrics


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

    @abstractmethod
    def evaluate():
        """
        Evaluate the model.
        """
        pass


class SIM_CLR_Trainer(Trainer):

    """
    Inspired by
    
        ''X. Hao, Z. Feng, R. Liu, S. Yang, L. Jiao, and R. Luo, 
        "Contrastive self-supervised clustering for specific emitter identification,” IEEE Internet of 
        Things Journal, vol. 10, no. 23, pp. 20 803–20 818, 2023.''
    
    and 
        ''Viewmaker Networks: Learning Views for Unsupervised Representation Learnin. Alex Tamkin, Mike Wu, Noah Goodman''
    

    DataLoader dataset objects must have the flag "return_indices",
    which enebles or removes sample indices: (inputs, device_ids, ids) or (inputs, device_ids).

    Features excractor must return two outputs: (features, smth) where smth is not used.

    """
    
    def __init__(self, 
                 models: nn.ModuleDict, 
                 optimizers: dict,
                 hard_postitves_mining: bool=False, 
                 hard_negatives_mining: bool=False, 
                 temperature: float = 1, 
                 clusters_loss: bool = False, 
                 augs_type = 'static', 
                 num_epochs = 200,
                 device = 'cpu',
                 large_augs = False
                ):
        """
        Init the ContrastiveTrainer object.
        
        Args:
            models (nn.ModuleDict): Dict of models used in training. Format: {
                'features extrcactor': features extractor model, 
                'mlp_instance': mlp head for instance loss,
                'mlp_cluster': mlp head for cluster loss, optional,
                'augs': augmentations}.
            hard_postitves_mining (bool): remove 1/4 positives with the hieghst similarity, defalut False.
            hard_negatives_mining (bool): remove 1/4 negatives with the lowest similarity, defalut False.
            temperature (float): temperature in cross entropy loss.
            clusters_loss (bool): to use cluster loss or not.
            augs_type (str): 'static' or 'learnable'.
            total_epochs (int): num epochs to learn.
            optimizers (dict): Optimizers for learning in format: {
                'main_optimizer': optimzer for features extractors and mlp heads,
                'augs_optimizer': optimzer for augs,
                }
        """
        self.hard_postitves_mining = hard_postitves_mining
        self.hard_negatives_mining = hard_negatives_mining
        self.models = models
        self.optimizers = optimizers
        self.augs_type = augs_type
        self.clusters_loss = clusters_loss
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.device = device
        self.large_augs = large_augs
        
        
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
            a_norm = torch.norm(a, dim = 1, p = 2) 
            b_norm = torch.norm(b, dim = 1, p = 2)

            a = a / a_norm[:, None]
            b = b / b_norm[:, None]

            #(batch_size_1, batch_size_2)
            sims = a @ b.T

            return sims

        if type == 'l_2':

            #(batch_size_1, batch_size_2, features)
            diff = (a[:, None, :] - b[None, :, :])

            diff  = torch.norm(diff, p=2, dim=-1)

            return diff
    def _normalize(self, x):
        mean = torch.tensor([0.491, 0.482, 0.446], device=x.device)
        std = torch.tensor([0.247, 0.243, 0.261], device=x.device)
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        return x

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
        p_p_sim = self.similarity(p_samples, p_samples, type = 'cosine')
        q_q_sim = self.similarity(q_samples, q_samples, type = 'cosine')
        p_q_sim = self.similarity(p_samples, q_samples, type = 'cosine')

        # Remove the diagonal elements from similarity matrices, (batch_size, batch_size-1)
        n = p_samples.size(0)
        mask = torch.eye(n, device=p_p_sim.device, dtype=torch.bool)
        p_p_sim = p_p_sim[~mask].view(n, n - 1)
        q_q_sim = q_q_sim[~mask].view(n, n - 1)
        
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
        
        if self.hard_postitves_mining:
            # Sort and select top 75% disimilar positives
            _, indices = torch.sort(torch.diag(p_q_sim),  descending=True)
            indices = indices[len(indices) // 4:]

            p_sims = p_sims[indices]
            q_sims = q_sims[indices]
            labels = labels[indices]

        # Compute loss
        l_p = loss_f(p_sims / self.temperature, labels)
        l_q = loss_f(q_sims / self.temperature, labels)

        loss = torch.mean(l_p + l_q) / 2

        return loss

    def _process_batch(self, batch, type = 'features extractor'):
        
        batch = batch.to(self.device)

        features_extractor = self.models['feature_extractor'].to(self.device)
        features_extractor.train()

        augs = self.models['augs'].to(self.device)
        augs.train()

        if self.clusters_loss:
            mlp_cluster = self.models['mlp_cluster'].to(self.device)
            mlp_cluster.train()

        mlp_instance = self.models['mlp_instance'].to(self.device)
        mlp_instance.train()

        p_aug, q_aug = np.random.choice(augs, size = 2)
            
        batch_p = (p_aug(batch))
        batch_q = (q_aug(batch))

        if self.large_augs:
            
            large_augs = self.models['large_augs'].to(self.device)
            large_augs.train()

            p_features = features_extractor.second_part(
                large_augs(features_extractor.first_part(batch_p))
            )

            q_features = features_extractor.second_part(
                large_augs(features_extractor.first_part(batch_q))
            )
            
        else:
            p_features, _ = features_extractor(batch_p)
            q_features, _ = features_extractor(batch_q)

        
            
        p_instance, q_instance = mlp_instance(p_features), mlp_instance(q_features)

        loss = self.compute_loss(p_instance, q_instance)

        if self.clusters_loss:
            p_clusters, q_clusters = mlp_instance(mlp_cluster), mlp_instance(mlp_cluster)
            loss += self.compute_loss(p_clusters.T, q_clusters.T)

        if type == 'features extractor':
            optimizer = self.optimizers['main_optimizer']
            
        else:
            loss = -loss
            optimizer = self.optimizers['augs_optimizer']
            
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        return loss.item()
            
            
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, scheduler=None) -> float:
        """
        Train the feature extractor and MLP heads for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to train on (cpu or cuda).

        Returns:
            float: Average loss for the epoch.
        """
        
        running_loss = 0
        c = 0

        train_loader.dataset.return_indices = False

        for batch_inputs, batch_device_ids in train_loader:
            
            c += 1

            loss = self._process_batch(batch_inputs, 'features extractor')

            if self.augs_type == 'learnable':
                loss -= self._process_batch(batch_inputs, 'augs')
                loss /= 2
                
            running_loss += loss
            
        if scheduler:
            scheduler.step()

        return running_loss / c

    def get_features(self, loader: torch.utils.data.DataLoader, 
                    type = 'features') -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader. The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to use (cpu or cuda).
            
        Returns:
            torch.Tensor: Concatenated features from all batches.
        """
        model = self.models['feature_extractor'].to(self.device)
        model.eval()

        if self.clusters_loss and type == 'probas':
            mlp_cluster = self.models['mlp_cluster'].to(self.device)
            mlp_cluster.eval()
            
        all_features = []

        loader.dataset.return_indices = False
        
        with torch.no_grad():
            
            for inputs, target in tqdm.tqdm(loader):
                inputs, target = inputs.to(self.device), target.to(self.device)

                features, _ = model(inputs)
                
                if self.clusters_loss and type == 'probas':
                    features = mlp_cluster(features)
                    
                all_features.append(features.cpu())

        return torch.cat(all_features)

    def evaluate(self, train_loader, test_loader, targets, clusters_numbers=(40,)):
        
        train_features, test_features = self.get_features(train_loader), self.get_features(test_loader)


        test_devices = []
        for b, devices in test_loader:
            test_devices.append(devices)
        test_devices = torch.cat(test_devices).numpy()

        train_devices = []
        for b, devices in train_loader:
            train_devices.append(devices)
        train_devices = torch.cat(train_devices).numpy()

        supervised_metrics_features = metrics.get_supervised_metrics_features(
            train_features, test_features, targets, clusters_numbers = clusters_numbers,
            test_devices, train_devices
        )

        unsupervised_metrics_features = metrics.get_unsupervised_metrics_features(
            train_features, test_features, clusters_numbers = clusters_numbers
        )
        
        if self.clusters_loss:
            
            train_probas, test_probas = self.get_features(train_loader, type = 'probas'),\
                self.get_features(test_loader, type = 'probas')
            
            supervised_metrics_porbas = metrics.get_supervised_metrics_probas(
                train_probas, test_probas, targets, clusters_numbers)
            
        all_metics = supervised_metrics_features | unsupervised_metrics_features
        
        if self.clusters_loss:
           
           all_metics = all_metics | supervised_metrics_porbas

        return all_metics

    def save_checkpoint(self, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        checkpoint = {
            'state_dict': self.models.state_dict()
        }
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """

        checkpoint = torch.load(file_path)
        self.models.load_state_dict(checkpoint['state_dict'])

class AE_Trainer(Trainer):
    
    """
    Classical Auto Encoder.

    Inspired by L. Milosheski, M. Mohorčič and C. Fortuna, "Spectrum Sensing With Deep Clustering: Label-Free Radio Access Technology Recognition," in IEEE Open Journal of the Communications Society, vol. 5, pp. 4746-4763, 2024, doi: 10.1109/OJCOMS.2024.3436601.
    
    """
    
    def __init__(self, models: nn.ModuleDict, optimizers: dict, 
                noise_std: float = 0, num_epochs = 200, device = 'gpu', distance_loss = False, clusters_update_interval = 4, n_clusters = 40):
        """
        Args:
           noise_std (float): Amount of normal noise applied to the signal.
           models (nn.ModuleDict): Dict of models used in training. Format: {
                'features extrcactor': features extractor model}.
           optimizers (dict): {'main_optimizer': optimizer for features extractor}.
           num_epochs (int): num_epochs
        """
        self.noise_std = noise_std
        self.models = models
        self.optimizers = optimizers
        self.num_epochs = num_epochs
        self.num_epochs = num_epochs
        self.device = device
        self.clusters_centers = None
        self.clusters_update_interval = clusters_update_interval
        self.n_clusters = n_clusters
        self.distance_loss = False
        self.cur_counter = 0


    def _update_clusters(self, train_loader):
        
        train_features = self.get_features(train_loader)
        
        pca = PCA(20)
        
        train_features_reduced = pca.fit_transform(train_features)

        kmeans = KMeans(self.n_clusters)
        
        self.clusters_centers = torch.tensor(kmeans.cluster_centers_)

    def distance_loss(features):

        """
        H. Zhou, J. Bai, Y. Wang, J. Ren, X. Yang, and L. Jiao, “Deep radio
        signal clustering with interpretability analysis based on saliency map" 2023
        
        Algorithm 1
        """
        
        self.clusters_centres = self.clusters_centres.to(features.device)
        
        distances = torch.zeros((self.clusters_centres.shape[0], features.shape[0]), device = features.device)

        for i in range(clusters_centres):
            distances[i] = ((features - self.clusters_centres[i])**2).sum(axis=0)
            
        min_distances = distances.min(axis=0).detach()
        
        exp_shifted_distances = torch.exp(-(distances - min_distances))
        
        exp_shifted_distances_sums = exp_shifted_distances.sum(axis = 1)
        
        weighted_distances = distances * exp_shifted_distances / exp_shifted_distances_sums
        
        return weighted_distances.mean()
        
        

    def train_epoch(self, train_loader, scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> float:
        """
        Train the feature extractor for one epoch.

        Args:
            device (torch.device): Device to train on (CPU or CUDA).
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Default is None.

        Returns:
            float: Loss of the epoch.
        """
        model = self.models['feature_extractor'].to(self.device)
        model.train()

        optimizer = self.optimizers['main_optimizer']

        
        running_loss = 0
        c = 0

        train_loader.dataset.return_indices = False

        if self.distance_loss and self.cur_counter % self.clusters_update_interval == 0:
            self._update_clusters(train_loader)
            self.clusters_update_interval = 0
            
        self.clusters_update_interval += 1
        
        for inputs, target in train_loader:

            optimizer.zero_grad()

            inputs, target = inputs.to(self.device), target.to(self.device)

            # Apply noise to inputs
            x = (inputs + torch.randn(inputs.shape, device=inputs.device) * self.noise_std) / (1 + self.noise_std)

            features, reconstructed = model(x)

            features = features.view(features.shape[0], -1)

            loss = torch.mean((reconstructed - inputs)**2)

            if self.distance_loss:
               loss += self.distance_loss(features) 

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            c += 1
            
        return running_loss / c
        
    def get_features(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to use (CPU or CUDA).
            
        Returns:
            torch.Tensor: Concatenated features from all batches.
        """

        model = self.models['feature_extractor'].to(self.device)
        model.eval()
        
        all_features = []

        loader.dataset.return_indices = False

        for inputs, target in loader:

            inputs, target = inputs.to(self.device), target.to(self.device)

            _, features = model(inputs)

            all_features.append(features.detach().cpu().view(features.shape[0], -1))

        return torch.cat(all_features)

    def evaluate(self, train_loader, test_loader, targets, clusters_numbers=(40,)):
        
        train_features, test_features = self.get_features(train_loader), self.get_features(test_loader)


        test_devices = []
        for b, devices in test_loader:
            test_devices.append(devices)
        test_devices = torch.cat(test_devices).numpy()

        train_devices = []
        for b, devices in train_loader:
            train_devices.append(devices)
        train_devices = torch.cat(train_devices).numpy()

        supervised_metrics_features = metrics.get_supervised_metrics_features(
            train_features, test_features, targets, clusters_numbers = clusters_numbers,
            test_devices, train_devices
        )

        unsupervised_metrics_features = metrics.get_unsupervised_metrics_features(
            train_features, test_features, clusters_numbers = clusters_numbers
        )
            
        all_metics = supervised_metrics_features | unsupervised_metrics_features

        return all_metics

    def save_checkpoint(self, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        torch.save({
            'model_state_dict': self.models.state_dict(),
            }, file_path)

    def load_checkpoint(self, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """
        checkpoint = torch.load(file_path, weights_only=True)
        
        self.models.load_state_dict(checkpoint['model_state_dict'])
    


class PCA_Trainer(Trainer):

    def __init__(self, pca: PCA):
        """
        Makes PCA
        
        Args:
            pca: PCA solver from as in sklearn

        Returns:
            None
        """
        
        self.pca = pca
        

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train the PCA extractor.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).

        Returns:
            float: Explained variance ratio.
        """
        model = self.pca
        features_train = []

        train_loader.dataset.return_indices = False

        for inputs, _ in train_loader:

            features_train.append(inputs.reshape(inputs.shape[0], -1).detach().cpu())

        features_train = torch.cat(features_train)
        
        model.fit(features_train)

        return model.explained_variance_ratio_.sum()
        
    def get_features(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).

        Returns:
            torch.Tensor: Transformed features.
        """
        model = self.pca
        
        all_features = []

        loader.dataset.return_indices = False

        for inputs, _ in loader:

            all_features.append(inputs.reshape(inputs.shape[0], -1).detach().cpu())

        all_features = torch.cat(all_features)
        
        return torch.tensor( model.transform(all_features))

    def evaluate(self, train_loader, test_loader, targets, clusters_numbers=(40,)):
        
        train_features, test_features = self.get_features(train_loader), self.get_features(test_loader)


        test_devices = []
        for b, devices in test_loader:
            test_devices.append(devices)
        test_devices = torch.cat(test_devices).numpy()

        train_devices = []
        for b, devices in train_loader:
            train_devices.append(devices)
        train_devices = torch.cat(train_devices).numpy()

        supervised_metrics_features = metrics.get_supervised_metrics_features(
            train_features, test_features, targets, clusters_numbers = clusters_numbers,
            test_devices, train_devices
        )

        unsupervised_metrics_features = metrics.get_unsupervised_metrics_features(
            train_features, test_features, clusters_numbers = clusters_numbers
        )
            
        all_metics = supervised_metrics_features | unsupervised_metrics_features

        return all_metics
        
    def save_checkpoint(self,file_path: str) -> None:
        """
        Save a checkpoint of the PCA extractor.

        Args:
            pca_extractor (PCA): PCA extractor from sklearn.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        model = self.pca

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

       
    def load_checkpoint(self, file_path: str) -> PCA:
        """
        Load a checkpoint of the PCA extractor.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            PCA: Loaded PCA extractor.
        """
        
        with open(file_path, 'rb') as f:
            self.pca = pickle.load(f)
       


class Deep_Clustering_Trainer(Trainer):
    
    """
    Implementation of deep clustering algorithm. Caron, Mathilde, et al. "Deep clustering for unsupervised learning of visual features." Proceedings of the European conference on computer vision (ECCV). 2018.

    Dataloaders datasets objects must have the flag "return_indices" which makes dataloader return samples indices at the end of the touple: (inputs, deivce_ids, ids).

    
    """

    def __init__(self, models, optimizers, clusters_update_interval: int = 5, n_clusters: int = 10, num_epochs = 200,
                device = 'cuda'):
        """
        Args:
            clusters_update_interval (int): Number of epochs between clusters updates.
            n_clusters (int): Number of clusters.
            optimizers (dict): Optimizers for learning in format: {
                'main_optimizer': optimzer for features extractors and mlp heads}s
            models (nn.ModuleDict): Dict of models used in training. Format: {
                'features extrcactor': features extractor model }
                
        """

        self.num_epochs = num_epochs
        self.device = device

        self.models = models
        self.optimizers = optimizers

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

    def get_features(self, loader: torch.utils.data.DataLoader, 
                     type: str = 'features', return_indices: bool = False) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to use (cpu or cuda).
            type (str): Type of output ('features' or 'scores').
            return_indices (bool): Whether to return indices.

        Returns:
            torch.Tensor: Features or scores.
        """
        
        model = self.models['feature_extractor'].to(self.device)
        model.eval()

        features_list = []
        ids_list = []

        loader.dataset.return_indices = True

        with torch.no_grad():
            for inputs, _, ids in loader:

                inputs = inputs.to(self.device)

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
        

    def _update_labels(self, train_loader: torch.utils.data.DataLoader):
        """
        Recalculate pseudo labels.

        Args:
            train_loader (torch.utils.data.DataLoader): Train DataLoader. 
                The flag "return_indices" must exist in train_loader.dataset object, 
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).
            device (torch.device): Device to use (CPU or CUDA).
        """
        model = self.models['feature_extractor'].to(self.device)
        model.eval()
        
        train_features, ids = self.get_features( train_loader, return_indices=True)

        self.ids = ids
        
        pca = PCA(20)
        train_features_reduced = pca.fit_transform(train_features)

        kmeans = KMeans(self.n_clusters)
        self.p_labels =  torch.tensor(kmeans.fit_predict(train_features_reduced), dtype=torch.long)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> float:
        """
        Train the model for one epoch.

        Args:
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

        model = self.models['feature_extractor'].to(self.device)
        model.train()

        if self.cur_counter % self.clusters_update_interval == 0:
            self.cur_counter = 0
            self._update_labels(train_loader)

        self.cur_counter += 1

        train_loader.dataset.return_indices = True

        optimizer = self.optimizers['main_optimizer']
        
 
        for inputs, _, ids in train_loader:

            optimizer.zero_grad()
            
            inputs = inputs.to(self.device)

            p_labels = self.p_labels[ids].to(self.device)

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

    def evaluate(self, train_loader, test_loader, targets, clusters_numbers=(40,)):
        
        train_features, test_features = self.get_features(train_loader), self.get_features(test_loader)

        test_devices = []
        for b, devices in test_loader:
            test_devices.append(devices)
        test_devices = torch.cat(test_devices).numpy()

        train_devices = []
        for b, devices in train_loader:
            train_devices.append(devices)
        train_devices = torch.cat(train_devices).numpy()

        supervised_metrics_features = metrics.get_supervised_metrics_features(
            train_features, test_features, targets, clusters_numbers = clusters_numbers,
            test_devices, train_devices
        )

        unsupervised_metrics_features = metrics.get_unsupervised_metrics_features(
            train_features, test_features, clusters_numbers = clusters_numbers
        )

        train_probas, test_probas = self.get_features(train_loader, type = 'scores'),\
                self.get_features(test_loader, type = 'scores')
        
        supervised_metrics_porbas = metrics.get_supervised_metrics_probas(
                train_probas, test_probas, targets, clusters_numbers)


        all_metics = supervised_metrics_features | unsupervised_metrics_features | supervised_metrics_porbas

        return all_metics

    def save_checkpoint(self, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """

        torch.save({'model_state_dict': self.models.state_dict()}, file_path)

    def load_checkpoint(self, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """

        checkpoint = torch.load(file_path, weights_only=True)
        self.models.load_state_dict(checkpoint['model_state_dict'])