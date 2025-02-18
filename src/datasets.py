import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

class DronesDataset(Dataset):
    
    name = 'DronesDataset'
    
    def __init__(self, dataset_path, uavs=None, bursts=None, transform=None, limit=None, labels=None, complex_input = False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.limit = limit
        self.labels = labels
        self.sigma = 7239
        self.mu = 1.5

        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            self.drone_ids = fp['labels'][:,0]
            self.dists = fp['labels'][:,1]
            self.bursts = fp['labels'][:,2]
            
            mask = [self.drone_ids[i] in uavs and self.bursts[i] in bursts for i in range(len(self.drone_ids))]

            if complex_input:
                self.data_i = np.real(fp['data'][mask])
                self.data_q = np.im(fp['data'][mask])
            else:
                self.data_i = fp['data_i'][mask]
                self.data_q = fp['data_q'][mask]
            
            
            self.drone_ids = self.drone_ids[mask]
            self.dists = self.dists[mask]
            self.bursts = self.bursts[mask]

            self.data_length = sum(mask)
    
    def __len__(self):
        if self.limit:
            return self.limit
        else:
            return self.data_length

    def __getitem__(self, idx):
        
        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            sample_i = self.data_i[idx]
            sample_q = self.data_q[idx]
            label = self.drone_ids[idx]
            
            sample = np.concatenate([sample_i, sample_q])
            sample = sample.reshape((2,-1))
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

class WiFiDataset(Dataset):
    
    name = 'WiFiDataset'
    
    def __init__(self, dataset_path, uavs, transform=None, limit=None, labels=None, complex_input = False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.limit = limit
        self.labels = labels
        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            self.drone_ids = fp['labels'][:,0]
            
            mask = [self.drone_ids[i] in uavs for i in range(len(self.drone_ids))]

            if complex_input:
                self.data_i = np.real(fp['data'][mask])
                self.data_q = np.im(fp['data'][mask])
            else:
                self.data_i = fp['data_i'][mask]
                self.data_q = fp['data_q'][mask]
            
            self.drone_ids = self.drone_ids[mask]

            self.data_length = sum(mask)
    
    def __len__(self):
        if self.limit:
            return self.limit
        else:
            return self.data_length

    def __getitem__(self, idx):
        
        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            sample_i = self.data_i[idx]
            sample_q = self.data_q[idx]
            label = self.drone_ids[idx]
            
            sample = np.concatenate([sample_i, sample_q])
            sample = sample.reshape((2,-1))
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
            
    
        

