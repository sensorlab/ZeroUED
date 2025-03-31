import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle

class DronesDataset(Dataset):
    
    name = 'DronesDataset'
    
    def __init__(self, dataset_path, uavs=None, bursts=None,heights =None, transform=None, limit=None, complex_input = False, return_indices=False, limit_samples_per_device = None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.limit = limit
        self.sigma = 7239
        self.mu = 1.5
        self.return_indices = return_indices

        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            self.drone_ids = fp['labels'][:,0]
            self.dists = fp['labels'][:,1]
            self.bursts = fp['labels'][:,2]
            
            mask = [self.drone_ids[i] in uavs and self.bursts[i] in bursts and self.dists[i] in heights  for i in range(len(self.drone_ids))]

            if complex_input:
                self.data_i = np.real(fp['data'][mask])
                self.data_q = np.imag(fp['data'][mask])
            else:
                self.data_i = fp['data_i'][mask]
                self.data_q = fp['data_q'][mask]
            
            
            self.drone_ids = np.array(self.drone_ids[mask])
            self.dists = self.dists[mask]
            self.bursts = self.bursts[mask]
            self.data_length = sum(mask)
            
        if limit_samples_per_device is not None:
            mask = np.zeros(self.data_length, dtype = bool)
            ids = np.arange(self.data_length)
            for i in uavs:
                if sum(self.drone_ids == i) > limit_samples_per_device:
                    cur_ids = ids[self.drone_ids == i]
                    cur_ids = np.random.choice(cur_ids, size = limit_samples_per_device, replace = False)
                    mask[cur_ids] = True
                else:
                    cur_ids = ids[self.drone_ids == i]
                    mask[cur_ids] = True
                    

            self.drone_ids = self.drone_ids[mask]
            self.dists = self.dists[mask]
            self.bursts = self.bursts[mask]
            self.data_length = sum(mask)
            self.data_i = self.data_i[mask]
            self.data_q = self.data_q[mask]
        
    
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

        if self.return_indices:
            return sample, label, idx
        else:
            return sample, label

class WiFiDataset(Dataset):
    
    name = 'WiFiDataset'
    
    def __init__(self, dataset_path, uavs, transform=None, limit=None, complex_input = False, return_indices=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.limit = limit
        self.return_indices = return_indices
        
        with h5py.File(self.dataset_path, mode="r", swmr=True) as fp:
            self.drone_ids = fp['labels'][:,0]
            
            mask = [self.drone_ids[i] in uavs for i in range(len(self.drone_ids))]

            if complex_input:
                self.data_i = np.real(fp['data'][mask])
                self.data_q = np.imag(fp['data'][mask])
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
            
        if self.return_indices:
            return sample, label, idx
        else:
            return sample, label


class LorRaDataset(
   Dataset
):
    def __init__(self, filedir:str, slice_length:int=1024, selected_uavs=None, selected_days=None, return_indices=False, max_smaples_per_transmitions=400000):
        
        self.selected_uavs = selected_uavs
        self.selected_days = selected_days
        self.filedir = filedir
        self.total_days = len(selected_days)
        self.total_devices = len(selected_uavs)
        self.total_transmittions = 1
        self.transmittion_length = max_smaples_per_transmitions
        self.slice_length = slice_length
        self.total_len = self.total_days * self.total_devices * self.total_transmittions * self.transmittion_length // slice_length
        self.return_indices = return_indices
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        cur_div = self.total_devices * self.total_transmittions * self.transmittion_length // self.slice_length
        day = idx // cur_div
        idx = idx % cur_div

        cur_div = self.total_transmittions * self.transmittion_length // self.slice_length
        device = idx // cur_div
        idx = idx % cur_div

        cur_div = self.transmittion_length // self.slice_length
        transmission = idx // cur_div
        idx = idx % cur_div

        slice = idx

        day = self.selected_days[day]
        device = self.selected_uavs[device]
        print(slice)
        with open(f"{self.filedir}/Day_{day}/device_{device}/trans_{transmission+1}.dat", 'rb') as f:
            data = np.frombuffer(b''.join([line for line in f]), np.float32)
            data = data[slice * self.slice_length: (slice+1) * self.slice_length]
            data_i = data[::2]
            data_q = data[1::2]
            sample = np.concatenate([data_i, data_q])
            sample = sample.reshape((2,-1))

        if self.return_indices:
            return sample, device, idx
        else:
            return sample, device


class WiSig_Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                  file:str, 
                  selected_uavs=None, 
                  selected_days=None, 
                  selected_recivers=None, 
                  return_indices=False,
                  transforms = None):
         
        with open(file, 'rb') as f:
                self.data = pickle.load(f)
            
        self.return_indices = return_indices
        self.transforms = transforms

        self.num_transmitters = len(self.data['tx_list'])
        self.num_recivers = len(self.data['rx_list'])
        self.num_dates = len(self.data['capture_date_list'])
         
        if selected_uavs is not None:
            self.selected_uavs = selected_uavs
        else:
            self.selected_uavs = [i for i in range(self.num_transmitters)]

        if selected_recivers is not None:
            self.selected_recivers = selected_recivers
        else:
            self.selected_recivers = [i for i in range(self.num_recivers)]

        if selected_days is not None:
            self.selected_days = selected_days
        else:
            self.selected_days = [i for i in range(self.num_dates)]

        self.data_ordered = []

        for i in self.selected_uavs:
            for j in self.selected_recivers:
                for m in self.selected_days:
                    for k in range(len(self.data['data'][i][j][m][1])):
                        self.data_ordered.append(
                            (self.data['data'][i][j][m][1][k],
                            i)
                        )
    
    def __len__(self):
        return len(self.data_ordered)

    def __getitem__(self, idx):
        sample, tx = self.data_ordered[idx]

        sample = sample.T

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.return_indices:
            return sample, tx, idx
        else:
            return sample, tx

        
    
            
    
        

