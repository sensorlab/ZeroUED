import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import pywt
from torchvision import transforms
import torchvision
import glob
import json

class OracleDataset(Dataset):
    def __init__(self, file: str, input_size: int = 256, devices=None, 
                 return_indices=False, type = 'train', modality = 'iq_const'):
        self.return_indices = return_indices
        self.modality = modality
        self.devices_map = [10,  4,  6,  0, 15,  1,  9,  2, 11,  3, 12,  8,  5,  7, 13, 14]

        
        with h5py.File(file) as f:
            self.data_i = np.array(f['data_i']).astype(np.float32)
            self.data_q = np.array(f['data_q']).astype(np.float32)
            self.labels = np.array(f['labels'][:,0]).astype(int)
            self.data_i_saved = self.data_i.reshape(-1,256)
            self.data_q_saved = self.data_q.reshape(-1,256)
            self.labels_saved = self.labels

        if type == 'validation':
            self.data_i = self.data_i.reshape(16,2, 2000,256)[:,:,-400:].reshape(-1,256)
            self.data_q = self.data_q.reshape(16,2, 2000,256)[:,:,-400:].reshape(-1,256)
            self.labels = self.labels.reshape(16,2, 2000)[:,:,-400:].reshape(-1)
        else:
            self.data_i = self.data_i.reshape(16,2, 2000,256)[:,:,:-400].reshape(-1,256)
            self.data_q = self.data_q.reshape(16,2, 2000,256)[:,:,:-400].reshape(-1,256)
            self.labels = self.labels.reshape(16,2, 2000)[:,:,:-400].reshape(-1)
            
        self.data_i = self.data_i[[i for i in range(len(self.labels)) if self.devices_map[self.labels[i]] in  devices]]
        self.data_q = self.data_q[[i for i in range(len(self.labels)) if self.devices_map[self.labels[i]] in  devices]]
        self.labels = self.labels[[i for i in range(len(self.labels)) if self.devices_map[self.labels[i]] in  devices]]
    
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.modality == 'iq_const':
            i_samples = self.data_i[idx] / np.abs(self.data_i_saved[self.labels_saved == label]).max()
            q_samples = self.data_q[idx] / np.abs(self.data_q_saved[self.labels_saved == label]).max()

            i_samples = (i_samples + 1) / 2 * 100
            q_samples = (q_samples + 1) / 2 * 100
            
            i_indices = i_samples.astype(int)
            q_indices = q_samples.astype(int)
            i_indices = np.clip(i_indices, 0, 99)
            q_indices = np.clip(q_indices, 0, 99)
            sample = np.zeros((1,100,100))
            for j in range(len(i_samples)):
                sample[0][i_indices[j], q_indices[j]] += 1
            #sample = sample / sample.max()
            sample = sample[:,20:80,20:80].astype(np.float32)
        else:
            data_q = self.data_q[idx] /  np.abs(self.data_i_saved[self.labels_saved == label]).max()
            data_i = self.data_i[idx] /  np.abs(self.data_q_saved[self.labels_saved == label]).max()
            sample = np.stack([data_i, data_q],axis = 0)
            sample = sample
        
        return (sample, self.devices_map[label], idx) if self.return_indices else (sample, self.devices_map[label])

class LoRaDataset(Dataset):
    """
    Soruce:  LoRa Device Fingerprinting in the Wild: Disclosing RF Data-Driven Fingerprint Sensitivity to Deployment
Variability. IEEE Access, pp: 142893–142909, October 2021.
    """
    def __init__(self, filedir: str, input_size: int = 1024, devices=None, selected_days=None, 
                 return_indices=False, transform_to_2d=None):
        self.devices = devices
        self.selected_days = selected_days
        self.filedir = filedir
        self.transform_to_2d = transform_to_2d
        self.return_indices = return_indices
        self.slice_length = input_size * 2
        self.total_days = len(selected_days)
        self.total_devices = len(devices)
        self.total_transmissions = 1
        self.transmission_length = 400000
        self.total_len = (self.total_days * self.total_devices * self.total_transmissions * self.transmission_length) // self.slice_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        cur_div = self.total_devices * self.total_transmissions * self.transmission_length // self.slice_length
        day = idx // cur_div
        idx %= cur_div

        cur_div = self.total_transmissions * self.transmission_length // self.slice_length
        device = idx // cur_div
        idx %= cur_div

        cur_div = self.transmission_length // self.slice_length
        transmission = idx // cur_div
        slice_idx = idx % cur_div

        with open(f"{self.filedir}/Day_{self.selected_days[day]}/device_{self.devices[device]}/trans_{transmission+1}.dat", 'rb') as f:
            data = np.frombuffer(f.read(), np.float32)
            data = data[slice_idx * self.slice_length: (slice_idx + 1) * self.slice_length]
            data_i = data[::2]
            data_q = data[1::2]
            sample = np.stack([data_i, data_q], axis=0)

        device = self.devices[device]
        return (sample, device, idx) if self.return_indices else (sample, device)


class WiSig_Dataset(Dataset):
    """
    Soure: S. Hanna, S. Karunaratne, and D. Cabric, “WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting,” IEEE Access, vol. 10, pp. 22808–22818, 2022, doi: 10.1109/ACCESS.2022.3154790.
    """
    def __init__(self, file: str, devices=None, days=None, selected_receivers=None, return_indices=False, transform_to_2d=None, train_test_split = False, type = 'train', polars_c = False, k_fold_samples = 0):
        with open(file, 'rb') as f:
            self.data = pickle.load(f)

        self.transform_to_2d = transform_to_2d
        self.return_indices = return_indices
        self.devices = devices or list(range(len(self.data['tx_list'])))
        self.selected_receivers = selected_receivers or list(range(len(self.data['rx_list'])))
        self.selected_days = days or list(range(len(self.data['capture_date_list'])))
        self.polars_c = polars_c

        if type == 'validation' and train_test_split:
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1][-200:]
            ]
            
            
        elif type == 'train' and train_test_split:
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1][:-200]
            ]
            
        else:
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1]
            ]

    def __len__(self):
        return len(self.data_ordered)


    def __getitem__(self, idx):
        sample, tx = self.data_ordered[idx]
        sample = sample.T
        
        sample_ = np.zeros(sample.shape)
        
        sample = sample / np.max(np.abs(sample))
        sample = sample.astype(np.float32)

        return (sample, tx, idx) if self.return_indices else (sample, tx)
