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

class DronesDataset(Dataset):
    """
    Soruce: Caron, Mathilde, et al. "Deep clustering for unsupervised learning of visual features." Proceedings of the European conference on computer vision (ECCV). 2018.
    """
    def __init__(self, filedir: str, input_size: int = 1024, devices=None, bursts=None, heights=None, 
                 return_indices=False, transform_to_2d=None):
        self.transform_to_2d = transform_to_2d
        self.return_indices = return_indices

        files_list = list(glob.glob(f"{filedir}*.bin"))
        self.data = []


        for file_name in files_list:
            json_file_name = f"{file_name[:-3]}json"

            try:
                with open(json_file_name, 'r') as f:
                    json_file = json.load(f)
            except Exception as e:
                print(f"Error reading {json_file_name}: {e}")
                continue

            burst_number = int(file_name.split('_')[2][5:])
            uav = int(json_file['annotations']['transmitter']['core:UAV'][3:])
            height = int(json_file['annotations']['core:distance'][:-2])

            if (devices and uav not in devices) or (bursts and burst_number not in bursts) or (heights and height not in heights):
                continue

            with open(file_name, 'rb') as f:
                cur_data = np.frombuffer(f.read(), np.float16)
                data_i = cur_data[::2]
                data_q = cur_data[1::2]
                cut_len = len(data_i) // input_size * input_size

                data_i = data_i[:cut_len]
                data_q = data_q[:cut_len]

                for i in range(cut_len // input_size):
                    self.data.append((np.concatenate(
                        [
                        data_i[i * input_size:(i + 1) * input_size][np.newaxis, :],
                        data_q[i * input_size:(i + 1) * input_size][np.newaxis, :]
                        ]), uav))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, uav = self.data[idx]
        
        if self.transform_to_2d:
            wavelet = self.transform_to_2d['wavelet']
            scales = self.transform_to_2d['scales']
            resize_to = self.transform_to_2d['resize_to']

            data = data[0] + 1j * data[1]
            coefs, _ = pywt.cwt(data, np.arange(1, scales + 1), wavelet)
            data = np.stack([coefs.real, coefs.imag], axis=0)
            data = transforms.Resize(size=resize_to)(torch.tensor(data, dtype=torch.float32))

        return (data, uav, idx) if self.return_indices else (data, uav)


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
    def __init__(self, file: str, devices=None, days=None, selected_receivers=None, return_indices=False, transform_to_2d=None):
        with open(file, 'rb') as f:
            self.data = pickle.load(f)

        self.transform_to_2d = transform_to_2d
        self.return_indices = return_indices
        self.devices = devices or list(range(len(self.data['tx_list'])))
        self.selected_receivers = selected_receivers or list(range(len(self.data['rx_list'])))
        self.selected_days = days or list(range(len(self.data['capture_date_list'])))

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

        sample = sample.astype(np.float32)
        
        if self.transform_to_2d:
            wavelet = self.transform_to_2d['wavelet']
            scales = self.transform_to_2d['scales']
            resize_to = self.transform_to_2d['resize_to']

            sample = sample[0] + 1j * sample[1]
            coefs, _ = pywt.cwt(sample, np.arange(1, scales + 1), wavelet)
            sample = np.stack([coefs.real, coefs.imag], axis=0)
            sample = transforms.Resize(size=resize_to)(torch.tensor(sample, dtype=torch.float32))

        return (sample, tx, idx) if self.return_indices else (sample, tx)
        
    
            
    
        

