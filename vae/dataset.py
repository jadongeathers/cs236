import numpy as np
import os
import torch

from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, train=True, train_test_split=0.8):
        self.data_dir = data_dir
        file_paths = os.listdir(data_dir)

        split = int(train_test_split * len(file_paths))
        if train:
            self.file_paths = file_paths[:split]
        else:
            self.file_paths = file_paths[split:]

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_paths[idx])
        spectrogram = np.load(file_path)
        spectrogram = torch.from_numpy(spectrogram)      
        return spectrogram