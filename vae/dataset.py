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


class MaskedSpectrogramDataset(Dataset):
    def __init__(self, unmasked_dir, masked_dir, train=True, train_test_split=0.8):
        self.unmasked_dir = unmasked_dir
        self.masked_dir = masked_dir
        unmasked_file_names = sorted(os.listdir(unmasked_dir))
        masked_file_names = sorted(os.listdir(masked_dir))          # sorted so that they have the same indexing order

        split = int(train_test_split * len(unmasked_file_names))
        if train:
            self.unmasked_file_names = unmasked_file_names[:split]
            self.masked_file_names = masked_file_names[:split]
        else:
            self.unmasked_file_names = unmasked_file_names[split:]
            self.masked_file_names = masked_file_names[split:]

    def __len__(self):
        return len(self.unmasked_file_names)
    
    def __getitem__(self, idx):
        unmasked_path = os.path.join(self.unmasked_dir, self.unmasked_file_names[idx])
        masked_path = os.path.join(self.masked_dir, self.masked_file_names[idx])
        spectrogram = np.load(unmasked_path)
        spectrogram = torch.from_numpy(spectrogram)
        masked_spectrogram = np.load(masked_path)
        masked_spectrogram = torch.from_numpy(masked_spectrogram)

        return (masked_spectrogram, spectrogram)