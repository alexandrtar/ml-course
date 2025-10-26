import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MedicalDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), augment=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment
        
    def __len__(self):
        return 100  # Для тестирования
    
    def __getitem__(self, idx):
        # Создаем синтетические данные для теста
        image = np.random.rand(*self.image_size).astype(np.float32)
        mask = np.random.randint(0, 2, self.image_size).astype(np.float32)
        
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image_tensor, mask_tensor