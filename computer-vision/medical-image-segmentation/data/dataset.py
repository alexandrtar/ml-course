import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .augmentations import MedicalAugmentations


class MedicalDataset(Dataset):
    """
    Dataset for medical image segmentation with comprehensive preprocessing
    and augmentations
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        normalize: bool = True,
        target_transform: Optional[Callable] = None,
        synthetic: bool = False
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.target_transform = target_transform
        self.synthetic = synthetic
        
        # Paths
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        
        # Get file lists
        self.image_files = self._get_valid_files(self.image_dir)
        self.mask_files = self._get_valid_files(self.mask_dir)
        
        # Verify matching files
        self._validate_files()
        
        # Augmentations
        self.augmentations = MedicalAugmentations(image_size) if augment else None
        self.preprocessor = MedicalPreprocessor(image_size, normalize)
        
        print(f"📊 Loaded {len(self.image_files)} samples from {data_dir}")
    
    def _get_valid_files(self, directory: str) -> list:
        """Get list of valid image files"""
        if not os.path.exists(directory):
            if self.synthetic:
                return []
            raise FileNotFoundError(f"Directory {directory} does not exist")
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        files = [f for f in os.listdir(directory) 
                if os.path.splitext(f)[1].lower() in valid_extensions]
        return sorted(files)
    
    def _validate_files(self):
        """Validate that image and mask files match"""
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(
                f"Image count ({len(self.image_files)}) doesn't match "
                f"mask count ({len(self.mask_files)})"
            )
        
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_name = os.path.splitext(img_file)[0]
            mask_name = os.path.splitext(mask_file)[0]
            if img_name != mask_name:
                raise ValueError(f"File mismatch: {img_file} vs {mask_file}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Load image and mask
            image = self._load_image(idx)
            mask = self._load_mask(idx)
            
            # Apply augmentations
            if self.augment and self.augmentations:
                augmented = self.augmentations(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            else:
                # Just preprocessing without augmentation
                image = self.preprocessor.preprocess_image(image)
                mask = self.preprocessor.preprocess_mask(mask)
            
            # Apply target transform if provided
            if self.target_transform:
                mask = self.target_transform(mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data to avoid breaking training
            return self._create_dummy_sample()
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load image with proper error handling"""
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        try:
            # Try different loading strategies
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                # Try with PIL as fallback
                from PIL import Image
                image = np.array(Image.open(img_path).convert('L'))
            
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
                
            return image
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
    
    def _load_mask(self, idx: int) -> np.ndarray:
        """Load mask with proper error handling"""
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                from PIL import Image
                mask = np.array(Image.open(mask_path).convert('L'))
            
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            
            # Binarize mask
            mask = (mask > 0.5).astype(np.float32)
            return mask
            
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            raise
    
    def _create_dummy_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy sample for error recovery"""
        image = np.random.rand(*self.image_size).astype(np.float32)
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        image = self.preprocessor.preprocess_image(image)
        mask = self.preprocessor.preprocess_mask(mask)
        
        return image, mask
    
    def get_sample_info(self, idx: int) -> dict:
        """Get information about a specific sample"""
        return {
            'image_file': self.image_files[idx],
            'mask_file': self.mask_files[idx],
            'image_path': os.path.join(self.image_dir, self.image_files[idx]),
            'mask_path': os.path.join(self.mask_dir, self.mask_files[idx])
        }


class MedicalDataLoader:
    """Convenience class for creating data loaders"""
    
    @staticmethod
    def create_loaders(
        train_dir: str,
        val_dir: str,
        batch_size: int = 8,
        image_size: Tuple[int, int] = (256, 256),
        num_workers: int = 4,
        **kwargs
    ):
        """Create train and validation data loaders"""
        
        train_dataset = MedicalDataset(
            data_dir=train_dir,
            image_size=image_size,
            augment=True,
            **kwargs
        )
        
        val_dataset = MedicalDataset(
            data_dir=val_dir, 
            image_size=image_size,
            augment=False,
            **kwargs
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader