import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any


class MedicalAugmentations:
    """
    Comprehensive augmentations for medical images
    """
    
    def __init__(self, image_size: tuple = (256, 256)):
        self.image_size = image_size
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """Build augmentation pipeline"""
        return A.Compose([
            # Spatial transformations
            A.OneOf([
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    alpha_affine=10,
                    p=0.3
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.2,
                    p=0.3
                ),
                A.OpticalDistortion(
                    distort_limit=0.2,
                    shift_limit=0.1,
                    p=0.3
                ),
            ], p=0.5),
            
            # Flip and rotate
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.5),
            
            # Color transformations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.5),
            ], p=0.5),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    per_channel=True,
                    p=0.3
                ),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.5),
            
            # Morphological operations
            A.OneOf([
                A.Erosion(p=0.2),
                A.Dilation(p=0.2),
            ], p=0.3),
            
            # Always resize to target size
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            
            # Normalize and convert to tensor
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Apply augmentations to image and mask"""
        transformed = self.transform(image=image, mask=mask)
        return transformed


class MedicalPreprocessor:
    """Preprocessing for medical images without augmentation"""
    
    def __init__(self, image_size: tuple = (256, 256), normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        transforms = [
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
        ]
        
        if self.normalize:
            transforms.extend([
                A.Normalize(mean=[0.5], std=[0.5]),
            ])
        
        transforms.append(ToTensorV2())
        return A.Compose(transforms)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess single image"""
        transformed = self.transform(image=image)
        return transformed['image']
    
    def preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Preprocess single mask"""
        # For masks, we don't normalize
        transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            ToTensorV2(),
        ])
        transformed = transform(image=mask)
        return transformed['image']