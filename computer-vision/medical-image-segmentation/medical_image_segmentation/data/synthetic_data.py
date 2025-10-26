import os
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


class MedicalDataGenerator:
    """
    Generate synthetic medical data for segmentation tasks
    Creates realistic-looking medical images with various structures
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.rng = np.random.default_rng(42)
    
    def generate_organ_shape(self, center: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
        """Generate realistic organ-like shape"""
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Create ellipse with some distortion
        y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
        
        # Add some random distortion to make it more organic
        distortion = self.rng.normal(1.0, 0.1, 2)
        rx, ry = size[0] * distortion[0], size[1] * distortion[1]
        
        # Ellipse equation with noise
        ellipse_mask = ((x - center[0]) / rx)**2 + ((y - center[1]) / ry)**2 <= 1
        
        # Add some texture variations
        texture = self.rng.normal(0.8, 0.2, ellipse_mask.shape)
        mask[ellipse_mask] = np.clip(texture[ellipse_mask], 0.3, 1.0)
        
        return mask
    
    def generate_tumor_like_shape(self, base_mask: np.ndarray) -> np.ndarray:
        """Add tumor-like structures to organ mask"""
        tumor_mask = np.zeros_like(base_mask)
        
        # Find organ boundaries
        coords = np.argwhere(base_mask > 0.1)
        if len(coords) == 0:
            return tumor_mask
        
        # Add several tumor-like structures
        num_tumors = self.rng.integers(1, 4)
        
        for _ in range(num_tumors):
            # Random location within organ
            center_idx = self.rng.choice(len(coords))
            center = coords[center_idx]
            
            # Tumor size
            size = self.rng.integers(5, 20)
            
            # Create circular tumor
            y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
            tumor = ((x - center[1]) / size)**2 + ((y - center[0]) / size)**2 <= 1
            
            # Add to tumor mask with higher intensity
            tumor_mask[tumor] = self.rng.uniform(0.7, 1.0)
        
        return tumor_mask
    
    def generate_medical_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic medical image and corresponding mask"""
        # Base image with background tissue texture
        image = self.rng.normal(0.4, 0.1, self.image_size)
        
        # Create several organs
        num_organs = self.rng.integers(2, 5)
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        for _ in range(num_organs):
            # Random organ parameters
            center = (
                self.rng.integers(30, self.image_size[0] - 30),
                self.rng.integers(30, self.image_size[1] - 30)
            )
            size = (
                self.rng.integers(20, 60),
                self.rng.integers(25, 70)
            )
            
            # Generate organ shape
            organ_mask = self.generate_organ_shape(center, size)
            
            # Add organ to image with varying intensity
            organ_intensity = self.rng.normal(0.3, 0.1)
            image[organ_mask > 0] += organ_intensity * organ_mask[organ_mask > 0]
            
            # Add to segmentation mask
            mask[organ_mask > 0.3] = 1.0
            
            # Possibly add tumors
            if self.rng.random() < 0.3:
                tumor_mask = self.generate_tumor_like_shape(organ_mask)
                image[tumor_mask > 0] += 0.4  # Tumors are brighter
                mask[tumor_mask > 0] = 1.0
        
        # Add noise and artifacts
        image = self.add_medical_artifacts(image)
        
        # Normalize image
        image = np.clip(image, 0, 1)
        
        return image, mask
    
    def add_medical_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add realistic medical imaging artifacts"""
        # Gaussian noise
        noise = self.rng.normal(0, 0.02, image.shape)
        image = image + noise
        
        # Add scan lines (MRI-like artifacts)
        if self.rng.random() < 0.2:
            line_intensity = self.rng.uniform(0.1, 0.3)
            line_pos = self.rng.integers(0, self.image_size[0])
            line_width = self.rng.integers(1, 3)
            image[line_pos:line_pos+line_width, :] += line_intensity
        
        # Add blur in some regions (motion artifacts)
        if self.rng.random() < 0.3:
            blur_region = self.rng.integers(0, self.image_size[0] - 50, size=2)
            region = image[blur_region[0]:blur_region[0]+50, 
                          blur_region[1]:blur_region[1]+50]
            image[blur_region[0]:blur_region[0]+50,
                  blur_region[1]:blur_region[1]+50] = cv2.GaussianBlur(region, (5, 5), 0)
        
        return image
    
    def generate_dataset(
        self, 
        output_dir: str, 
        num_samples: int = 1000,
        splits: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    ):
        """Generate complete dataset with train/val/test splits"""
        
        train_ratio, val_ratio, test_ratio = splits
        
        # Create directories
        splits_info = {
            'train': int(num_samples * train_ratio),
            'val': int(num_samples * val_ratio),
            'test': int(num_samples * test_ratio)
        }
        
        for split in splits_info.keys():
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
        
        # Generate samples
        sample_count = 0
        for split, split_count in splits_info.items():
            print(f"Generating {split} data...")
            
            for i in tqdm(range(split_count)):
                image, mask = self.generate_medical_image()
                
                # Save image and mask
                img_path = os.path.join(output_dir, split, 'images', f'sample_{sample_count:05d}.png')
                mask_path = os.path.join(output_dir, split, 'masks', f'sample_{sample_count:05d}.png')
                
                # Convert to uint8 for saving
                img_uint8 = (image * 255).astype(np.uint8)
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                cv2.imwrite(img_path, img_uint8)
                cv2.imwrite(mask_path, mask_uint8)
                
                sample_count += 1
        
        print(f"✅ Generated {sample_count} samples in {output_dir}")


def create_medical_dataset(
    output_dir: str = 'data/medical_data',
    num_samples: int = 1000,
    image_size: Tuple[int, int] = (256, 256)
):
    """Convenience function to create medical dataset"""
    generator = MedicalDataGenerator(image_size)
    generator.generate_dataset(output_dir, num_samples)