import os
import numpy as np
import cv2
from tqdm import tqdm

def create_medical_dataset(output_dir='data/medical_data', num_samples=1000, image_size=(256, 256)):
    """Create synthetic medical dataset for testing"""
    print("🔄 Creating synthetic medical dataset...")
    
    # Create directories
    splits = ['train', 'val', 'test']
    split_ratios = [0.7, 0.2, 0.1]  # 70% train, 20% val, 10% test
    
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
    
    # Create samples
    sample_count = 0
    for split, ratio in zip(splits, split_ratios):
        split_count = int(num_samples * ratio)
        print(f"Creating {split} split with {split_count} samples...")
        
        for i in tqdm(range(split_count)):
            # Create more realistic synthetic medical image
            image = np.random.normal(0.4, 0.15, image_size)
            
            # Add some organ-like structures (ellipses)
            num_organs = np.random.randint(2, 5)
            mask = np.zeros(image_size, dtype=np.float32)
            
            for _ in range(num_organs):
                center = (
                    np.random.randint(30, image_size[0]-30),
                    np.random.randint(30, image_size[1]-30)
                )
                radius_x = np.random.randint(20, 50)
                radius_y = np.random.randint(25, 60)
                
                y, x = np.ogrid[:image_size[0], :image_size[1]]
                organ_mask = ((x - center[0]) / radius_x)**2 + ((y - center[1]) / radius_y)**2 <= 1
                
                # Add organ to image with varying intensity
                image[organ_mask] += np.random.normal(0.2, 0.05)
                mask[organ_mask] = 1.0
            
            # Clip and normalize
            image = np.clip(image, 0, 1)
            
            # Save files
            img_path = os.path.join(output_dir, split, 'images', f'sample_{sample_count:05d}.png')
            mask_path = os.path.join(output_dir, split, 'masks', f'sample_{sample_count:05d}.png')
            
            cv2.imwrite(img_path, (image * 255).astype(np.uint8))
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            
            sample_count += 1
    
    print(f"✅ Created {sample_count} samples in {output_dir}")