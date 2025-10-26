import pytest
import torch
import numpy as np
from pathlib import Path

from medical_image_segmentation.data.dataset import MedicalDataset
from medical_image_segmentation.data.synthetic_data import MedicalDataGenerator
from medical_image_segmentation.data.augmentations import MedicalAugmentations


class TestMedicalDataset:
    """Test MedicalDataset class"""
    
    def test_dataset_initialization(self, synthetic_dataset):
        """Test dataset initialization"""
        dataset = MedicalDataset(
            data_dir=synthetic_dataset,
            image_size=(256, 256),
            augment=False
        )
        
        assert len(dataset) == 10
        assert dataset.image_size == (256, 256)
        assert not dataset.augment
    
    def test_dataset_getitem(self, synthetic_dataset):
        """Test dataset item access"""
        dataset = MedicalDataset(
            data_dir=synthetic_dataset,
            image_size=(128, 128),
            augment=False
        )
        
        image, mask = dataset[0]
        
        # Check types and shapes
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape == (1, 128, 128)  # (C, H, W)
        assert mask.shape == (1, 128, 128)
        
        # Check value ranges
        assert image.min() >= -1.0 and image.max() <= 1.0  # Normalized
        assert mask.min() >= 0.0 and mask.max() <= 1.0  # Binary
    
    def test_dataset_augmentations(self, synthetic_dataset):
        """Test dataset with augmentations"""
        dataset = MedicalDataset(
            data_dir=synthetic_dataset,
            image_size=(256, 256),
            augment=True
        )
        
        image, mask = dataset[0]
        
        assert image.shape == (1, 256, 256)
        assert mask.shape == (1, 256, 256)
    
    def test_dataset_sample_info(self, synthetic_dataset):
        """Test sample information retrieval"""
        dataset = MedicalDataset(
            data_dir=synthetic_dataset,
            image_size=(256, 256),
            augment=False
        )
        
        info = dataset.get_sample_info(0)
        
        assert 'image_file' in info
        assert 'mask_file' in info
        assert 'image_path' in info
        assert 'mask_path' in info
        assert 'sample_0' in info['image_file']


class TestMedicalDataGenerator:
    """Test MedicalDataGenerator class"""
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        generator = MedicalDataGenerator(image_size=(256, 256))
        
        assert generator.image_size == (256, 256)
        assert generator.rng is not None
    
    def test_organ_shape_generation(self):
        """Test organ shape generation"""
        generator = MedicalDataGenerator(image_size=(256, 256))
        
        center = (128, 128)
        size = (30, 40)
        organ_mask = generator.generate_organ_shape(center, size)
        
        assert organ_mask.shape == (256, 256)
        assert organ_mask.dtype == np.float32
        assert organ_mask.min() >= 0.0
        assert organ_mask.max() <= 1.0
    
    def test_medical_image_generation(self):
        """Test medical image generation"""
        generator = MedicalDataGenerator(image_size=(128, 128))
        
        image, mask = generator.generate_medical_image()
        
        assert image.shape == (128, 128)
        assert mask.shape == (128, 128)
        assert image.dtype == np.float64
        assert mask.dtype == np.float32
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    
    def test_dataset_generation(self, tmp_path):
        """Test complete dataset generation"""
        generator = MedicalDataGenerator(image_size=(64, 64))
        
        output_dir = tmp_path / "generated_data"
        generator.generate_dataset(output_dir, num_samples=10)
        
        # Check if directories are created
        assert (output_dir / "train" / "images").exists()
        assert (output_dir / "train" / "masks").exists()
        assert (output_dir / "val" / "images").exists()
        assert (output_dir / "val" / "masks").exists()
        assert (output_dir / "test" / "images").exists()
        assert (output_dir / "test" / "masks").exists()
        
        # Check if files are created
        train_images = list((output_dir / "train" / "images").iterdir())
        assert len(train_images) > 0


class TestMedicalAugmentations:
    """Test MedicalAugmentations class"""
    
    def test_augmentations_initialization(self):
        """Test augmentations initialization"""
        augmentations = MedicalAugmentations(image_size=(256, 256))
        
        assert augmentations.image_size == (256, 256)
        assert augmentations.transform is not None
    
    def test_augmentations_application(self):
        """Test augmentations application"""
        augmentations = MedicalAugmentations(image_size=(128, 128))
        
        # Create sample image and mask
        image = np.random.rand(256, 256).astype(np.float32)
        mask = np.random.randint(0, 2, (256, 256)).astype(np.float32)
        
        # Apply augmentations
        transformed = augmentations(image=image, mask=mask)
        
        assert 'image' in transformed
        assert 'mask' in transformed
        assert transformed['image'].shape == (1, 128, 128)  # After ToTensorV2
        assert transformed['mask'].shape == (1, 128, 128)
    
    def test_preprocessor(self):
        """Test medical preprocessor"""
        from medical_image_segmentation.data.augmentations import MedicalPreprocessor
        
        preprocessor = MedicalPreprocessor(image_size=(128, 128), normalize=True)
        
        image = np.random.rand(256, 256).astype(np.float32)
        processed_image = preprocessor.preprocess_image(image)
        
        assert processed_image.shape == (1, 128, 128)
        assert isinstance(processed_image, torch.Tensor)


def test_dataloader_creation(synthetic_dataset):
    """Test data loader creation"""
    from medical_image_segmentation.data.dataset import MedicalDataLoader
    
    train_loader, val_loader = MedicalDataLoader.create_loaders(
        train_dir=synthetic_dataset,
        val_dir=synthetic_dataset,
        batch_size=2,
        image_size=(128, 128),
        num_workers=0  # Use 0 workers for tests
    )
    
    # Check loaders
    assert train_loader is not None
    assert val_loader is not None
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2
    
    # Check one batch
    for images, masks in train_loader:
        assert images.shape == (2, 1, 128, 128)  # (batch, channels, height, width)
        assert masks.shape == (2, 1, 128, 128)
        break