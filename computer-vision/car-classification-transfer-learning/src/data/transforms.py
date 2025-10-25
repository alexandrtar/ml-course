import albumentations as A
import albumentations.pytorch as albu_pytorch
from torchvision import transforms
import numpy as np

def get_advanced_transforms(config):
    """Создание продвинутых трансформаций с albumentations (исправленная версия)"""
    
    image_size = config['data']['image_size']
    aug_config = config['augmentation']['train']
    
    train_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=aug_config['horizontal_flip']),
        A.Rotate(limit=aug_config['rotation'], p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=15, 
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=aug_config['brightness_contrast']
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=aug_config['hue_saturation']
        ),
        # Убраны проблемные трансформации
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        albu_pytorch.ToTensorV2()
    ])
    
    test_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        albu_pytorch.ToTensorV2()
    ])
    
    return train_transform, test_transform

def get_basic_transforms(config):
    """Базовые трансформации (как в оригинальном ноутбуке)"""
    
    image_size = config['data']['image_size']
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

class AlbumentationsDataset:
    """Dataset для работы с albumentations (исправленный)"""
    
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        image = np.array(Image.open(self.file_paths[idx]).convert('RGB'))
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label