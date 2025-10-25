import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
import numpy as np


class TestDataset(Dataset):
    """Dataset для тестовых данных (без меток)"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, -1  # -1 как фиктивная метка


def create_dataloaders(config, use_albumentations=False):
    """Создание DataLoader'ов для обучения и тестирования"""

    # Определение путей
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    train_path = os.path.join(project_root, config['data']['train_path'])
    test_path = os.path.join(project_root, config['data']['test_path'])

    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")

    # Проверка существования путей
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train directory not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    # Выбор трансформаций
    if use_albumentations:
        try:
            from .transforms import get_advanced_transforms
            train_transform, test_transform = get_advanced_transforms(config)
            print("Using albumentations transforms")
        except ImportError:
            print("Albumentations not available, using basic transforms")
            from .transforms import get_basic_transforms
            train_transform, test_transform = get_basic_transforms(config)
    else:
        from .transforms import get_basic_transforms
        train_transform, test_transform = get_basic_transforms(config)
        print("Using basic transforms")

    # Загрузка train данных
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)

    # Определение количества классов
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_dataset.classes}")

    # Разделение на train/validation
    val_size = int(config['data']['validation_split'] * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Создание test dataset
    test_dataset = TestDataset(test_path, transform=test_transform)

    # Создание DataLoader'ов
    batch_size = config['experiment']['batch_size']
    num_workers = min(config['experiment']['num_workers'], 4)  # Ограничиваем для стабильности

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_classes': num_classes,
        'class_names': train_dataset.dataset.classes
    }

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    return dataloaders