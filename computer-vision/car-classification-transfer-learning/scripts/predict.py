#!/usr/bin/env python3
"""
Скрипт для предсказаний с обученными моделями
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.utils.config import load_config
from src.data.dataloaders import TestDataset
from src.models.custom_cnn import ImprovedCarCNN
from src.models.resnet_wrapper import AdvancedResNetWrapper
from src.utils.visualization import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

def load_trained_model(model_path, model_type, num_classes, device):
    """Загрузка обученной модели"""
    if model_type == 'custom_cnn':
        model = ImprovedCarCNN(num_classes)
    elif model_type == 'simple_cnn':
        from src.models.custom_cnn import SimpleCarCNN
        model = SimpleCarCNN(num_classes)
    else:
        # Transfer learning модели
        wrapper = AdvancedResNetWrapper(model_type, num_classes, pretrained=False)
        model = wrapper.get_model()
    
    # Загрузка весов
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def make_predictions(model, test_loader, device):
    """Создание предсказаний"""
    predictions = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    return predictions

def main():
    """Основная функция предсказаний"""
    config = load_config()
    device = torch.device(config['experiment']['device'])
    
    # Загрузка тестовых данных
    from src.data.transforms import get_basic_transforms
    _, test_transform = get_basic_transforms(config)
    
    test_dataset = TestDataset(config['data']['test_path'], transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['experiment']['batch_size'],
        shuffle=False
    )
    
    # Загрузка модели (пример с лучшей моделью)
    model_path = "checkpoints/best_resnet18.pth"  # Измените на вашу лучшую модель
    model_type = "resnet18"  # Измените на тип вашей модели
    
    print(f"Loading model: {model_type}")
    model = load_trained_model(model_path, model_type, num_classes=10, device=device)
    
    # Предсказания
    print("Making predictions...")
    predictions = make_predictions(model, test_loader, device)
    
    # Сохранение предсказаний
    with open('final_predictions.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"Predictions saved to 'final_predictions.txt'")
    print(f"Total predictions: {len(predictions)}")
    
    # Анализ распределения предсказаний
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples ({count/len(predictions):.1%})")
    
    # Визуализация нескольких примеров
    print("\nVisualizing sample predictions...")
    visualize_sample_predictions(model, test_dataset, device, predictions, num_samples=10)

def visualize_sample_predictions(model, test_dataset, device, predictions, num_samples=10):
    """Визуализация примеров предсказаний"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(test_dataset))):
        image, _ = test_dataset[i]
        
        # Денормализация для отображения
        image_np = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        axes[i].imshow(image_np)
        axes[i].set_title(f'Pred: {predictions[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()