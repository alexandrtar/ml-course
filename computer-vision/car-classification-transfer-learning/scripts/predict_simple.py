#!/usr/bin/env python3
"""
Простой скрипт для предсказаний с обученной моделью
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def load_model(model_path, num_classes=4, device='cuda'):
    """Загрузка обученной модели"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform, device, class_names):
    """Предсказание для одного изображения"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    return {
        'class': class_names[prediction.item()],
        'class_idx': prediction.item(),
        'confidence': confidence.item(),
        'probabilities': probabilities.cpu().numpy()[0]
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загрузка модели
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    model = load_model(model_path, num_classes=4, device=device)
    print("Model loaded successfully!")
    
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    class_names = ['0', '1', '2', '3']  # Замените на ваши реальные названия классов
    
    # Предсказания для тестовой директории
    test_dir = "test_upload"
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found")
        return
    
    # Собираем все изображения
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Found {len(image_files)} images in {test_dir}")
    
    # Создаем предсказания
    predictions = []
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        result = predict_image(model, image_path, transform, device, class_names)
        
        predictions.append({
            'file': image_file,
            'prediction': result['class_idx'],
            'confidence': result['confidence']
        })
    
    # Сохраняем результаты
    with open('final_predictions_simple.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred['prediction']}\n")
    
    # Анализ результатов
    pred_classes = [p['prediction'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    print(f"\n📊 Prediction Summary:")
    print(f"Total predictions: {len(predictions)}")
    
    for class_idx in range(4):
        count = pred_classes.count(class_idx)
        percentage = (count / len(predictions)) * 100
        avg_conf = np.mean([c for p, c in zip(pred_classes, confidences) if p == class_idx])
        print(f"Class {class_idx}: {count:4d} samples ({percentage:5.1f}%), avg confidence: {avg_conf:.3f}")
    
    print(f"\nOverall average confidence: {np.mean(confidences):.3f}")
    print(f"Predictions saved to: final_predictions_simple.txt")
    
    # Визуализация распределения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(pred_classes, bins=4, alpha=0.7, edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.xticks(range(4))
    
    plt.subplot(1, 2, 2)
    plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()