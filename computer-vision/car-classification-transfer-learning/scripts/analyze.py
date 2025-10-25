#!/usr/bin/env python3
"""
Скрипт для анализа результатов и создания предсказаний
"""

import sys
import os

# Добавляем корневую директорию в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image


def main():
    """Анализ результатов и создание предсказаний"""
    try:
        from src.utils.config import load_config
        from src.data.dataloaders import create_dataloaders
        from src.models.resnet_wrapper import AdvancedResNetWrapper
        from src.training.metrics import calculate_comprehensive_metrics, print_detailed_metrics
        from src.utils.visualization import plot_confusion_matrix

        # Загрузка конфигурации и данных
        config_path = os.path.join(project_root, 'configs', 'experiment_config.yaml')
        config = load_config(config_path)

        print("Loading data and model...")
        dataloaders = create_dataloaders(config, use_albumentations=False)

        # Загрузка обученной модели
        model = AdvancedResNetWrapper(
            model_name='resnet18',
            num_classes=dataloaders['num_classes'],
            pretrained=False
        ).get_model()

        model_path = "checkpoints/best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=config['experiment']['device']))
        model.eval()

        device = torch.device(config['experiment']['device'])
        model = model.to(device)

        print(f"Model loaded from: {model_path}")

        # 📊 Детальная оценка на валидации
        print("\n" + "=" * 60)
        print("DETAILED VALIDATION ANALYSIS")
        print("=" * 60)

        metrics, val_predictions = calculate_comprehensive_metrics(
            model, dataloaders['val'], device, torch.nn.CrossEntropyLoss()
        )
        print_detailed_metrics(metrics, dataloaders['class_names'])

        # Confusion Matrix
        cm = confusion_matrix(
            [label for _, label in dataloaders['val'].dataset],
            val_predictions[:len(dataloaders['val'].dataset)]
        )

        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(cm, dataloaders['class_names'], 'Validation Confusion Matrix')
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 🔮 Предсказания на тестовых данных
        print("\n" + "=" * 60)
        print("TEST PREDICTIONS")
        print("=" * 60)

        test_predictions = []
        test_probabilities = []

        with torch.no_grad():
            for images, _ in dataloaders['test']:
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                test_predictions.extend(preds.cpu().numpy())
                test_probabilities.extend(probabilities.cpu().numpy())

        # Комплексный отчет
        create_comprehensive_report(model, dataloaders, device, test_predictions, test_probabilities)

        # Визуализация примеров
        visualize_sample_predictions(model, dataloaders['test'], device,
                                     test_predictions, test_probabilities,
                                     dataloaders['class_names'])

        print(f"\n🎉 Analysis completed successfully!")
        print(f"   Model achieves {metrics['accuracy']:.4f} validation accuracy")
        print(f"   Generated {len(test_predictions)} test predictions")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def create_comprehensive_report(model, dataloaders, device, predictions, probabilities):
    """Создание комплексного отчета о модели"""

    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL REPORT")
    print("=" * 60)

    # Анализ по классам
    class_stats = {}
    for class_idx in range(dataloaders['num_classes']):
        class_predictions = [p for p in predictions if p == class_idx]
        class_confidences = [prob[pred] for pred, prob in zip(predictions, probabilities) if pred == class_idx]

        if class_predictions:
            class_stats[class_idx] = {
                'count': len(class_predictions),
                'avg_confidence': np.mean(class_confidences),
                'min_confidence': np.min(class_confidences),
                'max_confidence': np.max(class_confidences)
            }

    print("\n📈 Class-wise Statistics:")
    for class_idx, stats in class_stats.items():
        class_name = dataloaders['class_names'][class_idx]
        print(f"  {class_name}:")
        print(f"    Samples: {stats['count']:4d}")
        print(f"    Avg Confidence: {stats['avg_confidence']:.4f}")
        print(f"    Confidence Range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")

    # Анализ уверенности
    all_confidences = [prob[pred] for pred, prob in zip(predictions, probabilities)]
    confidence_thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]

    print(f"\n🎯 Confidence Analysis:")
    print(f"  Overall Average: {np.mean(all_confidences):.4f}")
    print(f"  Standard Deviation: {np.std(all_confidences):.4f}")

    for threshold in confidence_thresholds:
        above_threshold = sum(1 for c in all_confidences if c >= threshold)
        percentage = (above_threshold / len(all_confidences)) * 100
        print(f"  ≥{threshold}: {above_threshold:5d} samples ({percentage:6.2f}%)")

    # Создание финального файла предсказаний
    save_final_predictions(predictions, dataloaders['class_names'])


def save_final_predictions(predictions, class_names):
    """Сохранение предсказаний в финальном формате"""

    # Простой формат (только номера классов)
    with open('final_predictions.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    # Детальный формат
    with open('detailed_predictions.csv', 'w') as f:
        f.write("image_id,predicted_class,class_name\n")
        for i, pred in enumerate(predictions):
            class_name = class_names[pred]
            f.write(f"{i + 1},{pred},{class_name}\n")

    print(f"\n💾 Predictions saved:")
    print(f"   - final_predictions.txt: {len(predictions)} predictions")
    print(f"   - detailed_predictions.csv: detailed report")


def visualize_sample_predictions(model, test_loader, device, predictions, probabilities, class_names, num_samples=10):
    """Визуализация примеров предсказаний"""
    # Получаем несколько тестовых изображений
    test_images = []
    test_batch = next(iter(test_loader))

    # Берем первые num_samples изображений из батча
    sample_images = test_batch[0][:num_samples]
    sample_predictions = predictions[:num_samples]
    sample_probabilities = probabilities[:num_samples]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    for i in range(num_samples):
        image = sample_images[i]
        pred = sample_predictions[i]
        prob = sample_probabilities[i]
        confidence = prob[pred]

        # Денормализация для отображения
        image_np = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)

        axes[i].imshow(image_np)
        axes[i].set_title(f'Pred: {class_names[pred]}\nConf: {confidence:.3f}',
                          color='green' if confidence > 0.9 else 'orange')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('results/sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Визуализация распределения уверенности
    all_confidences = [prob[pred] for pred, prob in zip(predictions, probabilities)]

    plt.figure(figsize=(10, 6))
    plt.hist(all_confidences, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()