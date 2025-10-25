#!/usr/bin/env python3
"""
Упрощенный скрипт для запуска обучения моделей
"""

import sys
import os

# Добавляем корневую директорию в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import matplotlib.pyplot as plt


def main():
    """Основная функция обучения"""
    try:
        # Импорты после добавления пути
        from src.utils.config import load_config, setup_experiment_env
        from src.data.dataloaders import create_dataloaders
        from src.models.custom_cnn import ImprovedCarCNN
        from src.models.resnet_wrapper import AdvancedResNetWrapper
        from src.training.trainer import ModelTrainer
        from src.utils.visualization import plot_training_history

        # Загрузка конфигурации
        config_path = os.path.join(project_root, 'configs', 'experiment_config.yaml')
        config = load_config(config_path)
        setup_experiment_env(config)

        # Создание DataLoader'ов
        print("Creating data loaders...")
        dataloaders = create_dataloaders(config, use_albumentations=False)

        print(f"Number of classes: {dataloaders['num_classes']}")
        print(f"Class names: {dataloaders['class_names']}")

        # Создаем директории для результатов
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # Обучаем только одну модель для начала
        model_name = "resnet18"
        print(f"\n{'=' * 60}")
        print(f"TRAINING MODEL: {model_name.upper()}")
        print(f"{'=' * 60}")

        # Создание модели
        if model_name == 'custom_cnn':
            model = ImprovedCarCNN(dataloaders['num_classes'])
        else:
            wrapper = AdvancedResNetWrapper(
                model_name=model_name,
                num_classes=dataloaders['num_classes'],
                pretrained=config['models']['pretrained']
            )
            model = wrapper.get_model()

        # Обучение с простым путем
        trainer = ModelTrainer(model, config)
        results = trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            save_path="checkpoints/best_model.pth"  # Простой относительный путь
        )

        # Визуализация результатов
        fig = plot_training_history(results, model_name)
        plt.savefig("results/training_history.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nTraining completed!")
        print(f"Best F1 Score: {results['best_metric']:.4f}")
        print(f"Model saved to: {results['best_model_path']}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()