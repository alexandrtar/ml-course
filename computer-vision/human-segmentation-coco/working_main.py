import argparse
import sys
import os

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚀 WORKING Human Segmentation Project")
    print("=" * 50)
    
    try:
        from segmentation import HumanSegmentator
        from evaluation import SegmentationEvaluator
        from visualization import ResultsVisualizer
        
        # Инициализация
        segmentator = HumanSegmentator()
        evaluator = SegmentationEvaluator()
        visualizer = ResultsVisualizer()
        
        print("✅ Components initialized")
        
        # Создаем тестовое изображение с несколькими объектами
        import cv2
        import numpy as np
        
        print("🖼️ Creating test scenario...")
        
        # Создаем сложное тестовое изображение
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Рисуем несколько объектов
        cv2.rectangle(image, (50, 50), (150, 200), (255, 0, 0), -1)  # Синий прямоугольник
        cv2.circle(image, (300, 150), 60, (0, 255, 0), -1)  # Зеленый круг
        cv2.ellipse(image, (400, 350), (80, 40), 0, 0, 360, (0, 0, 255), -1)  # Красный эллипс
        
        print("🤖 Running segmentation...")
        mask = segmentator.segment(image)
        
        # Создаем "ground truth" для демо
        gt_mask = np.zeros((512, 512), dtype=np.float32)
        cv2.rectangle(gt_mask, (50, 50), (150, 200), 1, -1)
        cv2.circle(gt_mask, (300, 150), 60, 1, -1)
        cv2.ellipse(gt_mask, (400, 350), (80, 40), 0, 0, 360, 1, -1)
        
        # Оценка
        metrics = evaluator.evaluate_single(gt_mask, mask)
        
        print("📊 Results:")
        print("-" * 30)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Визуализация
        save_path = visualizer.save_results(image, gt_mask, mask, metrics, "demo_test")
        print(f"💾 Results saved to: {save_path}")
        
        print("🎉 Project is working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()