import sys
import os

# Добавляем src в путь Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("🔧 Testing imports...")
    
    # Проверяем основные импорты
    from segmentation import HumanSegmentator
    from evaluation import SegmentationEvaluator
    from visualization import ResultsVisualizer
    
    print("✅ All imports successful!")
    
    # Тестируем базовую функциональность
    print("🧪 Testing basic functionality...")
    
    # Инициализация компонентов
    segmentator = HumanSegmentator()
    evaluator = SegmentationEvaluator()
    visualizer = ResultsVisualizer()
    
    print("✅ Components initialized successfully!")
    
    # Тест вычисления IoU
    import numpy as np
    test_mask1 = np.random.rand(100, 100) > 0.5
    test_mask2 = np.random.rand(100, 100) > 0.5
    iou = evaluator.calculate_iou(test_mask1, test_mask2)
    print(f"✅ IoU calculation test: {iou:.4f}")
    
    print("🎉 All tests passed! The project structure is correct.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()