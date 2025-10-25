import os
import sys

# Добавляем текущую директорию в Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

def run_project():
    print("🎯 Human Segmentation Project Runner")
    print("=" * 40)
    
    # Проверяем существование необходимых файлов
    required_files = [
        'src/__init__.py',
        'src/segmentation.py', 
        'src/evaluation.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📁 Current directory structure:")
        for root, dirs, files in os.walk('.'):
            level = root.replace('.', '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f'{subindent}{file}')
        return
    
    print("✅ All required files found!")
    
    try:
        # Динамический импорт
        from src.segmentation import HumanSegmentator
        from src.evaluation import SegmentationEvaluator
        
        print("🧪 Running basic functionality test...")
        
        # Тестируем
        segmentator = HumanSegmentator()
        evaluator = SegmentationEvaluator()
        
        # Создаем тестовые данные
        import numpy as np
        test_img = np.random.rand(100, 100, 3).astype(np.float32)
        test_mask = segmentator.segment(test_img)
        
        print(f"✅ Segmentation successful! Mask shape: {test_mask.shape}")
        print("🎉 Project is ready to use!")
        
        # Показываем доступные функции
        print("\n📚 Available functions:")
        print("  - HumanSegmentator().segment(image)")
        print("  - SegmentationEvaluator().calculate_iou(mask1, mask2)")
        print("  - ResultsVisualizer().plot_comparison(...)")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_project()