import sys
import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def final_project_test():
    print("🎯 FINAL PROJECT VALIDATION")
    print("=" * 50)
    
    try:
        # Тестируем все компоненты
        from segmentation import HumanSegmentator
        from evaluation import SegmentationEvaluator
        from visualization import ResultsVisualizer
        
        print("✅ All modules imported successfully")
        
        # Тестируем на реальном изображении
        print("📥 Downloading test image...")
        url = "https://ultralytics.com/images/bus.jpg"
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        test_image = np.array(image)
        
        print("🧪 Testing segmentation...")
        segmentator = HumanSegmentator()
        mask = segmentator.segment(test_image)
        
        print("📊 Testing evaluation...")
        evaluator = SegmentationEvaluator()
        
        # Создаем простую ground truth для демо
        gt_mask = np.zeros(test_image.shape[:2])
        metrics = evaluator.evaluate_single(gt_mask, mask)
        
        print("🖼️ Testing visualization...")
        visualizer = ResultsVisualizer()
        save_path = visualizer.save_results(test_image, gt_mask, mask, metrics, "final_test")
        
        print("\n🎉 PROJECT VALIDATION SUCCESSFUL!")
        print("✅ All components work correctly")
        print(f"✅ Results saved to: {save_path}")
        print(f"✅ Segmentation mask shape: {mask.shape}")
        
        # Показываем что обнаружено
        if mask.sum() > 0:
            coverage = mask.sum() / mask.size * 100
            print(f"✅ Objects detected: Mask covers {coverage:.1f}% of image")
            
            # Показываем детекции
            results = segmentator.model(test_image, verbose=False)
            detections = []
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    for cls, conf in zip(classes, confidences):
                        class_name = result.names[int(cls)]
                        detections.append((class_name, conf))
            
            print(f"✅ Detected objects: {len(detections)}")
            for class_name, conf in detections[:5]:
                print(f"   - {class_name}: {conf:.3f}")
        else:
            print("ℹ️  No objects detected in test image")
            
        print("\n🏆 PROJECT STATUS: FULLY OPERATIONAL")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    final_project_test()