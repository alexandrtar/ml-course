import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import os

def download_test_image():
    """Скачиваем тестовое изображение с человеком"""
    print("📥 Downloading test image with human...")
    
    # URL изображения с человеком
    urls = [
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = np.array(image)
                print(f"✅ Downloaded image from {url}")
                return image
        except Exception as e:
            print(f"❌ Failed to download from {url}: {e}")
            continue
    
    # Если не удалось скачать, создаем простое изображение
    print("⚠️  Using fallback image...")
    return create_simple_human_image()

def create_simple_human_image():
    """Создаем простое изображение с человеком"""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    return img

def run_demo():
    print("🎯 YOLO Human Segmentation DEMO")
    print("=" * 50)
    
    try:
        # Загружаем модель
        print("1. Loading YOLO model...")
        model = YOLO('yolov8n-seg.pt')
        print("   ✅ Model loaded")
        
        # Получаем тестовое изображение
        print("2. Getting test image...")
        image = download_test_image()
        print(f"   ✅ Image shape: {image.shape}")
        
        # Выполняем сегментацию
        print("3. Running segmentation...")
        results = model(image, verbose=False)
        
        # Обрабатываем результаты - ИСПРАВЛЕННАЯ ВЕРСИЯ
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        detections = []
        
        for result in results:
            # Исправленный доступ к bounding boxes
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bbox
                classes = result.boxes.cls.cpu().numpy()  # Классы
                confidences = result.boxes.conf.cpu().numpy()  # Уверенности
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    class_name = result.names[int(cls)]
                    detections.append((class_name, conf))
                    print(f"   ✅ Detected: {class_name} (confidence: {conf:.3f})")
            
            # Обработка масок
            if result.masks is not None:
                for i, m in enumerate(result.masks):
                    mask_data = m.data[0].cpu().numpy()
                    # Изменяем размер маски к размеру исходного изображения
                    mask_resized = cv2.resize(mask_data, (image.shape[1], image.shape[0]))
                    mask = np.logical_or(mask, mask_resized)
                print(f"   🎭 Found {len(result.masks)} segmentation mask(s)")
        
        # Визуализация
        print("4. Creating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Оригинальное изображение
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Маска сегментации
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Segmentation Mask')
        axes[0, 1].axis('off')
        
        # Наложение маски
        axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if mask.sum() > 0:
            axes[1, 0].imshow(mask, cmap='jet', alpha=0.5)
            axes[1, 0].set_title('Mask Overlay')
        else:
            axes[1, 0].set_title('No Segmentation Found')
        axes[1, 0].axis('off')
        
        # Информация
        axes[1, 1].axis('off')
        info_text = "Detection Results:\n\n"
        if detections:
            for class_name, confidence in detections[:8]:  # Показываем первые 8
                info_text += f"{class_name}: {confidence:.3f}\n"
        else:
            info_text += "No objects detected\n\n"
            info_text += "Image might not contain detectable objects"
        
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, va='center', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('quick_demo_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📁 Results saved to 'quick_demo_result.png'")
        
        # Показываем статистику маски
        if mask.sum() > 0:
            mask_coverage = (mask.sum() / (mask.shape[0] * mask.shape[1])) * 100
            print(f"📊 Mask covers {mask_coverage:.1f}% of image")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()