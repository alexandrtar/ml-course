import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def minimal_segmentation_example():
    """Минимальный рабочий пример сегментации"""
    print("🎯 Minimal YOLO Segmentation Example")
    
    # Загружаем модель
    model = YOLO('yolov8n-seg.pt')
    
    # Создаем тестовое изображение (можно заменить на загрузку реального)
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Выполняем сегментацию
    results = model(test_image, classes=[0], conf=0.25)  # class 0 = person
    
    # Создаем маску
    mask = np.zeros((512, 512), dtype=np.float32)
    
    for result in results:
        if result.masks is not None:
            for m in result.masks:
                mask = np.logical_or(mask, m.data[0].cpu().numpy())
    
    # Визуализация
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title("Test Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Segmentation Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('minimal_test_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Segmentation completed! Mask shape: {mask.shape}")
    print(f"📁 Result saved to 'minimal_test_result.png'")

if __name__ == "__main__":
    minimal_segmentation_example()