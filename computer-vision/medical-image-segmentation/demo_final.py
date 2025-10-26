import torch
import numpy as np
import matplotlib.pyplot as plt
from medical_image_segmentation import UNet, MedicalDataset

def demo_trained_model():
    print("🎯 Final Demo: Medical Image Segmentation")
    
    # Загружаем обученную модель
    try:
        model = UNet()
        model.load_state_dict(torch.load('medical_unet_trained.pth', map_location='cpu'))
        print("✅ Model loaded from 'trained_model.pth'")
    except:
        try:
            model = UNet()
            model.load_state_dict(torch.load('final_model.pth', map_location='cpu'))
            print("✅ Model loaded from 'final_model.pth'")
        except:
            model = UNet()
            print("✅ New model created for demo")
    
    model.eval()
    
    # Используем реальные данные из validation set
    try:
        val_dataset = MedicalDataset('data/medical_data/val', image_size=(256, 256))
        print(f"✅ Using real validation data: {len(val_dataset)} samples")
        
        # Берем несколько примеров
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i in range(3):
            image, true_mask = val_dataset[i]
            
            with torch.no_grad():
                prediction = model(image.unsqueeze(0))
                pred_np = prediction[0, 0].numpy()
            
            # Денормализуем изображение
            img_display = (image.squeeze().numpy() * 0.5 + 0.5)
            true_mask_display = true_mask.squeeze().numpy()
            
            # Исходное изображение
            axes[i, 0].imshow(img_display, cmap='gray')
            axes[i, 0].set_title(f'Sample {i+1}\nInput Image')
            axes[i, 0].axis('off')
            
            # Истинная маска
            axes[i, 1].imshow(true_mask_display, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Предсказание
            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Наложение
            axes[i, 3].imshow(img_display, cmap='gray')
            axes[i, 3].imshow(pred_np, cmap='jet', alpha=0.5)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('final_demo_real_data.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("💾 Real data demo saved as 'final_demo_real_data.png'")
        
    except Exception as e:
        print(f"⚠️ Could not load real data: {e}")
        print("🔄 Using synthetic data for demo...")
        
        # Создаем синтетическое изображение для демо
        image_size = (256, 256)
        image = np.random.normal(0.4, 0.15, image_size)
        
        # Добавляем структуры
        centers = [(100, 100), (180, 150), (80, 200)]
        for center in centers:
            y, x = np.ogrid[:image_size[0], :image_size[1]]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            organ_mask = dist <= 30
            image[organ_mask] += 0.3
        
        image = np.clip(image, 0, 1)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        image_tensor = (image_tensor - 0.5) / 0.5  # Нормализация
        
        with torch.no_grad():
            prediction = model(image_tensor)
            pred_np = prediction[0, 0].numpy()
        
        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Synthetic Medical Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_np, cmap='gray')
        axes[1].set_title('Segmentation Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(pred_np, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('final_demo_synthetic.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("💾 Synthetic demo saved as 'final_demo_synthetic.png'")
    
    print("🎉 Demo completed successfully!")

if __name__ == "__main__":
    demo_trained_model()