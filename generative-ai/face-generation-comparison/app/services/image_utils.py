import io
import base64
from PIL import Image
import numpy as np
import cv2

class ImageUtils:
    @staticmethod
    def numpy_to_base64(image_array: np.ndarray) -> str:
        """Конвертация numpy array в base64 строку"""
        # Приведение к правильному формату
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Если batch изображений, берем первое
        if len(image_array.shape) == 4:
            image_array = image_array[0]
        
        # Убедимся, что изображение имеет правильную форму (H, W, C)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            pil_image = Image.fromarray(image_array)
        else:
            # Если что-то не так, конвертируем в правильный формат
            if len(image_array.shape) == 3:
                image_array = image_array.transpose(1, 2, 0)
            pil_image = Image.fromarray(image_array)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str
    
    @staticmethod
    def create_image_grid(images: np.ndarray, cols: int = 5) -> np.ndarray:
        """Создание сетки изображений"""
        if len(images.shape) != 4:
            raise ValueError("Ожидается batch изображений")
        
        n, h, w, c = images.shape
        rows = (n + cols - 1) // cols
        
        # Создание пустой сетки
        grid = np.zeros((rows * h, cols * w, c), dtype=images.dtype)
        
        for i in range(n):
            row = i // cols
            col = i % cols
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = images[i]
        
        return grid
    
    @staticmethod
    def resize_images(images: np.ndarray, target_size: tuple) -> np.ndarray:
        """Изменение размера batch изображений"""
        resized = []
        for img in images:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
            resized.append(np.array(pil_img))
        
        return np.array(resized)