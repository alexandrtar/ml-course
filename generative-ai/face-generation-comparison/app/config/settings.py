import os
from pathlib import Path

class Settings:
    # Пути
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "lfw-deepfunneled"
    MODEL_DIR = BASE_DIR / "trained_models"
    
    # Настройки данных
    IMAGE_SIZE = (64, 64)
    MAX_IMAGES = 5000
    BATCH_SIZE = 32
    
    # GAN настройки
    GAN_LATENT_DIM = 256
    GAN_EPOCHS = 10000
    GAN_LEARNING_RATE = 0.0001
    
    # VAE настройки  
    VAE_LATENT_DIM = 256
    VAE_EPOCHS = 100
    VAE_LEARNING_RATE = 0.0005
    
    # API настройки
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    DEBUG = True
    
    # Пути к моделям
    GAN_GENERATOR_PATH = MODEL_DIR / "gan_generator.h5"
    VAE_ENCODER_PATH = MODEL_DIR / "vae_encoder.h5"
    VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.h5"
    
    def __init__(self):
        # Создание директорий если не существуют
        self.MODEL_DIR.mkdir(exist_ok=True)

settings = Settings()