import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os
import logging
from ..config.settings import settings
from ..models.sampling_layer import Sampling

logger = logging.getLogger(__name__)

class VAETrainer:
    def __init__(self):
        self.images = None
        self.encoder = None
        self.decoder = None
        self.vae = None
        
    def load_images(self):
        """Загрузка изображений для обучения"""
        from tensorflow.keras.preprocessing.image import img_to_array
        from PIL import Image
        
        logger.info("Загрузка изображений для VAE...")
        images = []
        count = 0
        
        for root, _, files in os.walk(settings.DATA_DIR):
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(root, file)
                    try:
                        img = Image.open(img_path)
                        img = img.resize(settings.IMAGE_SIZE)
                        img_array = img_to_array(img)
                        images.append(img_array)
                        count += 1
                        if settings.MAX_IMAGES and count >= settings.MAX_IMAGES:
                            break
                    except Exception as e:
                        logger.warning(f"Ошибка загрузки {img_path}: {e}")
                        
        self.images = np.array(images)
        # Нормализация в [-1, 1]
        self.images = (self.images - 127.5) / 127.5
        logger.info(f"Загружено {len(self.images)} изображений")
        return self.images

    def build_models(self):
        """Построение VAE моделей с использованием кастомного слоя Sampling"""
        # Энкодер
        inputs = Input(shape=settings.IMAGE_SIZE + (3,))
        x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(inputs)
        x = Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2D(256, 3, activation='relu', strides=2, padding='same')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        
        z_mean = Dense(settings.VAE_LATENT_DIM)(x)
        z_log_var = Dense(settings.VAE_LATENT_DIM)(x)
        
        # Используем кастомный слой Sampling вместо Lambda
        z = Sampling()([z_mean, z_log_var])
        
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Декодер
        latent_inputs = Input(shape=(settings.VAE_LATENT_DIM,))
        x = Dense(512, activation='relu')(latent_inputs)
        x = Dense(8 * 8 * 256, activation='relu')(x)
        x = Reshape((8, 8, 256))(x)
        x = Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        outputs = Conv2DTranspose(3, 3, activation='tanh', padding='same')(x)
        
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        
        # VAE модель
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')
        
        return self.encoder, self.decoder, self.vae

    def vae_loss(self, y_true, y_pred):
        """Функция потерь для VAE"""
        # Reconstruction loss
        reconstruction_loss = MeanSquaredError()(
            tf.keras.layers.Flatten()(y_true), 
            tf.keras.layers.Flatten()(y_pred)
        )
        reconstruction_loss *= settings.IMAGE_SIZE[0] * settings.IMAGE_SIZE[1] * 3
        
        # KL divergence
        z_mean, z_log_var, _ = self.encoder(y_true)
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 
            axis=1
        )
        
        return reconstruction_loss + kl_loss

    def train(self, epochs=None, batch_size=None):
        """Обучение VAE модели"""
        if self.images is None:
            self.load_images()
            
        if epochs is None:
            epochs = settings.VAE_EPOCHS
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
            
        logger.info("Начало обучения VAE...")
        
        # Построение моделей
        self.build_models()
        
        # Компиляция
        self.vae.compile(optimizer=Adam(settings.VAE_LEARNING_RATE), 
                        loss=self.vae_loss)
        
        # Обучение
        history = self.vae.fit(
            self.images, self.images,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Сохранение моделей
        self.encoder.save(str(settings.VAE_ENCODER_PATH))
        self.decoder.save(str(settings.VAE_DECODER_PATH))
        
        logger.info(f"VAE обучение завершено. Модели сохранены:")
        logger.info(f"Encoder: {settings.VAE_ENCODER_PATH}")
        logger.info(f"Decoder: {settings.VAE_DECODER_PATH}")
        
        return history

# Глобальный экземпляр тренера
vae_trainer = VAETrainer()