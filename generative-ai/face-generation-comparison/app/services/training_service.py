import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Reshape, Conv2D, Conv2DTranspose, 
                                    Flatten, Dropout, BatchNormalization, 
                                    LeakyReLU, Input, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image
import os
import logging
from ..config.settings import settings
from ..models.sampling_layer import Sampling  # Импортируем кастомный слой

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.images = None
        self.gan_generator = None
        self.vae_encoder = None
        self.vae_decoder = None
        
    def load_images(self):
        """Загрузка и предобработка изображений"""
        logger.info("Загрузка изображений...")
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

    def build_gan_generator(self):
        """Построение генератора GAN"""
        model = Sequential()
        
        model.add(Dense(256 * 8 * 8, input_dim=settings.GAN_LATENT_DIM))
        model.add(Reshape((8, 8, 256)))
        
        # Блок 1
        model.add(Conv2DTranspose(256, 4, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        
        # Блок 2
        model.add(Conv2DTranspose(128, 4, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        
        # Блок 3
        model.add(Conv2DTranspose(64, 4, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        
        # Выходной слой
        model.add(Conv2DTranspose(3, 4, padding='same', activation='tanh'))
        
        self.gan_generator = model
        return model

    def build_gan_discriminator(self):
        """Построение дискриминатора GAN"""
        model = Sequential()
        
        # Блок 1
        model.add(Conv2D(64, 4, strides=2, padding='same', 
                        input_shape=settings.IMAGE_SIZE + (3,)))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        
        # Блок 2
        model.add(Conv2D(128, 4, strides=2, padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        
        # Блок 3
        model.add(Conv2D(256, 4, strides=2, padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        
        # Блок 4
        model.add(Conv2D(512, 4, strides=2, padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        
        # Выходной слой
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        return model

    def train_gan(self):
        """Обучение GAN модели"""
        if self.images is None:
            self.load_images()
            
        logger.info("Начало обучения GAN...")
        
        # Построение моделей
        generator = self.build_gan_generator()
        discriminator = self.build_gan_discriminator()
        
        # Компиляция дискриминатора
        discriminator.compile(optimizer=Adam(settings.GAN_LEARNING_RATE, 0.5),
                             loss='binary_crossentropy',
                             metrics=['accuracy'])
        
        # Комбинированная модель (GAN)
        discriminator.trainable = False
        z = Input(shape=(settings.GAN_LATENT_DIM,))
        img = generator(z)
        validity = discriminator(img)
        gan = Model(z, validity)
        gan.compile(optimizer=Adam(settings.GAN_LEARNING_RATE, 0.5), 
                   loss='binary_crossentropy')
        
        # Обучение
        valid = np.ones((settings.BATCH_SIZE, 1))
        fake = np.zeros((settings.BATCH_SIZE, 1))
        
        for epoch in range(settings.GAN_EPOCHS):
            # Обучение дискриминатора
            idx = np.random.randint(0, self.images.shape[0], settings.BATCH_SIZE)
            real_imgs = self.images[idx]
            
            noise = np.random.normal(0, 1, (settings.BATCH_SIZE, settings.GAN_LATENT_DIM))
            gen_imgs = generator.predict(noise, verbose=0)
            
            d_loss_real = discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Обучение генератора
            noise = np.random.normal(0, 1, (settings.BATCH_SIZE, settings.GAN_LATENT_DIM))
            g_loss = gan.train_on_batch(noise, valid)
            
            if epoch % 500 == 0:
                logger.info(f"GAN Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
                
                # Сохранение промежуточной модели
                if epoch % 2000 == 0:
                    generator.save(f"trained_models/gan_generator_epoch_{epoch}.h5")
        
        # Сохранение финальной модели
        generator.save(str(settings.GAN_GENERATOR_PATH))
        logger.info(f"GAN обучение завершено. Модель сохранена в {settings.GAN_GENERATOR_PATH}")
        
        self.gan_generator = generator
        return generator

    def build_vae(self):
        """Построение VAE модели с использованием кастомного слоя Sampling"""
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
    
        self.vae_encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
        # Декодер
        latent_inputs = Input(shape=(settings.VAE_LATENT_DIM,))
        x = Dense(512, activation='relu')(latent_inputs)
        x = Dense(8 * 8 * 256, activation='relu')(x)
        x = Reshape((8, 8, 256))(x)
        x = Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        outputs = Conv2DTranspose(3, 3, activation='tanh', padding='same')(x)
    
        self.vae_decoder = Model(latent_inputs, outputs, name='decoder')
    
        # VAE модель
        outputs = self.vae_decoder(self.vae_encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')
    
        return vae

    class VAELoss(tf.keras.losses.Loss):
        def __init__(self):
            super().__init__()
            self.mse = MeanSquaredError()
        
        def call(self, y_true, y_pred):
            # Reconstruction loss
            reconstruction_loss = self.mse(
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
        
        def set_encoder(self, encoder):
            self.encoder = encoder

    def train_vae(self):
        """Обучение VAE модели"""
        if self.images is None:
            self.load_images()
            
        logger.info("Начало обучения VAE...")
        
        # Построение модели
        vae = self.build_vae()
        
        # Настройка функции потерь
        vae_loss = self.VAELoss()
        vae_loss.set_encoder(self.vae_encoder)
        
        # Компиляция
        vae.compile(optimizer=Adam(settings.VAE_LEARNING_RATE), loss=vae_loss)
        
        # Обучение
        history = vae.fit(
            self.images, self.images,
            epochs=settings.VAE_EPOCHS,
            batch_size=settings.BATCH_SIZE,
            validation_split=0.2,
            verbose=1
        )
        
        # Сохранение моделей
        self.vae_encoder.save(str(settings.VAE_ENCODER_PATH))
        self.vae_decoder.save(str(settings.VAE_DECODER_PATH))
        
        logger.info(f"VAE обучение завершено. Модели сохранены")
        logger.info(f"Encoder: {settings.VAE_ENCODER_PATH}")
        logger.info(f"Decoder: {settings.VAE_DECODER_PATH}")
        
        return self.vae_encoder, self.vae_decoder

    def train_all_models(self):
        """Обучение всех моделей"""
        logger.info("Запуск обучения всех моделей...")
        
        # Обучение GAN
        gan_result = self.train_gan()
        
        # Обучение VAE  
        vae_encoder, vae_decoder = self.train_vae()
        
        logger.info("Все модели успешно обучены!")
        return {
            'gan_generator': gan_result,
            'vae_encoder': vae_encoder,
            'vae_decoder': vae_decoder
        }

# Глобальный экземпляр сервиса тренировки
training_service = TrainingService()