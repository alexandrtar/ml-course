#!/usr/bin/env python3
"""
Скрипт для обучения моделей из командной строки
"""

import argparse
import logging
from app.services.training_service import training_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train face generation models')
    parser.add_argument('--model', type=str, choices=['gan', 'vae', 'all'], 
                       default='all', help='Model to train')
    parser.add_argument('--max-images', type=int, default=5000,
                       help='Maximum number of images to use for training')
    
    args = parser.parse_args()
    
    # Обновление настроек
    from app.config.settings import settings
    settings.MAX_IMAGES = args.max_images
    
    logger.info(f"Starting training for {args.model} model(s)")
    logger.info(f"Using up to {args.max_images} images")
    
    if args.model == 'gan':
        training_service.train_gan()
    elif args.model == 'vae':
        training_service.train_vae()
    else:  # all
        training_service.train_all_models()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()