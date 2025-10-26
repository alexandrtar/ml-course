#!/usr/bin/env python3
"""
Training script for medical image segmentation with comprehensive features
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_image_segmentation.data.dataset import MedicalDataset, MedicalDataLoader
from medical_image_segmentation.data.synthetic_data import create_medical_dataset
from medical_image_segmentation.models.unet import create_model
from medical_image_segmentation.models.losses import create_loss
from medical_image_segmentation.training.trainer import MedicalTrainer
from medical_image_segmentation.utils.logger import setup_logger
from medical_image_segmentation.utils.device_utils import setup_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train medical segmentation model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--data_dir', type=str, default='data/medical_data',
                       help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Path to output directory for models and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to train on (auto/cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    return parser.parse_args()


def setup_environment(config):
    """Setup training environment"""
    # Create directories
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['data']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['data']['random_seed'])
    
    print("🔧 Environment setup completed")


def load_config(args):
    """Load and update configuration"""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    return config


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Setup environment
    setup_environment(config)
    
    # Setup device
    device = setup_device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Setup logger
    logger = setup_logger(
        experiment_name=config['tracking']['experiment_name'],
        run_name=config['tracking']['run_name'].replace('${timestamp}', 
                                                       f"{os.getpid()}_{torch.randint(1000, 10000, (1,)).item()}"),
        use_mlflow=config['tracking']['use_mlflow'],
        use_wandb=config['tracking']['use_wandb'],
        log_dir='logs'
    )
    
    # Create synthetic data if needed
    if not os.path.exists(args.data_dir):
        print("🔄 Creating synthetic medical dataset...")
        create_medical_dataset(
            output_dir=args.data_dir,
            num_samples=config['data']['synthetic_samples'],
            image_size=config['training']['image_size']
        )
    
    # Create datasets and data loaders
    print("📊 Preparing data loaders...")
    train_loader, val_loader = MedicalDataLoader.create_loaders(
        train_dir=os.path.join(args.data_dir, 'train'),
        val_dir=os.path.join(args.data_dir, 'val'),
        batch_size=config['training']['batch_size'],
        image_size=config['training']['image_size'],
        num_workers=config['training']['num_workers']
    )
    
    print(f"📈 Training samples: {len(train_loader.dataset)}")
    print(f"📈 Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("🤖 Creating model...")
    model = create_model(
        model_name=config['model']['name'],
        n_channels=config['model']['in_channels'],
        n_classes=config['model']['out_channels'],
        features=config['model']['features'],
        dropout=config['model']['dropout'],
        normalization=config['model']['normalization'],
        activation=config['model']['activation']
    )
    
    print(f"📐 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = create_loss(
        config['loss']['name'],
        dice_weight=config['loss']['dice_weight'],
        bce_weight=config['loss']['bce_weight'],
        iou_weight=config['loss']['iou_weight'],
        focal_weight=config['loss']['focal_weight']
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=config['training']['scheduler_patience'],
        factor=config['training']['scheduler_factor'],
        verbose=True
    )
    
    # Log parameters
    logger.log_params({
        'model': config['model']['name'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'optimizer': 'adam',
        'scheduler': 'reduce_lr_on_plateau',
        'loss_function': config['loss']['name']
    })
    
    # Create trainer
    trainer = MedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        config=config
    )
    
    # Resume training if checkpoint provided
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("🚀 Starting training...")
    history = trainer.train(
        epochs=config['training']['epochs'],
        save_dir='checkpoints'
    )
    
    # Save final model
    trainer.save_checkpoint(os.path.join('checkpoints', 'final_model.pth'))
    
    # Log completion
    logger.log_metrics({'final_best_dice': trainer.best_dice})
    logger.finish()
    
    print("🎉 Training completed successfully!")
    print(f"🏆 Best Dice: {trainer.best_dice:.4f}")
    
    return history


if __name__ == '__main__':
    try:
        history = main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise