import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import time
import os
from tqdm import tqdm

from ..models.losses import DiceLoss, CombinedLoss
from ..utils.logger import TrainingLogger
from ..evaluation.metrics import calculate_metrics


class MedicalTrainer:
    """
    Professional trainer for medical image segmentation models
    with comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        logger: TrainingLogger = None,
        config: Dict = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.config = config or {}
        
        # Loss functions
        self.criterion = CombinedLoss(
            dice_weight=0.7,
            bce_weight=0.3,
            iou_weight=0.0
        )
        
        # Training state
        self.best_dice = 0.0
        self.epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'learning_rate': []
        }
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            try:
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Log batch metrics
                if self.logger and batch_idx % 10 == 0:
                    self.logger.log_batch_metrics({
                        'batch_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]["lr"]
                    })
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        return epoch_loss / num_batches if num_batches > 0 else float('inf')
    
    def validate_epoch(self) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        all_metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': []
        }
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                try:
                    images, masks = images.to(self.device), masks.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    
                    # Calculate metrics
                    batch_metrics = calculate_metrics(outputs, masks)
                    for key in all_metrics.keys():
                        all_metrics[key].append(batch_metrics[key])
                    
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Aggregate metrics
        avg_metrics = {
            key: np.mean(values) if values else 0.0 
            for key, values in all_metrics.items()
        }
        avg_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        
        return avg_loss, avg_metrics
    
    def train(self, epochs: int, save_dir: str = "checkpoints") -> Dict:
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("🚀 Starting training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['dice'].append(val_metrics['dice'])
            self.history['iou'].append(val_metrics['iou'])
            self.history['precision'].append(val_metrics['precision'])
            self.history['recall'].append(val_metrics['recall'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]["lr"])
            
            # Print epoch results
            print(f'Epoch {epoch+1:02d}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Dice: {val_metrics["dice"]:.4f} | '
                  f'IoU: {val_metrics["iou"]:.4f} | '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Log epoch metrics
            if self.logger:
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **val_metrics,
                    'learning_rate': self.optimizer.param_groups[0]["lr"]
                }
                self.logger.log_epoch_metrics(epoch_metrics)
            
            # Save best model
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    metrics=val_metrics
                )
                print(f"💾 Saved best model with Dice: {self.best_dice:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                    metrics=val_metrics
                )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"🏆 Best Dice: {self.best_dice:.4f}")
        
        return self.history
    
    def save_checkpoint(self, path: str, metrics: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_dice': self.best_dice,
            'history': self.history,
            'config': self.config,
            'metrics': metrics
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_dice = checkpoint['best_dice']
        self.history = checkpoint['history']
        self.epoch = checkpoint['epoch']
        
        print(f"✅ Loaded checkpoint from epoch {self.epoch} with Dice: {self.best_dice:.4f}")