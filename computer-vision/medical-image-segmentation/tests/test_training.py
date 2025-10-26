import pytest
import torch
import tempfile
import os
from pathlib import Path

from medical_image_segmentation.training.trainer import MedicalTrainer
from medical_image_segmentation.training.metrics import calculate_metrics
from medical_image_segmentation.utils.logger import TrainingLogger


class TestTrainingComponents:
    """Test training components"""
    
    def test_metrics_calculation(self, sample_batch):
        """Test metrics calculation"""
        predictions, targets = sample_batch
        
        # Apply sigmoid for proper probability range
        predictions = torch.sigmoid(predictions)
        
        metrics = calculate_metrics(predictions, targets)
        
        expected_metrics = ['dice', 'iou', 'precision', 'recall']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert 0.0 <= metrics[metric] <= 1.0
    
    def test_trainer_initialization(self, unet_model, sample_batch):
        """Test trainer initialization"""
        from torch.utils.data import DataLoader, TensorDataset
        
        images, masks = sample_batch
        dataset = TensorDataset(images, masks)
        
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
        
        trainer = MedicalTrainer(
            model=unet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device='cpu'
        )
        
        assert trainer.model == unet_model
        assert trainer.device == torch.device('cpu')
        assert trainer.best_dice == 0.0
        assert len(trainer.history) == 7  # train_loss, val_loss, dice, iou, precision, recall, learning_rate
    
    def test_train_epoch(self, unet_model, sample_batch):
        """Test training for one epoch"""
        from torch.utils.data import DataLoader, TensorDataset
        
        images, masks = sample_batch
        dataset = TensorDataset(images, masks)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
        
        trainer = MedicalTrainer(
            model=unet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device='cpu'
        )
        
        # Test training for one epoch
        train_loss = trainer.train_epoch()
        
        assert isinstance(train_loss, float)
        assert train_loss >= 0.0
    
    def test_validation_epoch(self, unet_model, sample_batch):
        """Test validation for one epoch"""
        from torch.utils.data import DataLoader, TensorDataset
        
        images, masks = sample_batch
        dataset = TensorDataset(images, masks)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
        
        trainer = MedicalTrainer(
            model=unet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device='cpu'
        )
        
        # Test validation
        val_loss, metrics = trainer.validate_epoch()
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0.0
        assert isinstance(metrics, dict)
        assert 'dice' in metrics
        assert 'iou' in metrics
    
    def test_model_checkpointing(self, unet_model, sample_batch, tmp_path):
        """Test model checkpoint saving and loading"""
        from torch.utils.data import DataLoader, TensorDataset
        
        images, masks = sample_batch
        dataset = TensorDataset(images, masks)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
        
        trainer = MedicalTrainer(
            model=unet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device='cpu'
        )
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        trainer.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Create new trainer and load checkpoint
        new_trainer = MedicalTrainer(
            model=unet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device='cpu'
        )
        
        new_trainer.load_checkpoint(str(checkpoint_path))
        
        assert new_trainer.epoch == trainer.epoch
        assert new_trainer.best_dice == trainer.best_dice


class TestLogger:
    """Test training logger"""
    
    def test_logger_initialization(self, tmp_path):
        """Test logger initialization"""
        logger = TrainingLogger(
            experiment_name="test_experiment",
            run_name="test_run",
            use_mlflow=False,
            use_wandb=False,
            log_dir=str(tmp_path)
        )
        
        assert logger.experiment_name == "test_experiment"
        assert logger.run_name == "test_run"
        assert not logger.use_mlflow
        assert not logger.use_wandb
        
        # Cleanup
        logger.finish()
    
    def test_logger_params(self, tmp_path):
        """Test parameter logging"""
        logger = TrainingLogger(
            experiment_name="test_experiment",
            run_name="test_run",
            use_mlflow=False,
            use_wandb=False,
            log_dir=str(tmp_path)
        )
        
        params = {
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 10
        }
        
        logger.log_params(params)
        
        # Check if log file was created
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        
        logger.finish()
    
    def test_logger_metrics(self, tmp_path):
        """Test metrics logging"""
        logger = TrainingLogger(
            experiment_name="test_experiment",
            run_name="test_run",
            use_mlflow=False,
            use_wandb=False,
            log_dir=str(tmp_path)
        )
        
        metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "dice": 0.7
        }
        
        logger.log_metrics(metrics, step=1)
        
        logger.finish()


def test_end_to_end_training(unet_model, synthetic_dataset):
    """Test end-to-end training with synthetic data"""
    from medical_image_segmentation.data.dataset import MedicalDataset
    from torch.utils.data import DataLoader
    
    # Create small datasets for quick testing
    train_dataset = MedicalDataset(
        data_dir=synthetic_dataset,
        image_size=(64, 64),
        augment=False
    )
    
    val_dataset = MedicalDataset(
        data_dir=synthetic_dataset,
        image_size=(64, 64),
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
    
    trainer = MedicalTrainer(
        model=unet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device='cpu'
    )
    
    # Train for a few epochs
    history = trainer.train(epochs=2)
    
    assert isinstance(history, dict)
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert 'dice' in history
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2
    
    # Check if model improved
    assert trainer.best_dice > 0.0