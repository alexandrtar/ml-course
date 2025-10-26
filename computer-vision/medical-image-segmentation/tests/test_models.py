import pytest
import torch
import numpy as np

from medical_image_segmentation.models.unet import UNet, DoubleConv, AttentionGate
from medical_image_segmentation.models.losses import (
    DiceLoss, IoULoss, FocalLoss, TverskyLoss, CombinedLoss, create_loss
)


class TestUNet:
    """Test UNet model and components"""
    
    def test_double_conv(self):
        """Test DoubleConv block"""
        conv_block = DoubleConv(3, 64, dropout=0.1)
        
        # Test forward pass
        x = torch.randn(2, 3, 64, 64)
        output = conv_block(x)
        
        assert output.shape == (2, 64, 64, 64)
        assert not torch.isnan(output).any()
    
    def test_attention_gate(self):
        """Test attention gate"""
        attention_gate = AttentionGate(64, 64, 32)
        
        g = torch.randn(2, 64, 32, 32)  # Gate signal
        x = torch.randn(2, 64, 64, 64)  # Skip connection
        
        output = attention_gate(g, x)
        
        assert output.shape == (2, 64, 64, 64)
        assert not torch.isnan(output).any()
    
    def test_unet_initialization(self):
        """Test UNet initialization"""
        model = UNet(
            n_channels=1,
            n_classes=2,
            features=[32, 64],  # Smaller for testing
            dropout=0.1,
            normalization="batch_norm",
            activation="relu"
        )
        
        assert model.n_channels == 1
        assert model.n_classes == 2
        assert model.features == [32, 64]
        assert len(model.encoder) == 2
        assert len(model.decoder) == 1
        assert len(model.upconvs) == 1
    
    def test_unet_forward(self, unet_model, sample_batch):
        """Test UNet forward pass"""
        images, _ = sample_batch
        
        with torch.no_grad():
            output = unet_model(images)
        
        assert output.shape == (4, 1, 256, 256)  # (batch, classes, height, width)
        assert not torch.isnan(output).any()
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)  # Sigmoid output
    
    def test_unet_with_attention(self):
        """Test UNet with attention gates"""
        model = UNet(
            n_channels=1,
            n_classes=1,
            features=[32, 64],
            use_attention=True
        )
        
        x = torch.randn(2, 1, 128, 128)
        output = model(x)
        
        assert output.shape == (2, 1, 128, 128)
        assert model.attention_gates is not None
        assert len(model.attention_gates) == 1
    
    def test_unet_different_configurations(self):
        """Test UNet with different configurations"""
        configs = [
            {"normalization": "batch_norm", "activation": "relu"},
            {"normalization": "instance_norm", "activation": "leaky_relu"},
            {"normalization": "group_norm", "activation": "elu"},
        ]
        
        for config in configs:
            model = UNet(
                n_channels=3,
                n_classes=1,
                features=[16, 32],
                **config
            )
            
            x = torch.randn(1, 3, 64, 64)
            output = model(x)
            
            assert output.shape == (1, 1, 64, 64)
            assert not torch.isnan(output).any()


class TestLossFunctions:
    """Test loss functions"""
    
    def test_dice_loss(self, sample_batch):
        """Test Dice loss"""
        dice_loss = DiceLoss()
        predictions, targets = sample_batch
        
        # Apply sigmoid to predictions for proper range
        predictions = torch.sigmoid(predictions)
        
        loss = dice_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss >= 0.0 and loss <= 1.0
    
    def test_iou_loss(self, sample_batch):
        """Test IoU loss"""
        iou_loss = IoULoss()
        predictions, targets = sample_batch
        
        predictions = torch.sigmoid(predictions)
        loss = iou_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss >= 0.0 and loss <= 1.0
    
    def test_focal_loss(self, sample_batch):
        """Test Focal loss"""
        focal_loss = FocalLoss()
        predictions, targets = sample_batch
        
        predictions = torch.sigmoid(predictions)
        loss = focal_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss >= 0.0
    
    def test_tversky_loss(self, sample_batch):
        """Test Tversky loss"""
        tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        predictions, targets = sample_batch
        
        predictions = torch.sigmoid(predictions)
        loss = tversky_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss >= 0.0 and loss <= 1.0
    
    def test_combined_loss(self, sample_batch):
        """Test combined loss"""
        combined_loss = CombinedLoss(
            dice_weight=0.6,
            bce_weight=0.4,
            iou_weight=0.0
        )
        predictions, targets = sample_batch
        
        predictions = torch.sigmoid(predictions)
        loss = combined_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss >= 0.0
    
    def test_loss_factory(self):
        """Test loss factory function"""
        losses_to_test = ["dice", "bce", "iou", "focal", "tversky", "combined"]
        
        for loss_name in losses_to_test:
            loss_fn = create_loss(loss_name)
            assert loss_fn is not None
            
            # Test forward pass with dummy data
            predictions = torch.sigmoid(torch.randn(2, 1, 64, 64))
            targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
            
            loss = loss_fn(predictions, targets)
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()


def test_model_parameters_count(unet_model):
    """Test model parameters count"""
    total_params = sum(p.numel() for p in unet_model.parameters())
    trainable_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params > 0
    assert total_params == trainable_params  # All parameters should be trainable


def test_model_gradient_flow(unet_model, sample_batch):
    """Test gradient flow through model"""
    images, masks = sample_batch
    
    # Forward pass
    outputs = unet_model(images)
    loss = torch.nn.functional.binary_cross_entropy(outputs, masks)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for name, param in unet_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"