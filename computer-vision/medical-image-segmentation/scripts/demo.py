#!/usr/bin/env python3
"""
Demo script for medical image segmentation
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_image_segmentation.data.synthetic_data import MedicalDataGenerator
from medical_image_segmentation.models.unet import create_model
from medical_image_segmentation.inference.predictor import MedicalPredictor
from medical_image_segmentation.utils.device_utils import setup_device


def parse_args():
    parser = argparse.ArgumentParser(description='Demo medical segmentation model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                       help='Path to output directory for demo results')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of demo samples to generate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for inference (auto/cuda/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    predictor = MedicalPredictor(args.model_path, device)
    print(f"✅ Model loaded from {args.model_path}")
    
    # Generate synthetic data for demo
    generator = MedicalDataGenerator(image_size=(256, 256))
    
    print(f"🎨 Generating {args.num_samples} demo samples...")
    
    for i in range(args.num_samples):
        # Generate synthetic medical image
        image, true_mask = generator.generate_medical_image()
        
        # Convert to proper format for prediction
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Make prediction
        prediction = predictor.predict(image_uint8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # True mask
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(image, cmap='gray')
        axes[3].imshow(prediction, cmap='jet', alpha=0.5)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        demo_path = os.path.join(args.output_dir, f'demo_sample_{i+1:02d}.png')
        plt.savefig(demo_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved demo sample {i+1} to {demo_path}")
    
    # Create a comparison grid
    create_comparison_grid(predictor, generator, args.output_dir, args.num_samples)
    
    print(f"🎉 Demo completed! Results saved to {args.output_dir}")


def create_comparison_grid(predictor, generator, output_dir, num_samples=5):
    """Create a grid comparison of multiple samples"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Generate sample
        image, true_mask = generator.generate_medical_image()
        image_uint8 = (image * 255).astype(np.uint8)
        prediction = predictor.predict(image_uint8)
        
        # Calculate metrics for this sample
        from medical_image_segmentation.evaluation.metrics import calculate_metrics
        metrics = calculate_metrics(
            torch.tensor(prediction).unsqueeze(0).unsqueeze(0),
            torch.tensor(true_mask).unsqueeze(0).unsqueeze(0)
        )
        
        # Original
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1} - Original')
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(prediction, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay with metrics
        axes[i, 3].imshow(image, cmap='gray')
        axes[i, 3].imshow(prediction, cmap='jet', alpha=0.5)
        axes[i, 3].set_title(f'Overlay\nDice: {metrics["dice"]:.3f}, IoU: {metrics["iou"]:.3f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'demo_comparison_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison grid saved to {grid_path}")


if __name__ == '__main__':
    main()