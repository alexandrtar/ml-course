#!/usr/bin/env python3
"""
Evaluation script for medical image segmentation models
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_image_segmentation.data.dataset import MedicalDataset
from medical_image_segmentation.models.unet import create_model
from medical_image_segmentation.evaluation.evaluator import MedicalEvaluator
from medical_image_segmentation.evaluation.visualization import ResultVisualizer
from medical_image_segmentation.utils.device_utils import setup_device


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate medical segmentation model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to evaluation data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Path to output directory for results')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for evaluation (auto/cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(
        model_name=config['model']['name'],
        n_channels=config['model']['in_channels'],
        n_classes=config['model']['out_channels'],
        features=config['model']['features'],
        dropout=config['model']['dropout']
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {args.model_path}")
    
    # Create dataset and data loader
    dataset = MedicalDataset(
        data_dir=args.data_dir,
        image_size=config['training']['image_size'],
        augment=False
    )
    
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"📊 Evaluation samples: {len(dataset)}")
    
    # Create evaluator
    evaluator = MedicalEvaluator(model, device)
    
    # Run evaluation
    print("🔍 Running evaluation...")
    metrics = evaluator.evaluate(data_loader)
    
    # Print results
    print("\n📈 Evaluation Results:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric:>15}: {value:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write("Medical Segmentation Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric:>15}: {value:.4f}\n")
    
    print(f"💾 Results saved to {results_file}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    visualizer = ResultVisualizer(model, device, args.output_dir)
    visualizer.create_visualizations(data_loader, num_samples=args.num_samples)
    
    # Create comprehensive report
    report_file = os.path.join(args.output_dir, 'evaluation_report.md')
    with open(report_file, 'w') as f:
        f.write("# Medical Segmentation Evaluation Report\n\n")
        f.write("## Metrics Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric, value in metrics.items():
            f.write(f"| {metric} | {value:.4f} |\n")
        
        f.write("\n## Model Information\n\n")
        f.write(f"- Model: {config['model']['name']}\n")
        f.write(f"- Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"- Dataset: {args.data_dir}\n")
        f.write(f"- Samples: {len(dataset)}\n")
    
    print(f"📄 Report saved to {report_file}")
    print("🎉 Evaluation completed successfully!")


if __name__ == '__main__':
    main()