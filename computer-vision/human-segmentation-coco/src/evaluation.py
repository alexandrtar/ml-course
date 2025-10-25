import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class SegmentationEvaluator:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_iou(self, mask1, mask2):
        """
        Calculate Intersection over Union (IoU) between two masks
        
        Args:
            mask1, mask2: numpy arrays with binary values (0 or 1)
        
        Returns:
            IoU score (0 to 1)
        """
        # Ensure masks are binary
        mask1_binary = (mask1 > 0.5).astype(np.float32)
        mask2_binary = (mask2 > 0.5).astype(np.float32)
        
        intersection = np.logical_and(mask1_binary, mask2_binary).sum()
        union = np.logical_or(mask1_binary, mask2_binary).sum()
        
        return intersection / (union + 1e-6)  # Avoid division by zero
    
    def calculate_dice(self, mask1, mask2):
        """
        Calculate Dice coefficient between two masks
        """
        mask1_binary = (mask1 > 0.5).astype(np.float32)
        mask2_binary = (mask2 > 0.5).astype(np.float32)
        
        intersection = np.logical_and(mask1_binary, mask2_binary).sum()
        return (2. * intersection) / (mask1_binary.sum() + mask2_binary.sum() + 1e-6)
    
    def calculate_precision_recall(self, true_mask, pred_mask):
        """
        Calculate precision and recall for binary segmentation
        """
        true_flat = (true_mask > 0.5).flatten()
        pred_flat = (pred_mask > 0.5).flatten()
        
        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)
        
        return precision, recall, f1
    
    def evaluate_single(self, true_mask, pred_mask):
        """
        Comprehensive evaluation for single image
        """
        iou = self.calculate_iou(true_mask, pred_mask)
        dice = self.calculate_dice(true_mask, pred_mask)
        precision, recall, f1 = self.calculate_precision_recall(true_mask, pred_mask)
        
        metrics = {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def evaluate_batch(self, true_masks, pred_masks):
        """
        Evaluate batch of images
        """
        batch_metrics = []
        for true_mask, pred_mask in zip(true_masks, pred_masks):
            metrics = self.evaluate_single(true_mask, pred_mask)
            batch_metrics.append(metrics)
        
        return self.aggregate_metrics(batch_metrics)
    
    def aggregate_metrics(self, metrics_list):
        """
        Aggregate metrics across multiple images
        """
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [metrics[key] for metrics in metrics_list]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def print_metrics(self, metrics):
        """Print formatted metrics"""
        print("📊 Segmentation Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            if 'mean' in key:
                print(f"{key.replace('_mean', '').upper():<12}: {value:.4f}")
    
    def plot_metrics_comparison(self, metrics_list, model_names):
        """Plot comparison of different models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics_to_plot = ['iou', 'dice', 'precision', 'recall']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = [m[f'{metric}_mean'] for m in metrics_list]
            
            bars = ax.bar(model_names, values)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel('Score')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig