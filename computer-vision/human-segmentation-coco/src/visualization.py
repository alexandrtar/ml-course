import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

class ResultsVisualizer:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_comparison(self, original_img, true_mask, pred_mask, metrics, 
                       image_id=None, save_path=None):
        """
        Create comprehensive comparison plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth mask
        axes[0, 1].imshow(true_mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # Prediction mask
        axes[0, 2].imshow(pred_mask, cmap='gray')
        axes[0, 2].set_title('Prediction Mask')
        axes[0, 2].axis('off')
        
        # Overlay: Ground truth on original
        axes[1, 0].imshow(original_img)
        axes[1, 0].imshow(true_mask, cmap='jet', alpha=0.5)
        axes[1, 0].set_title('GT Mask Overlay')
        axes[1, 0].axis('off')
        
        # Overlay: Prediction on original
        axes[1, 1].imshow(original_img)
        axes[1, 1].imshow(pred_mask, cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Pred Mask Overlay')
        axes[1, 1].axis('off')
        
        # Metrics text
        axes[1, 2].axis('off')
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items() 
                                if not k.endswith(('std', 'min', 'max'))])
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, va='center')
        
        if image_id:
            plt.suptitle(f'Image ID: {image_id}', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_confidence_threshold_plot(self, iou_scores, conf_thresholds, save_path=None):
        """Plot IoU vs Confidence Threshold"""
        plt.figure(figsize=(10, 6))
        plt.plot(conf_thresholds, iou_scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('IoU Score')
        plt.title('IoU vs Confidence Threshold')
        plt.grid(True, alpha=0.3)
        
        # Annotate best threshold
        best_idx = np.argmax(iou_scores)
        plt.annotate(f'Best: {iou_scores[best_idx]:.3f}\nThreshold: {conf_thresholds[best_idx]}',
                    xy=(conf_thresholds[best_idx], iou_scores[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return plt.gcf()
    
    def save_results(self, original_img, true_mask, pred_mask, metrics, 
                    image_id, subfolder=''):
        """Save results to file"""
        if subfolder:
            save_folder = os.path.join(self.save_dir, subfolder)
            os.makedirs(save_folder, exist_ok=True)
        else:
            save_folder = self.save_dir
        
        save_path = os.path.join(save_folder, f'result_{image_id}.png')
        self.plot_comparison(original_img, true_mask, pred_mask, metrics, 
                           image_id, save_path)
        plt.close()
        
        return save_path