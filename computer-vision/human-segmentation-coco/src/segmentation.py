import torch
import torchvision
import numpy as np
from ultralytics import YOLO
import cv2

class HumanSegmentator:
    def __init__(self, model_type='yolo', device='auto', conf_threshold=0.25):
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.device = device
        
        if model_type == 'yolo':
            self.model = YOLO('yolov8n-seg.pt')
            if device != 'auto':
                self.model.to(device)
        elif model_type == 'mask_rcnn':
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            self.model.to(self.device)
            self.model.eval()
    
    def segment_with_yolo(self, img, target_size=(512, 512)):
        """Segment objects using YOLOv8 - IMPROVED VERSION"""
        # Convert to uint8 if needed
        if img.dtype == np.float32:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img
        
        # Store original shape for mask resizing
        original_shape = img_uint8.shape[:2]
        
        # Run inference
        results = self.model(img_uint8, conf=self.conf_threshold, verbose=False)
        
        # Create empty mask
        mask = np.zeros(original_shape, dtype=np.float32)
        
        # Process results
        for result in results:
            if result.masks is not None:
                for m in result.masks:
                    # Get mask and resize to original image size
                    mask_data = m.data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_data, 
                                            (original_shape[1], original_shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                    mask = np.logical_or(mask, mask_resized)
        
        return mask.astype(np.float32)
    
    def segment(self, img, target_size=(512, 512)):
        """Main segmentation method"""
        if self.model_type == 'yolo':
            return self.segment_with_yolo(img, target_size)
        elif self.model_type == 'mask_rcnn':
            return self.segment_with_mask_rcnn(img)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")