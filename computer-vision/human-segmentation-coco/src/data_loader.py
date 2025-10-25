import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import requests
import zipfile
from tqdm import tqdm
import warnings

class COCODataLoader:
    def __init__(self, data_dir='coco', data_type='train2017'):
        self.data_dir = data_dir
        self.data_type = data_type
        self.coco = None
        self.cat_ids = None
        
    def download_dataset(self, num_images=1000):
        """Download COCO dataset subset"""
        os.makedirs(f'{self.data_dir}/annotations', exist_ok=True)
        os.makedirs(f'{self.data_dir}/{self.data_type}', exist_ok=True)
        
        # Download annotations
        annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        annotations_zip_path = f'{self.data_dir}/annotations_trainval2017.zip'
        
        if not os.path.exists(annotations_zip_path):
            print("Downloading annotations...")
            self._download_file(annotations_url, annotations_zip_path)
            
            print("Extracting annotations...")
            with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir + '/')
        
        # Download sample images
        train_images_url = 'http://images.cocodataset.org/zips/train2017.zip'
        train_zip_path = f'{self.data_dir}/train2017.zip'
        
        if not os.path.exists(train_zip_path):
            print("Downloading train images...")
            self._download_file(train_images_url, train_zip_path)
            
            print("Extracting train images...")
            with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir + '/')
    
    def _download_file(self, url, save_path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
    
    def initialize_coco(self):
        """Initialize COCO API"""
        ann_file = f'{self.data_dir}/annotations/instances_{self.data_type}.json'
        try:
            self.coco = COCO(ann_file)
            self.cat_ids = self.coco.getCatIds(catNms=['person'])
            print(f"COCO API initialized successfully!")
            print(f"Number of images: {len(self.coco.imgs)}")
            return True
        except Exception as e:
            print(f"Error initializing COCO API: {e}")
            return False
    
    def get_human_images(self, num_images=1000):
        """Get image IDs containing humans"""
        if self.coco is None:
            if not self.initialize_coco():
                return []
        
        img_ids = self.coco.getImgIds(catIds=self.cat_ids)[:num_images]
        print(f"Found {len(img_ids)} images containing people")
        return img_ids
    
    def load_image_and_mask(self, img_id, target_size=(512, 512)):
        """Load and preprocess image with segmentation mask"""
        try:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = f"{self.data_dir}/{self.data_type}/{img_info['file_name']}"
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_size = img.shape[:2]
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            
            # Create segmentation mask
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            
            mask = np.zeros(target_size[::-1], dtype=np.float32)  # (height, width)
            
            for ann in anns:
                if 'segmentation' in ann and ann['segmentation']:
                    seg = ann['segmentation']
                    if isinstance(seg, list):
                        for poly in seg:
                            poly = np.array(poly).reshape((-1, 2))
                            # Scale polygon to target size
                            poly[:, 0] = poly[:, 0] * (target_size[0] / img_info['width'])
                            poly[:, 1] = poly[:, 1] * (target_size[1] / img_info['height'])
                            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            
            return {
                'image': img,
                'mask': mask,
                'image_id': img_id,
                'original_size': original_size,
                'target_size': target_size
            }
            
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            # Return empty arrays as fallback
            empty_img = np.zeros((*target_size, 3), dtype=np.float32)
            empty_mask = np.zeros(target_size[::-1], dtype=np.float32)
            return {
                'image': empty_img,
                'mask': empty_mask,
                'image_id': img_id,
                'original_size': target_size,
                'target_size': target_size
            }