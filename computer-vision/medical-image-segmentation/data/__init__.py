# Полностью перезапишем файл
echo "from .dataset import MedicalDataset
from .synthetic_data import create_medical_dataset

__all__ = ['MedicalDataset', 'create_medical_dataset']" > medical_image_segmentation/data/__init__.py