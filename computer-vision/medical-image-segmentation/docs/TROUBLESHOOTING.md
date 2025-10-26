# 🔧 Troubleshooting Guide

Common issues and solutions for the medical image segmentation project.

## Installation Issues

### CUDA Compatibility

**Problem**: CUDA not available or version mismatch.

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
Verification:

python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
Dependency Conflicts
Problem: Package version conflicts during installation.

Solution:

bash
# Create fresh virtual environment
python -m venv medseg_env
source medseg_env/bin/activate  # Linux/Mac
# OR
medseg_env\Scripts\activate    # Windows

# Install with conda (alternative)
conda create -n medseg_env python=3.9
conda activate medseg_env
Memory Issues
Problem: Out of memory errors during training.

Solutions:

Reduce batch size:

yaml
# In config/training_config.yaml
training:
  batch_size: 4  # Instead of 8
Use gradient accumulation:

python
# In training script
accumulation_steps = 4
loss = loss / accumulation_steps
loss.backward()

if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
Enable mixed precision:

python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
Training Issues
Loss Not Decreasing
Problem: Training loss stagnates or increases.

Solutions:

Check learning rate:

python
# Try different learning rates
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower LR
Verify data pipeline:

python
# Check if data is loading correctly
from medical_image_segmentation.data import MedicalDataset

dataset = MedicalDataset("data/train")
sample_image, sample_mask = dataset[0]
print(f"Image range: {sample_image.min()} - {sample_image.max()}")
print(f"Mask unique values: {torch.unique(sample_mask)}")
Inspect model output:

python
# Check model predictions
model.eval()
with torch.no_grad():
    sample_output = model(sample_image.unsqueeze(0))
    print(f"Output range: {sample_output.min()} - {sample_output.max()}")
Overfitting
Problem: Model performs well on training data but poorly on validation.

Solutions:

Increase regularization:

yaml
# In config/training_config.yaml
model:
  dropout: 0.5  # Increase dropout

training:
  weight_decay: 0.0001  # Add weight decay
Add more augmentations:

python
# In data/augmentations.py
augmentations = MedicalAugmentations(
    image_size=(256, 256),
    use_elastic=True,
    use_grid_distortion=True
)
Use early stopping:

python
# In training/trainer.py
early_stopping = EarlyStopping(patience=10)
if early_stopping(val_loss):
    break
Gradient Explosion/Vanishing
Problem: Gradients become too large or too small.

Solutions:

Gradient clipping:

python
# In training loop
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
Proper weight initialization:

python
# In model definition
def _initialize_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
Batch normalization:

python
# Ensure batch norm is used
model = UNet(normalization="batch_norm")
Data Issues
Data Loading Problems
Problem: Errors when loading images or masks.

Solutions:

Check file formats:

bash
# Verify image files
file data/train/images/sample_001.png
Validate data structure:

text
data/
├── train/
│   ├── images/
│   │   ├── sample_001.png
│   │   └── ...
│   └── masks/
│       ├── sample_001.png
│       └── ...
└── val/
    ├── images/
    └── masks/
Use synthetic data for testing:

python
from medical_image_segmentation.data import create_medical_dataset
create_medical_dataset("test_data", num_samples=100)
Memory Issues with Large Datasets
Problem: Running out of memory with large datasets.

Solutions:

Use data loader with multiple workers:

python
train_loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True
)
Implement progressive loading:

python
class ProgressiveMedicalDataset(MedicalDataset):
    def __getitem__(self, idx):
        # Load only when needed
        return self._load_sample(idx)
Model Issues
Poor Segmentation Quality
Problem: Model produces low-quality segmentations.

Solutions:

Try different loss functions:

python
from medical_image_segmentation.models.losses import CombinedLoss

criterion = CombinedLoss(
    dice_weight=0.7,
    bce_weight=0.3,
    focal_weight=0.0
)
Adjust model architecture:

python
# Deeper network
model = UNet(features=[64, 128, 256, 512, 1024, 2048])

# Or wider network
model = UNet(features=[128, 256, 512, 1024, 2048])
Post-processing:

python
# Apply morphological operations
import cv2

def postprocess_mask(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask_clean
Slow Training
Problem: Training is slower than expected.

Solutions:

Enable cuDNN benchmark:

python
torch.backends.cudnn.benchmark = True
Use larger batch sizes:

python
# If memory allows
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
Optimize data loading:

python
# Use pinned memory
train_loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
Deployment Issues
Model Loading Errors
Problem: Cannot load saved model.

Solutions:

Check model compatibility:

python
# Ensure model architecture matches
model = UNet(n_channels=1, n_classes=1)
checkpoint = torch.load("model.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
Handle device mapping:

python
# Load on CPU regardless of training device
checkpoint = torch.load("model.pth", map_location=lambda storage, loc: storage)
API Deployment Problems
Problem: FastAPI server not working correctly.

Solutions:

Check port availability:

bash
# Check if port is in use
netstat -tulpn | grep 8000

# Use different port
python inference/api.py --port 8080
Verify model path:

python
# Ensure model exists
import os
assert os.path.exists("checkpoints/best_model.pth"), "Model file not found"
Test API locally:

bash
curl -X POST "http://localhost:8000/health"
Performance Optimization
GPU Utilization
Problem: Low GPU utilization during training.

Diagnosis:

python
# Monitor GPU usage
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
Solutions:

Increase batch size:

python
# Find optimal batch size
for batch_size in [4, 8, 16, 32]:
    try:
        train_loader = DataLoader(dataset, batch_size=batch_size)
        # Test one iteration
        break
    except RuntimeError:  # Out of memory
        continue
Use data prefetching:

python
class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
Memory Leaks
Problem: Memory usage increases over time.

Diagnosis:

python
# Track memory usage
import gc
import torch

def get_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**3
    return 0

print(f"Memory: {get_memory_usage():.2f} GB")
Solutions:

Clear cache:

python
# In training loop
torch.cuda.empty_cache()
gc.collect()
Use gradient checkpointing:

python
model = UNet()
model.set_gradient_checkpointing(True)
Common Error Messages
"CUDA out of memory"
Solution: Reduce batch size or use gradient accumulation.

"FileNotFoundError" for data
Solution: Verify data paths and directory structure.

"KeyError" in model loading
Solution: Check model state_dict keys match current model.

"Shape mismatch" errors
Solution: Verify input dimensions and model architecture.

Getting Help
If you encounter issues not covered here:

Check the documentation in docs/

Review the test cases in tests/

Examine the example notebooks in notebooks/

Create an issue on GitHub with:

Error message and traceback

Your environment details

Steps to reproduce

Relevant code and configuration

Environment Details
Include these when reporting issues:

python
import torch, sys
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Python: {sys.version}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
This troubleshooting guide covers the most common scenarios. For specific issues, refer to the module-specific documentation or create a detailed issue report.

text

## 🔧 **GITHUB ACTIONS**

### `.github/workflows/ci-cd.yml`
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8
    
    - name: Check code formatting with black
      run: |
        black --check medical_image_segmentation tests scripts
    
    - name: Lint with flake8
      run: |
        flake8 medical_image_segmentation tests scripts --count --show-source --statistics
    
    - name: Run tests with pytest
      run: |
        pytest tests/ -v --cov=medical_image_segmentation --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  docker-build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t medical-segmentation:${{ github.sha }} .
    
    - name: Log in to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Push Docker image
      run: |
        docker tag medical-segmentation:${{ github.sha }} ${{ secrets.DOCKERHUB_USERNAME }}/medical-segmentation:latest
        docker push ${{ secrets.DOCKERHUB_USERNAME }}/medical-segmentation:latest

  deploy-docs:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs