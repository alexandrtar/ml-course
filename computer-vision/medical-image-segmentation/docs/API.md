# 🏥 Medical Image Segmentation API Documentation

## Overview

This document describes the API for the medical image segmentation project. The project provides a complete pipeline for training, evaluating, and deploying UNet-based models for medical image segmentation.

## Core Modules

### Data Module

#### `MedicalDataset`

The main dataset class for medical image segmentation.

```python
from medical_image_segmentation.data import MedicalDataset

dataset = MedicalDataset(
    data_dir="path/to/data",
    image_size=(256, 256),
    augment=True,
    normalize=True
)
Parameters:

data_dir: Path to dataset directory

image_size: Target image size (height, width)

augment: Whether to apply data augmentation

normalize: Whether to normalize images

MedicalDataGenerator
Generates synthetic medical data for training and testing.

python
from medical_image_segmentation.data import MedicalDataGenerator

generator = MedicalDataGenerator(image_size=(256, 256))
image, mask = generator.generate_medical_image()
Models Module
UNet
The main UNet architecture for segmentation.

python
from medical_image_segmentation.models import UNet

model = UNet(
    n_channels=1,
    n_classes=1,
    features=[64, 128, 256, 512, 1024],
    dropout=0.2,
    normalization="batch_norm",
    activation="relu"
)
Loss Functions
Various loss functions for segmentation tasks:

python
from medical_image_segmentation.models.losses import (
    DiceLoss, IoULoss, FocalLoss, CombinedLoss, create_loss
)

# Individual losses
dice_loss = DiceLoss()
iou_loss = IoULoss()

# Combined loss
combined_loss = CombinedLoss(
    dice_weight=0.7,
    bce_weight=0.3
)

# Factory function
loss_fn = create_loss("combined", dice_weight=0.7, bce_weight=0.3)
Training Module
MedicalTrainer
The main training class with comprehensive features.

python
from medical_image_segmentation.training import MedicalTrainer

trainer = MedicalTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    logger=logger
)

history = trainer.train(epochs=100)
Inference Module
MedicalPredictor
For making predictions with trained models.

python
from medical_image_segmentation.inference import MedicalPredictor

predictor = MedicalPredictor(
    model_path="path/to/model.pth",
    device="cuda"
)

prediction = predictor.predict(image)
mask = predictor.predict_batch(images)
FastAPI Server
Start a web server for model inference:

bash
python inference/api.py --host 0.0.0.0 --port 8000
Scripts
Training
bash
# Basic training
python scripts/train.py --config config/training_config.yaml

# With custom parameters
python scripts/train.py \
    --data_dir data/medical_data \
    --batch_size 16 \
    --epochs 100 \
    --device cuda
Evaluation
bash
python scripts/evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/medical_data/val
Demo
bash
python scripts/demo.py \
    --model_path checkpoints/best_model.pth \
    --num_samples 5
Configuration
The project uses YAML configuration files. Main configuration files:

config/training_config.yaml: Training parameters

config/model_config.yaml: Model architectures

config/inference_config.yaml: Inference settings

Example configuration:

yaml
training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.001

model:
  name: "UNet"
  features: [64, 128, 256, 512, 1024]
  dropout: 0.2
Logging and Tracking
MLflow Integration
python
from medical_image_segmentation.utils.logger import setup_logger

logger = setup_logger(
    experiment_name="medical_segmentation",
    use_mlflow=True,
    use_wandb=False
)

logger.log_params(params)
logger.log_metrics(metrics, step=epoch)
Custom Logging
python
logger.log_artifact("model.pth")
logger.log_image(image, "prediction", step=epoch)
Utilities
Device Management
python
from medical_image_segmentation.utils.device_utils import setup_device

device = setup_device("auto")  # Automatically chooses CUDA if available
Configuration Loading
python
from medical_image_segmentation.utils.config_loader import load_config

config = load_config("config/training_config.yaml")
Testing
Run the test suite:

bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_models.py

# Run with coverage
pytest --cov=medical_image_segmentation
Examples
Complete Training Pipeline
python
import torch
from medical_image_segmentation.data import MedicalDataset, MedicalDataLoader
from medical_image_segmentation.models import UNet
from medical_image_segmentation.training import MedicalTrainer
from medical_image_segmentation.utils.logger import setup_logger

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = setup_logger("my_experiment")

# Data
train_loader, val_loader = MedicalDataLoader.create_loaders(
    train_dir="data/train",
    val_dir="data/val"
)

# Model
model = UNet(n_channels=1, n_classes=1).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = MedicalTrainer(model, train_loader, val_loader, optimizer, device, logger)
history = trainer.train(epochs=50)
Inference Example
python
from medical_image_segmentation.inference import MedicalPredictor
import cv2

# Load model
predictor = MedicalPredictor("checkpoints/best_model.pth")

# Load and preprocess image
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
prediction = predictor.predict(image)

# Visualize results
predictor.visualize_prediction(image, prediction)
For more detailed examples, check the notebooks/ directory and script examples.

text

### `docs/DEPLOYMENT.md`
```markdown
# 🚀 Deployment Guide

This guide covers various deployment options for the medical image segmentation project.

## Local Deployment

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/medical-image-segmentation-unet.git
cd medical-image-segmentation-unet
Install dependencies:

bash
pip install -r requirements.txt
Install in development mode:

bash
pip install -e .
Training Deployment
Prepare your data:

bash
# Use synthetic data for testing
python scripts/create_data.py --output_dir data/medical_data --num_samples 1000
Start training:

bash
python scripts/train.py --config config/training_config.yaml
Monitor training:

bash
# MLflow UI (if enabled)
mlflow ui --backend-store-uri mlruns
Inference Deployment
Start the API server:

bash
python inference/api.py --host 0.0.0.0 --port 8000
Test the API:

bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@test_image.png"
Docker Deployment
Build Docker Image
bash
docker build -t medical-segmentation .
Run Training in Docker
bash
docker run -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  medical-segmentation \
  python scripts/train.py --config config/training_config.yaml
Run API in Docker
bash
docker run -d --name medical-api \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  medical-segmentation \
  python inference/api.py --host 0.0.0.0 --port 8000
Docker Compose
bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
Cloud Deployment
AWS SageMaker
Create SageMaker training job:

python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='scripts/train.py',
    role='SageMakerRole',
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='2.0.0',
    py_version='py39',
    hyperparameters={
        'config': 'config/training_config.yaml',
        'epochs': 100
    }
)

estimator.fit({'training': 's3://my-bucket/data/'})
Deploy model endpoint:

python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
Google Cloud AI Platform
Submit training job:

bash
gcloud ai-platform jobs submit training medical_segmentation_$(date +%Y%m%d_%H%M%S) \
    --package-path medical_image_segmentation \
    --module-name scripts.train \
    --region us-central1 \
    --python-version 3.9 \
    --runtime-version 2.8 \
    --scale-tier BASIC_GPU \
    --stream-logs
Deploy to AI Platform:

bash
gcloud ai-platform models create medical_segmentation
gcloud ai-platform versions create v1 \
    --model medical_segmentation \
    --origin gs://my-bucket/model/ \
    --runtime-version 2.8 \
    --python-version 3.9 \
    --framework pytorch
Azure Machine Learning
Create workspace and compute:

python
from azureml.core import Workspace, Experiment, Environment
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()
compute_target = ws.compute_targets['gpu-cluster']

env = Environment.from_conda_specification(
    name='medical-segmentation',
    file_path='environment.yml'
)

experiment = Experiment(ws, 'medical-segmentation')
Submit training run:

python
from azureml.train.estimator import Estimator

estimator = Estimator(
    source_directory='.',
    entry_script='scripts/train.py',
    compute_target=compute_target,
    environment_definition=env,
    script_params={
        '--config': 'config/training_config.yaml'
    }
)

run = experiment.submit(estimator)
Kubernetes Deployment
Create Kubernetes Deployment
yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-segmentation-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: medical-segmentation
  template:
    metadata:
      labels:
        app: medical-segmentation
    spec:
      containers:
      - name: api
        image: medical-segmentation:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "2Gi"
            cpu: "1"
        env:
        - name: PYTHONPATH
          value: "/app"
Create Service
yaml
apiVersion: v1
kind: Service
metadata:
  name: medical-segmentation-service
spec:
  selector:
    app: medical-segmentation
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
Model Export
Export to TorchScript
python
from medical_image_segmentation.inference import export_model

export_model.export_to_torchscript(
    model_path="checkpoints/best_model.pth",
    output_path="model_scripted.pt",
    image_size=(256, 256)
)
Export to ONNX
python
export_model.export_to_onnx(
    model_path="checkpoints/best_model.pth",
    output_path="model.onnx",
    image_size=(256, 256)
)
Monitoring and Logging
Application Metrics
python
from prometheus_client import start_http_server, Counter, Histogram

# Metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction latency')

@PREDICTION_DURATION.time()
def predict(image):
    PREDICTION_COUNTER.inc()
    # Prediction logic
Health Checks
python
from healthcheck import HealthCheck

health = HealthCheck()

def model_available():
    return model is not None, "model loaded"

health.add_check(model_available)

# Add to FastAPI app
app.add_route("/health", health.run)
Performance Optimization
GPU Optimization
python
# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
Batch Processing
python
# Optimize batch size based on available memory
def find_optimal_batch_size(model, dataset, start_batch_size=8):
    # Implementation for automatic batch size tuning
    pass
Security Considerations
Input Validation
python
from pydantic import BaseModel, conint, confloat
from fastapi import HTTPException

class PredictionRequest(BaseModel):
    image: bytes
    confidence_threshold: confloat(ge=0.0, le=1.0) = 0.5

def validate_image(image: bytes):
    if len(image) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "Image too large")
Rate Limiting
python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("5/minute")
async def predict(request: Request):
    # Prediction logic
This deployment guide covers the most common scenarios. For specific cloud provider details or advanced configurations, refer to the respective provider's documentation.

text

### `docs/STRUCTURE.md`
```markdown
# 🏗️ Project Structure

This document describes the complete structure of the medical image segmentation project.

## Root Directory
medical-image-segmentation-unet/
├── 📁 config/ # Configuration files
├── 📁 data/ # Data handling and processing
├── 📁 models/ # Model architectures
├── 📁 training/ # Training logic and utilities
├── 📁 inference/ # Inference and deployment
├── 📁 evaluation/ # Model evaluation and metrics
├── 📁 utils/ # Utility functions
├── 📁 tests/ # Test suite
├── 📁 notebooks/ # Jupyter notebooks
├── 📁 scripts/ # Executable scripts
├── 📁 docs/ # Documentation
├── 📁 .github/ # GitHub workflows
├── 📄 requirements.txt # Python dependencies
├── 📄 setup.py # Package setup
├── 📄 pyproject.toml # Modern package configuration
├── 📄 Dockerfile # Container definition
├── 📄 docker-compose.yml # Multi-container setup
├── 📄 README.md # Project overview
├── 📄 CONTRIBUTING.md # Contribution guidelines
├── 📄 LICENSE # License information
├── 📄 .gitignore # Git ignore rules
└── 📄 .dockerignore # Docker ignore rules

text

## Detailed Structure

### 📁 config/
Configuration management using YAML files.
config/
├── init.py
├── training_config.yaml # Training hyperparameters
├── model_config.yaml # Model architectures
└── inference_config.yaml # Inference settings

text

**Key Features:**
- Centralized configuration management
- Environment-specific configurations
- Easy experimentation with different settings

### 📁 data/
Data loading, preprocessing, and augmentation.
data/
├── init.py
├── dataset.py # Main dataset class
├── preprocessing.py # Data preprocessing
├── synthetic_data.py # Synthetic data generation
├── augmentations.py # Data augmentation pipelines
└── data_loader.py # Data loader utilities

text

**Key Features:**
- Support for real and synthetic data
- Comprehensive data augmentation
- Efficient data loading with caching
- Medical image-specific preprocessing

### 📁 models/
Neural network architectures and loss functions.
models/
├── init.py
├── unet.py # UNet architecture
├── attention_unet.py # UNet with attention gates
├── losses.py # Loss functions
└── model_factory.py # Model creation factory

text

**Key Features:**
- Modular architecture design
- Multiple UNet variants
- Comprehensive loss functions
- Easy model extensibility

### 📁 training/
Training loops, metrics, and callbacks.
training/
├── init.py
├── trainer.py # Main training class
├── metrics.py # Evaluation metrics
├── callbacks.py # Training callbacks
└── early_stopping.py # Early stopping implementation

text

**Key Features:**
- Professional training pipeline
- Comprehensive metric tracking
- Flexible callback system
- Support for distributed training

### 📁 inference/
Model inference and deployment.
inference/
├── init.py
├── predictor.py # Prediction utilities
├── demo.py # Demonstration scripts
└── api.py # FastAPI web server

text

**Key Features:**
- Batch and single prediction
- Web API for model serving
- Real-time demonstration
- Model export capabilities

### 📁 evaluation/
Model evaluation and visualization.
evaluation/
├── init.py
├── evaluator.py # Model evaluation
├── visualization.py # Result visualization
└── metrics_calculation.py # Metric computations

text

**Key Features:**
- Comprehensive model evaluation
- Advanced visualization tools
- Statistical analysis
- Performance benchmarking

### 📁 utils/
Utility functions and helpers.
utils/
├── init.py
├── logger.py # Logging utilities
├── helpers.py # Helper functions
├── config_loader.py # Configuration loading
└── device_utils.py # Device management

text

**Key Features:**
- Centralized logging
- Configuration management
- Device optimization
- Common utilities

### 📁 tests/
Comprehensive test suite.
tests/
├── init.py
├── conftest.py # Test configuration
├── test_data.py # Data module tests
├── test_models.py # Model tests
└── test_training.py # Training tests

text

**Key Features:**
- Unit tests for all modules
- Integration tests
- Test fixtures and utilities
- Continuous integration support

### 📁 notebooks/
Jupyter notebooks for exploration.
notebooks/
├── init.py
├── Computer Vision UNet for Medical Segmentation.ipynb
├── exploration.ipynb # Data exploration
└── results_analysis.ipynb # Result analysis

text

**Key Features:**
- Interactive experimentation
- Result visualization
- Model prototyping
- Educational examples

### 📁 scripts/
Executable scripts for common tasks.
scripts/
├── train.py # Model training
├── evaluate.py # Model evaluation
├── demo.py # Demonstration
├── export_model.py # Model export
└── create_data.py # Data creation

text

**Key Features:**
- Command-line interface
- Production-ready scripts
- Easy automation
- Configuration support

### 📁 docs/
Project documentation.
docs/
├── API.md # API documentation
├── DEPLOYMENT.md # Deployment guide
├── STRUCTURE.md # This file
└── TROUBLESHOOTING.md # Troubleshooting guide

text

**Key Features:**
- Comprehensive documentation
- Usage examples
- Deployment guides
- Troubleshooting help

## File Descriptions

### Configuration Files

- `requirements.txt`: Python package dependencies
- `setup.py`: Traditional Python package setup
- `pyproject.toml`: Modern Python package configuration
- `Dockerfile`: Container image definition
- `docker-compose.yml`: Multi-service container setup

### Documentation

- `README.md`: Project overview and quick start
- `CONTRIBUTING.md`: Contribution guidelines
- `LICENSE`: Software license
- `.gitignore`: Git ignore patterns
- `.dockerignore`: Docker ignore patterns

## Development Workflow

### Adding New Features

1. **Update configuration** in `config/`
2. **Implement core logic** in appropriate module
3. **Add tests** in `tests/`
4. **Update documentation** in `docs/`
5. **Create examples** in `notebooks/`

### Code Organization Principles

- **Modularity**: Each module has a single responsibility
- **Testability**: All components are easily testable
- **Configurability**: Behavior controlled via configuration
- **Extensibility**: Easy to add new features and models

### Data Flow
Raw Data → Preprocessing → Model Training → Evaluation → Deployment
↓ ↓ ↓ ↓ ↓
data/ data/ training/ evaluation/ inference/

text

This structure supports both research experimentation and production deployment, making it suitable for academic research, industrial applications, and educational purposes.