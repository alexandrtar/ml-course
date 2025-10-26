import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import mlflow
import wandb

class TrainingLogger:
    """
    Comprehensive logger for training experiments
    Supports MLflow, Weights & Biases, and local logging
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        use_mlflow: bool = True,
        use_wandb: bool = False,
        log_dir: str = "logs"
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup local logging
        self._setup_local_logging()
        
        # Setup MLflow
        if use_mlflow:
            self._setup_mlflow()
        
        # Setup WandB
        if use_wandb:
            self._setup_wandb()
    
    def _setup_local_logging(self):
        """Setup local file and console logging"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = os.path.join(self.log_dir, f"{self.run_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            self.logger.info(f"✅ MLflow tracking started: {self.run_name}")
        except Exception as e:
            self.logger.warning(f"⚠️ MLflow setup failed: {e}")
            self.use_mlflow = False
    
    def _setup_wandb(self):
        """Setup Weights & Biases tracking"""
        try:
            wandb.init(
                project=self.experiment_name,
                name=self.run_name,
                config={}  # Will be updated later
            )
            self.logger.info(f"✅ WandB tracking started: {self.run_name}")
        except Exception as e:
            self.logger.warning(f"⚠️ WandB setup failed: {e}")
            self.use_wandb = False
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        self.logger.info(f"Parameters: {params}")
        
        if self.use_mlflow:
            try:
                mlflow.log_params(params)
            except Exception as e:
                self.logger.warning(f"MLflow params logging failed: {e}")
        
        if self.use_wandb:
            try:
                wandb.config.update(params)
            except Exception as e:
                self.logger.warning(f"WandB params logging failed: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Metrics (step {step}): {metrics_str}")
        
        if self.use_mlflow:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"MLflow metrics logging failed: {e}")
        
        if self.use_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"WandB metrics logging failed: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact (file)"""
        if self.use_mlflow:
            try:
                mlflow.log_artifact(local_path, artifact_path)
            except Exception as e:
                self.logger.warning(f"MLflow artifact logging failed: {e}")
        
        if self.use_wandb:
            try:
                wandb.save(local_path)
            except Exception as e:
                self.logger.warning(f"WandB artifact logging failed: {e}")
    
    def log_model(self, model, model_name: str = "model"):
        """Log model"""
        if self.use_mlflow:
            try:
                mlflow.pytorch.log_model(model, model_name)
            except Exception as e:
                self.logger.warning(f"MLflow model logging failed: {e}")
    
    def log_image(self, image, name: str, step: Optional[int] = None):
        """Log image"""
        if self.use_wandb:
            try:
                wandb.log({name: wandb.Image(image)}, step=step)
            except Exception as e:
                self.logger.warning(f"WandB image logging failed: {e}")
    
    def finish(self):
        """Finish logging and clean up"""
        if self.use_mlflow:
            try:
                mlflow.end_run()
            except Exception as e:
                self.logger.warning(f"MLflow finish failed: {e}")
        
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"WandB finish failed: {e}")


def setup_logger(experiment_name: str, **kwargs) -> TrainingLogger:
    """Convenience function to setup logger"""
    return TrainingLogger(experiment_name, **kwargs)