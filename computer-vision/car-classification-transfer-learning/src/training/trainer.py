import torch
import torch.nn as nn
from tqdm import tqdm
import time
import mlflow
import os
from .metrics import MetricsTracker, calculate_comprehensive_metrics


class ModelTrainer:
    """Универсальный тренер для моделей"""

    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or self._setup_device()

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []

    def _setup_device(self):
        """Настройка устройства"""
        if self.config['experiment']['device'] == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(self.config['experiment']['device'])

    def _setup_optimizer(self):
        """Настройка оптимизатора"""
        lr = self.config['training']['base_lr']

        if hasattr(self.model, 'get_optimizer'):
            # Для AdvancedResNetWrapper
            return self.model.get_optimizer(
                base_lr=lr,
                optimizer_type=self.config['training']['optimizer']
            )
        else:
            # Для обычных моделей
            if self.config['training']['optimizer'] == 'adamw':
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=self.config['training'].get('weight_decay', 0.01)
                )
            else:
                return torch.optim.Adam(
                    self.model.parameters(),
                    lr=lr
                )

    def _setup_scheduler(self):
        """Настройка scheduler"""
        if self.config['training'].get('scheduler') == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['experiment']['num_epochs']
            )
        else:
            return None

    def _ensure_dir(self, path):
        """Создание директории если не существует"""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def train_epoch(self, train_loader):
        """Обучение на одной эпохе"""
        self.model.train()
        tracker = MetricsTracker()

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Обновление метрик
            with torch.no_grad():
                _, predictions = torch.max(outputs, 1)
                tracker.update(loss.item(), predictions, targets)

            # Обновление progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

        return tracker.compute_epoch_metrics()

    def validate_epoch(self, val_loader):
        """Валидация на одной эпохе"""
        metrics, _ = calculate_comprehensive_metrics(
            self.model, val_loader, self.device, self.criterion
        )
        return metrics

    def train(self, train_loader, val_loader, save_path=None):
        """Полный цикл обучения"""
        num_epochs = self.config['experiment']['num_epochs']

        # Создание безопасного пути для сохранения
        if save_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            save_path = os.path.join(project_root, "checkpoints", f"best_model.pth")

        # Гарантируем что директория существует
        self._ensure_dir(save_path)

        print(f"Starting training on {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Model will be saved to: {save_path}")

        # MLflow logging
        if self.config['logging'].get('use_mlflow', False):
            try:
                mlflow.start_run(run_name=self.config['experiment']['name'])
            except:
                print("MLflow not available, continuing without logging")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Обучение
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])

            # Валидация
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_f1_scores.append(val_metrics['f1_weighted'])

            # Обновление scheduler
            if self.scheduler:
                self.scheduler.step()

            # Вывод метрик
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val F1: {val_metrics['f1_weighted']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Логирование в MLflow
            if self.config['logging'].get('use_mlflow', False):
                try:
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'val_f1': val_metrics['f1_weighted'],
                        'val_accuracy': val_metrics['accuracy'],
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=epoch)
                except:
                    pass

            # Сохранение лучшей модели
            if val_metrics['f1_weighted'] > self.best_metric:
                self.best_metric = val_metrics['f1_weighted']
                try:
                    # Сохраняем с безопасным путем
                    torch.save(self.model.state_dict(), save_path)
                    print(f"New best model saved! F1: {self.best_metric:.4f}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                    # Альтернативный путь сохранения
                    alt_path = "best_model.pth"
                    torch.save(self.model.state_dict(), alt_path)
                    print(f"Model saved to alternative path: {alt_path}")

        if self.config['logging'].get('use_mlflow', False):
            try:
                mlflow.end_run()
            except:
                pass

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'best_metric': self.best_metric,
            'best_model_path': save_path
        }