import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsTracker:
    """Трекер метрик во время обучения"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.predictions = []
        self.targets = []
        self.epoch_metrics = {}
    
    def update(self, loss, predictions, targets):
        self.losses.append(loss)
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute_epoch_metrics(self):
        """Вычисление метрик для эпохи"""
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(targets, predictions),
            'f1_weighted': f1_score(targets, predictions, average='weighted'),
            'f1_macro': f1_score(targets, predictions, average='macro'),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted')
        }
        
        self.epoch_metrics = metrics
        self.reset()  # Сброс для следующей эпохи
        
        return metrics

def calculate_comprehensive_metrics(model, dataloader, device, criterion=None):
    """Расширенный расчет метрик для модели"""
    model.eval()
    tracker = MetricsTracker()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # ВАЖНО: Перенос данных на устройство
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if criterion and targets[0] != -1:  # Если есть настоящие метки
                targets = targets.to(device)
                loss = criterion(outputs, targets)
                
                _, predictions = torch.max(outputs, 1)
                tracker.update(loss.item(), predictions, targets)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else:
                # Для тестовых данных без меток
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
    
    if len(all_targets) > 0:
        metrics = tracker.compute_epoch_metrics()
        
        # Дополнительные метрики
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        metrics['confusion_matrix'] = cm
        
        # Per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        metrics['class_accuracy'] = class_accuracy
        metrics['min_class_accuracy'] = np.min(class_accuracy)
        metrics['max_class_accuracy'] = np.max(class_accuracy)
        
        return metrics, all_predictions
    else:
        return None, all_predictions

def print_detailed_metrics(metrics, class_names=None):
    """Детальный вывод метрик"""
    print("\n" + "="*60)
    print("DETAILED MODEL METRICS")
    print("="*60)
    
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Min Class Accuracy: {metrics['min_class_accuracy']:.4f}")
    print(f"Max Class Accuracy: {metrics['max_class_accuracy']:.4f}")
    
    if class_names and 'class_accuracy' in metrics:
        print("\nPer-Class Accuracy:")
        for i, acc in enumerate(metrics['class_accuracy']):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"  {class_name}: {acc:.4f}")