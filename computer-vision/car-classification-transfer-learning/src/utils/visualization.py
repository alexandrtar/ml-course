import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
import pandas as pd

def plot_training_history(trainer_results, model_name):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(trainer_results['train_losses'], label='Train Loss')
    axes[0, 0].plot(trainer_results['val_losses'], label='Val Loss')
    axes[0, 0].set_title(f'{model_name} - Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score
    axes[0, 1].plot(trainer_results['val_f1_scores'], label='Val F1', color='green')
    axes[0, 1].set_title(f'{model_name} - F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate (если есть)
    if 'learning_rates' in trainer_results:
        axes[1, 0].plot(trainer_results['learning_rates'], label='LR', color='red')
        axes[1, 0].set_title(f'{model_name} - Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Best metrics comparison
    axes[1, 1].bar(['Best F1'], [trainer_results['best_metric']], color='orange')
    axes[1, 1].set_title(f'{model_name} - Best Metric')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_model_comparison(all_results):
    """Сравнение нескольких моделей"""
    models = list(all_results.keys())
    best_f1_scores = [all_results[m]['best_metric'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot сравнения F1 scores
    bars = axes[0].bar(models, best_f1_scores, color=['blue', 'green', 'red', 'orange'])
    axes[0].set_title('Model Comparison - Best F1 Scores')
    axes[0].set_ylabel('F1 Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Добавление значений на столбцы
    for bar, value in zip(bars, best_f1_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
    
    # Learning curves сравнение
    for model_name, results in all_results.items():
        axes[1].plot(results['val_f1_scores'], label=model_name)
    
    axes[1].set_title('Validation F1 Score Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Визуализация confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def plot_feature_embeddings(model, dataloader, device, class_names, n_samples=1000):
    """t-SNE визуализация эмбеддингов"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            if len(features) >= n_samples:
                break
                
            inputs = inputs.to(device)
            # Получаем эмбеддинги до классификатора
            if hasattr(model, 'features'):
                # Для CNN моделей
                embeddings = model.features(inputs)
                embeddings = torch.flatten(embeddings, 1)
            elif hasattr(model, 'model') and hasattr(model.model, 'avgpool'):
                # Для ResNet
                x = model.model.conv1(inputs)
                x = model.model.bn1(x)
                x = model.model.relu(x)
                x = model.model.maxpool(x)
                x = model.model.layer1(x)
                x = model.model.layer2(x)
                x = model.model.layer3(x)
                x = model.model.layer4(x)
                embeddings = model.model.avgpool(x)
                embeddings = torch.flatten(embeddings, 1)
            else:
                # Fallback - используем выход перед последним слоем
                embeddings = model(inputs)
            
            features.extend(embeddings.cpu().numpy())
            labels.extend(targets.numpy())
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features[:n_samples])
    labels = labels[:n_samples]
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c=labels, cmap='tab10', alpha=0.7)
    
    ax.set_title('t-SNE Visualization of Feature Embeddings')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    
    # Легенда
    legend_elements = scatter.legend_elements()[0]
    ax.legend(legend_elements, class_names, title="Classes")
    
    plt.tight_layout()
    return fig