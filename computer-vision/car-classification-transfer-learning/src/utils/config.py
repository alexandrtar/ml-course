import yaml
import torch
import os


def load_config(config_path=None):
    """Загрузка конфигурации из YAML файла"""
    if config_path is None:
        # Автоматический поиск config файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        config_path = os.path.join(project_root, 'configs', 'experiment_config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Автоматическое определение устройства
    if config['experiment']['device'] == 'auto':
        config['experiment']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    return config


def setup_experiment_env(config):
    """Настройка окружения для эксперимента"""
    # Создание директорий
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    os.makedirs(os.path.join(project_root, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)

    # Настройка воспроизводимости
    if config['experiment'].get('seed', None):
        torch.manual_seed(config['experiment']['seed'])
        if config['experiment']['device'] == 'cuda':
            torch.cuda.manual_seed_all(config['experiment']['seed'])

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Device: {config['experiment']['device']}")
    print(f"Number of epochs: {config['experiment']['num_epochs']}")
    print(f"Batch size: {config['experiment']['batch_size']}")


def save_config(config, save_path):
    """Сохранение конфигурации"""
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def update_config(config, updates):
    """Обновление конфигурации"""
    for key, value in updates.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current[k]
        current[keys[-1]] = value
    return config