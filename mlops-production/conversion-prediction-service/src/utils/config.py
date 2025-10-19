import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: str = "config/api_config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return {}
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def api_config(self):
        return self.get('api', {})
    
    @property
    def model_config(self):
        return self.get('model', {})
    
    @property
    def features_config(self):
        return self.get('features', {})

# Global config instance
config = Config()