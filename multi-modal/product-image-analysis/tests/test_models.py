import pytest
import pandas as pd
import numpy as np
from src.models.multimodal_model import MultiModalModel

class TestMultiModalModel:
    
    @pytest.fixture
    def sample_config(self):
        return {
            'model': {
                'params': {
                    'n_estimators': 10,
                    'random_state': 42
                }
            },
            'training': {
                'random_state': 42
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        return X, y
    
    def test_model_initialization(self, sample_config):
        """Test model initialization"""
        model = MultiModalModel(sample_config)
        
        assert model.config == sample_config
        assert model.model is None
        assert not model.is_trained
    
    def test_model_training(self, sample_config, sample_data):
        """Test model training"""
        X, y = sample_data
        model = MultiModalModel(sample_config)
        
        model.train(X, y)
        
        assert model.is_trained
        assert model.model is not None
    
    def test_model_prediction(self, sample_config, sample_data):
        """Test model prediction"""
        X, y = sample_data
        model = MultiModalModel(sample_config)
        
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})