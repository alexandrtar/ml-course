import pytest
import pandas as pd
import numpy as np
import os
from src.data.preprocessing import DataPreprocessor

class TestDataPreprocessing:
    
    @pytest.fixture
    def sample_config(self):
        return {
            'data': {
                'raw_path': 'data/raw',
                'metadata_file': 'styles.csv'
            },
            'features': {
                'categorical_cols': ['gender', 'articleType']
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'gender': ['Men', 'Women', 'Men', 'Women', 'Men'],
            'articleType': ['Shirt', 'Dress', 'Shirt', 'Dress', 'Pants'],
            'baseColour': ['Black', 'White', 'Black', 'Red', 'Blue'],
            'season': ['Summer', 'Winter', 'Summer', 'Winter', 'Summer'],
            'year': [2020, 2021, 2020, 2021, 2020],
            'usage': ['Casual', 'Formal', 'Casual', 'Formal', 'Casual'],
            'productDisplayName': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        })
    
    def test_target_creation(self, sample_config, sample_data):
        """Test target variable creation"""
        preprocessor = DataPreprocessor(sample_config)
        preprocessor.data = sample_data
        
        df_with_target = preprocessor._create_target(sample_data)
        
        assert 'target' in df_with_target.columns
        assert df_with_target['target'].dtype == int
        assert set(df_with_target['target'].unique()).issubset({0, 1})
    
    def test_data_cleaning(self, sample_config, sample_data):
        """Test data cleaning operations"""
        preprocessor = DataPreprocessor(sample_config)
        
        # Add some missing values
        sample_data.loc[0, 'articleType'] = None
        sample_data.loc[1, 'baseColour'] = None
        
        cleaned_data = preprocessor._clean_data(sample_data)
        
        # Check that rows with missing values are removed
        assert cleaned_data.isnull().sum().sum() == 0
        assert len(cleaned_data) < len(sample_data)