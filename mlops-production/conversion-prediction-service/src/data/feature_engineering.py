import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for conversion prediction"""
    
    def __init__(self):
        self.conversion_events = {
            'sub_car_claim_click': 'car_claim',
            'sub_car_claim_submit_click': 'car_claim_submit', 
            'sub_open_dialog_click': 'open_dialog',
            'sub_custom_question_submit_click': 'custom_question',
            'sub_call_number_click': 'call_number',
            'sub_callback_submit_click': 'callback',
            'sub_submit_success': 'submit_success',
            'sub_car_request_submit_click': 'car_request'
        }
    
    def create_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variable from event actions"""
        logger.info("Creating target variable...")
        
        data['conversion_type'] = data['event_action'].map(self.conversion_events)
        data['conversion_type'] = data['conversion_type'].fillna('no_conversion')
        data['target'] = (data['conversion_type'] != 'no_conversion').astype(int)
        
        conversion_rate = data['target'].mean()
        logger.info(f"Conversion rate: {conversion_rate:.4f}")
        
        return data
    
    def create_utm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create UTM-related features"""
        logger.info("Creating UTM features...")
        
        # Fill missing values
        data['utm_source'] = data['utm_source'].fillna('unknown')
        data['utm_medium'] = data['utm_medium'].fillna('unknown')
        
        # UTM source classification
        def classify_utm_source(source):
            social_sources = [
                'QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
            ]
            
            source_str = str(source)
            if source_str in social_sources:
                return 'social'
            elif source_str == 'unknown':
                return 'unknown'
            elif source_str == '(direct)':
                return 'direct'
            elif 'organic' in source_str.lower():
                return 'organic'
            else:
                return 'other'
        
        data['utm_source_type'] = data['utm_source'].apply(classify_utm_source)
        
        return data
    
    def create_device_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create device-related features"""
        logger.info("Creating device features...")
        
        # Fill missing values
        device_columns = ['device_brand', 'device_os', 'device_screen_resolution', 'device_browser']
        for col in device_columns:
            data[col] = data[col].fillna('unknown')
        
        # OS classification
        def classify_os(os):
            os_str = str(os)
            mobile_os = ['Android', 'iOS']
            desktop_os = ['Windows', 'Macintosh', 'Linux', 'Chrome OS']
            
            if os_str in mobile_os:
                return 'mobile'
            elif os_str in desktop_os:
                return 'desktop'
            elif os_str == 'unknown':
                return 'unknown'
            else:
                return 'other'
        
        data['os_type'] = data['device_os'].apply(classify_os)
        
        return data
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from visit date/time"""
        logger.info("Creating temporal features...")
        
        if 'visit_date' in data.columns:
            data['visit_date'] = pd.to_datetime(data['visit_date'])
            data['day_of_week'] = data['visit_date'].dt.dayofweek
            data['month'] = data['visit_date'].dt.month
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        if 'visit_time' in data.columns:
            try:
                data['hour_of_day'] = pd.to_datetime(
                    data['visit_time'], format='%H:%M:%S', errors='coerce'
                ).dt.hour
                data['hour_of_day'] = data['hour_of_day'].fillna(12)
            except:
                data['hour_of_day'] = 12
        
        # Cyclical features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
        
        # Time of day
        def classify_time_of_day(hour):
            if 5 <= hour <= 11:
                return 'morning'
            elif 12 <= hour <= 16:
                return 'afternoon'
            elif 17 <= hour <= 21:
                return 'evening'
            else:
                return 'night'
        
        data['time_of_day'] = data['hour_of_day'].apply(classify_time_of_day)
        data['is_peak_hours'] = ((data['hour_of_day'] >= 9) & (data['hour_of_day'] <= 18)).astype(int)
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering pipeline...")
        
        data = self.create_target_variable(data)
        data = self.create_utm_features(data)
        data = self.create_device_features(data)
        data = self.create_temporal_features(data)
        
        logger.info(f"Feature engineering completed. Final shape: {data.shape}")
        
        return data