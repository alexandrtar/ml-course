import logging
import sys
from .config import config

def setup_logger(name: str = "conversion_api"):
    """Setup application logger"""
    
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
    
    log_level = getattr(logging, config.get('logging.level', 'INFO'))
    log_format = config.get('logging.format', 
                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger