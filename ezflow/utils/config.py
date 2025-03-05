"""
Configuration module for the ezflow framework.

This module contains the Config class for managing project settings.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class Config:
    """
    Base configuration class for the ezflow framework.
    
    This class provides a central repository for all configuration settings.
    """
    
    # Data paths
    DATA_PATHS = {
        "train": "data/raw/train.csv",
        "test": "data/raw/test.csv",
        "validation": "data/raw/validation.csv"
    }
    
    # Target column
    TARGET_COLUMN = "target"
    
    # Model parameters
    MODEL = {
        "type": "xgb",  # Model type (e.g., 'xgb', 'lgbm', 'sklearn')
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "random_state": 42
        }
    }
    
    # Training parameters
    TRAIN_PARAMS = {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True  # Whether to use stratified sampling for train-test split
    }
    
    # Feature engineering parameters
    FEATURE_ENGINEERING = {
        "create_interactions": False,  # Whether to create interaction features
        "create_ratios": False,  # Whether to create ratio features
        "scaling_method": "standard",  # Scaling method ('standard', 'minmax', None)
        "numeric_impute_strategy": "mean",  # Strategy for imputing missing numeric values
        "categorical_impute_strategy": "most_frequent",  # Strategy for imputing missing categorical values
        "categorical_encoding": "onehot"  # Encoding method for categorical variables ('onehot', 'label', 'target')
    }
    
    # Evaluation metrics
    METRICS = ["accuracy", "precision", "recall", "f1"]
    
    # Cross-validation parameters
    CV_PARAMS = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42
    }
    
    # Paths for saving models and results
    SAVE_MODEL_PATH = "models/model.pkl"
    RESULTS_PATH = "models/results.json"
    FEATURE_IMPORTANCE_PATH = "models/feature_importance.png"
    ROC_CURVE_PATH = "models/roc_curve.png"
    CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"
    REGRESSION_PLOT_PATH = "models/regression_plot.png"
    
    # Hyperparameter tuning parameters
    HYPERPARAMETER_TUNING = {
        "method": "optuna",  # Tuning method ('optuna', 'hyperopt')
        "n_trials": 100,  # Number of trials for hyperparameter search
        "timeout": 600,  # Timeout in seconds
        "param_space": {
            "n_estimators": {"low": 50, "high": 500, "step": 50},
            "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"low": 3, "high": 10, "step": 1}
        }
    }
    
    # Deployment parameters
    DEPLOYMENT = {
        "type": "api",  # Deployment type ('api', 'dashboard')
        "port": 8000,  # Port for API server
        "host": "127.0.0.1"  # Host for API server
    }
    
    # Logging parameters
    LOGGING = {
        "level": "INFO",
        "log_file": "logs/ezflow.log"
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the Config with optional dictionary.
        
        Args:
            config_dict (Optional[Dict[str, Any]]): Configuration dictionary to override defaults.
        """
        # Override defaults with provided config
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    # If the attribute is a dict, update it instead of replacing
                    if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                        getattr(self, key).update(value)
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary.
        """
        # Get all uppercase attributes (conventional for constants/settings)
        return {key: value for key, value in self.__class__.__dict__.items()
                if key.isupper() and not key.startswith('_')}
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update config with new values.
        
        Args:
            config_dict (Dict[str, Any]): Configuration dictionary with new values.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                # If the attribute is a dict, update it instead of replacing
                if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key (str): Configuration key.
            default (Any): Default value if key not found.
            
        Returns:
            Any: Configuration value.
        """
        # Check if key is a nested path (e.g., 'MODEL.params.n_estimators')
        if '.' in key:
            parts = key.split('.')
            value = self
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
        
        # Simple direct attribute
        return getattr(self, key, default)
    
    def validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List[str]: List of validation errors. Empty if valid.
        """
        errors = []
        
        # Validate data paths
        if not self.DATA_PATHS.get("train"):
            errors.append("Training data path not specified")
        
        # Validate target column
        if not self.TARGET_COLUMN:
            errors.append("Target column not specified")
        
        # Validate model
        if not self.MODEL.get("type"):
            errors.append("Model type not specified")
        
        # Validate metrics
        if not self.METRICS:
            errors.append("No evaluation metrics specified")
        
        # Add more validation as needed
        
        return errors
    
    def __str__(self) -> str:
        """String representation of the config."""
        return str(self.to_dict())
    
    def __repr__(self) -> str:
        """Detailed string representation of the config."""
        return f"Config({self.to_dict()})"


def load_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Create a Config instance from a dictionary.
    
    Args:
        config_dict (Dict[str, Any]): Configuration dictionary.
        
    Returns:
        Config: Config instance.
    """
    return Config(config_dict)


def load_config_from_file(config_path: str) -> Config:
    """
    Load configuration from a file.
    
    Args:
        config_path (str): Path to configuration file (JSON or YAML).
        
    Returns:
        Config: Config instance.
    """
    import json
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    return Config(config_dict)


def save_config_to_file(config: Config, config_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config (Config): Config instance.
        config_path (str): Path to save configuration file.
    """
    import json
    import yaml
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    config_dict = config.to_dict()
    
    if config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    elif config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    logger.info(f"Configuration saved to {config_path}") 