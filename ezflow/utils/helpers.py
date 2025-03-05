import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import logging
from functools import wraps
import hashlib
import yaml
from datetime import datetime

logger = logging.getLogger("ezflow.utils")

def timed(func):
    """
    Decorator for timing function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Function {func.__name__} executed in {end - start:.2f} seconds")
        return result
    return wrapper


def generate_hash(obj: Any) -> str:
    """
    Generate a hash from an object.
    
    Args:
        obj: Object to hash
        
    Returns:
        String hash of the object
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        data = obj.to_json()
    elif isinstance(obj, np.ndarray):
        data = obj.tobytes()
    elif isinstance(obj, (dict, list, tuple)):
        data = json.dumps(obj, sort_keys=True).encode('utf-8')
    else:
        data = str(obj).encode('utf-8')
    
    return hashlib.md5(data).hexdigest()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif ext == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'w') as f:
        if ext in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif ext == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")


def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to save the object
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found at {file_path}")
    
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    
    return obj


def setup_logger(log_dir: str = 'logs', log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('ezflow')
    logger.setLevel(log_level)
    
    # Create handlers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'ezflow_{timestamp}.log'))
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten a dictionary with nested keys.
    
    Args:
        d: Flattened dictionary
        sep: Separator for keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        
        # Traverse the parts and create nested dictionaries
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (overrides values in dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def get_class_weights(y: np.ndarray, strategy: str = 'balanced') -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        strategy: Strategy for computing weights ('balanced' or 'balanced_sqrt')
        
    Returns:
        Dictionary mapping class indices to weights
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)
    
    # Compute weights based on strategy
    if strategy == 'balanced':
        weights = n_samples / (n_classes * class_counts)
    elif strategy == 'balanced_sqrt':
        weights = np.sqrt(n_samples / (n_classes * class_counts))
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")
    
    # Create dictionary mapping class to weight
    class_weights = {cls: weight for cls, weight in zip(unique_classes, weights)}
    
    return class_weights


def get_timestamp() -> str:
    """
    Get a formatted timestamp string.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_submission_file(predictions: Union[np.ndarray, List], ids: Union[np.ndarray, List], 
                          submission_path: str, columns: Optional[List[str]] = None) -> None:
    """
    Create a submission file for competitions.
    
    Args:
        predictions: Predicted values
        ids: Sample IDs
        submission_path: Path to save the submission file
        columns: Column names for the DataFrame
    """
    if columns is None:
        # Create default column names
        if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
            columns = ['id', 'prediction']
        else:
            # For multiclass predictions
            n_classes = predictions.shape[1]
            columns = ['id'] + [f'class_{i}' for i in range(n_classes)]
    
    # Convert predictions to proper format
    if predictions.ndim == 1:
        data = np.column_stack((ids, predictions))
    else:
        data = np.column_stack((ids, predictions))
    
    # Create and save DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    
    # Save file
    df.to_csv(submission_path, index=False)
    logger.info(f"Submission file saved to {submission_path}")

    
def load_dataset_from_config(config_path: str) -> 'BaseDataset':
    """
    Load a dataset based on configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Instantiated dataset
    """
    from importlib import import_module
    
    config = load_config(config_path)
    
    if 'dataset' not in config:
        raise ValueError("Configuration must contain a 'dataset' section")
    
    dataset_config = config['dataset']
    
    if 'class' not in dataset_config:
        raise ValueError("Dataset configuration must specify a 'class'")
    
    # Get the dataset class
    class_path = dataset_config['class']
    module_path, class_name = class_path.rsplit('.', 1)
    
    try:
        module = import_module(module_path)
        dataset_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import dataset class '{class_path}': {e}")
    
    # Extract parameters for dataset initialization
    params = {k: v for k, v in dataset_config.items() if k != 'class'}
    
    # Create and return dataset instance
    return dataset_class(**params) 