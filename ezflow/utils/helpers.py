"""
Helper functions for the ezflow framework.
"""

import os
import logging
import time
import json
import yaml
from typing import Dict, Any, List, Optional, Union, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

logger = logging.getLogger(__name__)

def setup_logging(
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file (Optional[str]): Path to log file. If None, only console logging is enabled.
        console_level (int): Logging level for console output.
        file_level (int): Logging level for file output.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    
    return logger

def timer(func):
    """
    Decorator to measure the execution time of a function.
    
    Args:
        func (Callable): Function to decorate.
        
    Returns:
        Callable: Decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path (str): Path to the configuration file (JSON or YAML).
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
        
    Raises:
        ValueError: If the file format is not supported.
    """
    logger.info(f"Loading configuration from {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    logger.info(f"Configuration loaded successfully from {config_path}")
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary.
        config_path (str): Path to save the configuration file.
    """
    logger.info(f"Saving configuration to {config_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    if config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    elif config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    logger.info(f"Configuration saved successfully to {config_path}")

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        Dict[str, Any]: Summary dictionary.
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.apply(lambda x: str(x)).to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict(),
        'numeric_columns': df.select_dtypes(include=np.number).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
    }
    
    # Add basic statistics for numeric columns
    if summary['numeric_columns']:
        summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()
    
    # Add basic statistics for categorical columns
    if summary['categorical_columns']:
        summary['categorical_stats'] = {
            col: {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
            for col in summary['categorical_columns']
        }
    
    return summary

def plot_missing_values(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        save_path (Optional[str]): Path to save the plot.
    """
    # Calculate missing values
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    
    # Get columns with missing values
    cols_with_missing = missing[missing > 0]
    
    if cols_with_missing.empty:
        logger.info("No missing values to plot")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    
    # Plot bars
    ax.bar(
        range(len(cols_with_missing)), 
        cols_with_missing,
        color='crimson',
        alpha=0.7
    )
    
    # Add percentage labels
    for i, val in enumerate(cols_with_missing):
        pct = missing_pct[cols_with_missing.index[i]]
        ax.text(
            i, val + 0.5, f"{pct}%", 
            ha='center', va='bottom',
            fontweight='bold'
        )
    
    # Set labels and title
    plt.xticks(range(len(cols_with_missing)), cols_with_missing.index, rotation=90)
    plt.ylabel('Number of Missing Values')
    plt.title('Missing Values by Column')
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save plot
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Missing values plot saved to {save_path}")
    else:
        plt.show()

def memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate the memory usage of a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        Dict[str, Any]: Memory usage information.
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    # Get memory usage by data type
    memory_by_dtype = df.dtypes.map(lambda x: str(x)).value_counts().to_dict()
    memory_by_dtype = {
        dtype: df.select_dtypes(include=dtype).memory_usage(deep=True).sum()
        for dtype in memory_by_dtype
    }
    
    return {
        'total_memory_bytes': total_memory,
        'total_memory_mb': total_memory / (1024 * 1024),
        'memory_by_column': memory_usage.to_dict(),
        'memory_by_dtype': memory_by_dtype
    }

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize the memory usage of a DataFrame by converting data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Optimized DataFrame.
    """
    df_optimized = df.copy()
    
    # Memory usage before optimization
    mem_before = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Memory usage before optimization: {mem_before:.2f} MB")
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=np.number).columns:
        # Convert integer columns to the smallest possible integer type
        if pd.api.types.is_integer_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            
            if min_val >= 0:  # Unsigned
                if max_val < np.iinfo(np.uint8).max:
                    df_optimized[col] = df[col].astype(np.uint8)
                elif max_val < np.iinfo(np.uint16).max:
                    df_optimized[col] = df[col].astype(np.uint16)
                elif max_val < np.iinfo(np.uint32).max:
                    df_optimized[col] = df[col].astype(np.uint32)
            else:  # Signed
                if min_val > np.iinfo(np.int8).min and max_val < np.iinfo(np.int8).max:
                    df_optimized[col] = df[col].astype(np.int8)
                elif min_val > np.iinfo(np.int16).min and max_val < np.iinfo(np.int16).max:
                    df_optimized[col] = df[col].astype(np.int16)
                elif min_val > np.iinfo(np.int32).min and max_val < np.iinfo(np.int32).max:
                    df_optimized[col] = df[col].astype(np.int32)
        
        # Convert float columns to float32 if possible
        elif pd.api.types.is_float_dtype(df[col]):
            df_optimized[col] = df[col].astype(np.float32)
    
    # Optimize categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        # Convert string columns to categorical if they have few unique values
        num_unique = df[col].nunique()
        num_total = len(df)
        
        if num_unique / num_total < 0.5:  # If less than 50% of values are unique
            df_optimized[col] = df[col].astype('category')
    
    # Memory usage after optimization
    mem_after = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    savings = (1 - mem_after / mem_before) * 100
    
    logger.info(f"Memory usage after optimization: {mem_after:.2f} MB")
    logger.info(f"Memory savings: {savings:.2f}%")
    
    return df_optimized

def create_datetime_features(df: pd.DataFrame, datetime_col: str, drop_original: bool = False) -> pd.DataFrame:
    """
    Create datetime features from a datetime column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        datetime_col (str): Name of the datetime column.
        drop_original (bool): Whether to drop the original datetime column.
        
    Returns:
        pd.DataFrame: DataFrame with datetime features.
    """
    df_new = df.copy()
    
    # Check if the column exists
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame")
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(df[datetime_col]):
        df_new[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Extract datetime components
    df_new[f"{datetime_col}_year"] = df_new[datetime_col].dt.year
    df_new[f"{datetime_col}_month"] = df_new[datetime_col].dt.month
    df_new[f"{datetime_col}_day"] = df_new[datetime_col].dt.day
    df_new[f"{datetime_col}_hour"] = df_new[datetime_col].dt.hour
    df_new[f"{datetime_col}_minute"] = df_new[datetime_col].dt.minute
    df_new[f"{datetime_col}_second"] = df_new[datetime_col].dt.second
    df_new[f"{datetime_col}_dayofweek"] = df_new[datetime_col].dt.dayofweek
    df_new[f"{datetime_col}_dayofyear"] = df_new[datetime_col].dt.dayofyear
    df_new[f"{datetime_col}_quarter"] = df_new[datetime_col].dt.quarter
    df_new[f"{datetime_col}_is_weekend"] = df_new[f"{datetime_col}_dayofweek"].isin([5, 6]).astype(int)
    df_new[f"{datetime_col}_is_month_start"] = df_new[datetime_col].dt.is_month_start.astype(int)
    df_new[f"{datetime_col}_is_month_end"] = df_new[datetime_col].dt.is_month_end.astype(int)
    
    # Drop original column if requested
    if drop_original:
        df_new = df_new.drop(columns=[datetime_col])
    
    return df_new

def create_cyclical_features(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """
    Create cyclical features for a cyclic variable (e.g., hour of day, month of year).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Name of the column to create cyclical features for.
        period (int): Cycle period (e.g., 24 for hour of day, 12 for month of year).
        
    Returns:
        pd.DataFrame: DataFrame with cyclical features.
    """
    df_new = df.copy()
    
    # Check if the column exists
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Create cyclical features using sine and cosine transformations
    df_new[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df_new[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    
    return df_new 