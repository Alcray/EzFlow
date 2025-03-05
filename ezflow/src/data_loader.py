"""
Data loading, cleaning and preprocessing module.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union, List

logger = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader class responsible for loading, cleaning, and preprocessing data.
    
    This class handles loading data from various sources (CSV, Excel, etc.),
    performing initial cleaning operations, and preparing data for feature engineering.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            config (Dict): Configuration dictionary with data paths and parameters.
        """
        self.config = config
        self.data_paths = config.get('DATA_PATHS', {})
        logger.info(f"DataLoader initialized with paths: {self.data_paths}")
    
    def load_data(self, mode: str) -> pd.DataFrame:
        """
        Load data based on mode (train, test, val).
        
        Args:
            mode (str): Data mode to load ('train', 'test', 'val').
            
        Returns:
            pd.DataFrame: Loaded data as DataFrame.
            
        Raises:
            FileNotFoundError: If the specified data file doesn't exist.
            ValueError: If the mode is not valid.
        """
        if mode not in self.data_paths:
            raise ValueError(f"Invalid mode: {mode}. Available modes: {list(self.data_paths.keys())}")
        
        file_path = self.data_paths[mode]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading {mode} data from {file_path}")
        
        # Determine file type and load accordingly
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        logger.info(f"Loaded {mode} data with shape: {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame, drop_na: bool = False) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Args:
            df (pd.DataFrame): Input DataFrame to clean.
            drop_na (bool): Whether to drop rows with missing values.
            
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logger.info("Starting data cleaning...")
        
        # Make a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        # Check for and remove duplicates
        n_duplicates = df_cleaned.duplicated().sum()
        if n_duplicates > 0:
            logger.info(f"Removing {n_duplicates} duplicate rows")
            df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
        
        # Handle missing values
        missing_values = df_cleaned.isna().sum()
        missing_cols = missing_values[missing_values > 0]
        if not missing_cols.empty:
            logger.info(f"Columns with missing values:\n{missing_cols}")
            
            if drop_na:
                n_before = len(df_cleaned)
                df_cleaned = df_cleaned.dropna()
                logger.info(f"Dropped {n_before - len(df_cleaned)} rows with missing values")
        
        # Basic type conversion (optional)
        # This can be expanded based on specific needs
        
        logger.info(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
        return df_cleaned
    
    def split_features_target(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Name of the target column.
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series.
            
        Raises:
            ValueError: If target column is not in the DataFrame.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Split data into features ({X.shape}) and target ({y.shape})")
        return X, y
    
    def train_test_split(
        self, df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Name of the target column.
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            Tuple containing (X_train, X_test, y_train, y_test).
        """
        from sklearn.model_selection import train_test_split as sk_split
        
        X, y = self.split_features_target(df, target_col)
        X_train, X_test, y_train, y_test = sk_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(
            f"Train-test split: X_train {X_train.shape}, X_test {X_test.shape}, "
            f"y_train {y_train.shape}, y_test {y_test.shape}"
        )
        return X_train, X_test, y_train, y_test
    
    def merge_data(
        self, df1: pd.DataFrame, df2: pd.DataFrame, on: Union[str, List[str]], how: str = 'inner'
    ) -> pd.DataFrame:
        """
        Merge two DataFrames.
        
        Args:
            df1 (pd.DataFrame): First DataFrame.
            df2 (pd.DataFrame): Second DataFrame.
            on (Union[str, List[str]]): Column(s) to merge on.
            how (str): Type of merge ('inner', 'outer', 'left', 'right').
            
        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        logger.info(f"Merging DataFrames of shapes {df1.shape} and {df2.shape} on {on} (how={how})")
        merged_df = pd.merge(df1, df2, on=on, how=how)
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> None:
        """
        Save processed data to disk.
        
        Args:
            df (pd.DataFrame): DataFrame to save.
            filename (str): Name of the file to save to.
            format (str): File format ('csv', 'parquet', etc.).
        """
        # Create processed directory if it doesn't exist
        processed_dir = os.path.dirname(filename)
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        logger.info(f"Saving processed data to {filename}")
        
        if format.lower() == 'csv':
            df.to_csv(filename, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(filename, index=False)
        elif format.lower() == 'pickle':
            df.to_pickle(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved successfully to {filename}") 