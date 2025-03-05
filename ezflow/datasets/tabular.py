import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from ..core.dataset import BaseDataset


class TabularDataset(BaseDataset):
    """
    Dataset implementation for tabular data stored in CSV or Excel files.
    
    This class provides a convenient implementation for the common case of
    tabular data in machine learning competitions.
    """
    
    def __init__(
        self, 
        data_dir: str,
        train_file: str,
        test_file: str,
        id_column: Optional[str] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
        val_file: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the tabular dataset.
        
        Args:
            data_dir: Directory containing the data files
            train_file: Name of the training data file
            test_file: Name of the test data file
            id_column: Name of the ID column
            target_column: Name of the target column
            feature_columns: List of feature column names. If None, all columns except 
                            id_column and target_column will be used.
            categorical_columns: List of categorical column names
            date_columns: List of date column names to parse
            val_file: Optional name of validation data file
            **kwargs: Additional keyword arguments
        """
        super().__init__(data_dir, **kwargs)
        
        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file
        
        self.id_column = id_column
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns or []
        self.date_columns = date_columns or []
        
        self.is_regression = kwargs.get('is_regression', False)
        self.drop_na = kwargs.get('drop_na', False)
        self.fill_na = kwargs.get('fill_na', None)
        
        # Auto-infer file types
        self.file_type = self._get_file_type(train_file)
        
        # Keep track of column statistics for normalization
        self.column_stats = {}
        
        # Flag to track if preprocessing has been applied
        self.is_preprocessed = False
    
    def _get_file_type(self, filename: str) -> str:
        """Determine file type from extension."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.csv', '.txt']:
            return 'csv'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        elif ext in ['.parquet', '.pq']:
            return 'parquet'
        elif ext == '.feather':
            return 'feather'
        else:
            self.logger.warning(f"Unknown file extension {ext}, defaulting to CSV")
            return 'csv'
    
    def _read_file(self, file_path: str) -> pd.DataFrame:
        """Read a data file based on its type."""
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if self.file_type == 'csv':
                df = pd.read_csv(file_path)
            elif self.file_type == 'excel':
                df = pd.read_excel(file_path)
            elif self.file_type == 'parquet':
                df = pd.read_parquet(file_path)
            elif self.file_type == 'feather':
                df = pd.read_feather(file_path)
            else:
                self.logger.warning(f"Unknown file type {self.file_type}, attempting CSV")
                df = pd.read_csv(file_path)
                
            return df
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def load_data(self) -> None:
        """
        Load the tabular data files.
        """
        # Load training data
        train_path = os.path.join(self.data_dir, self.train_file)
        self.train_data = self._read_file(train_path)
        self.logger.info(f"Loaded training data: {len(self.train_data)} rows, {len(self.train_data.columns)} columns")
        
        # Load test data
        test_path = os.path.join(self.data_dir, self.test_file)
        self.test_data = self._read_file(test_path)
        self.logger.info(f"Loaded test data: {len(self.test_data)} rows, {len(self.test_data.columns)} columns")
        
        # Load validation data if available
        if self.val_file:
            val_path = os.path.join(self.data_dir, self.val_file)
            self.val_data = self._read_file(val_path)
            self.logger.info(f"Loaded validation data: {len(self.val_data)} rows, {len(self.val_data.columns)} columns")
        
        # If feature columns not specified, use all columns except ID and target
        if self.feature_columns is None:
            excluded_cols = [col for col in [self.id_column, self.target_column] if col is not None]
            self.feature_columns = [col for col in self.train_data.columns if col not in excluded_cols]
            self.logger.info(f"Auto-detected {len(self.feature_columns)} feature columns")
        
        # Store metadata about the dataset
        self.meta_info = {
            'train_rows': len(self.train_data),
            'test_rows': len(self.test_data),
            'val_rows': len(self.val_data) if self.val_data is not None else 0,
            'feature_count': len(self.feature_columns),
            'categorical_count': len(self.categorical_columns),
            'has_missing_values': self.train_data[self.feature_columns].isna().any().any()
        }
    
    def preprocess(self, normalize: bool = True) -> None:
        """
        Preprocess the dataset.
        
        Args:
            normalize: Whether to normalize numeric features
        """
        if self.is_preprocessed:
            self.logger.info("Data already preprocessed")
            return
        
        # Handle missing values
        if self.drop_na:
            self.logger.info("Dropping rows with missing values")
            self.train_data = self.train_data.dropna(subset=self.feature_columns)
            if self.val_data is not None:
                self.val_data = self.val_data.dropna(subset=self.feature_columns)
        elif self.fill_na is not None:
            self.logger.info(f"Filling missing values with {self.fill_na}")
            self.train_data[self.feature_columns] = self.train_data[self.feature_columns].fillna(self.fill_na)
            self.test_data[self.feature_columns] = self.test_data[self.feature_columns].fillna(self.fill_na)
            if self.val_data is not None:
                self.val_data[self.feature_columns] = self.val_data[self.feature_columns].fillna(self.fill_na)
        
        # Process date columns
        for col in self.date_columns:
            if col in self.train_data.columns:
                self.logger.info(f"Converting {col} to datetime")
                self.train_data[col] = pd.to_datetime(self.train_data[col])
                self.test_data[col] = pd.to_datetime(self.test_data[col])
                if self.val_data is not None:
                    self.val_data[col] = pd.to_datetime(self.val_data[col])
                
                # Extract common date features
                for data in [self.train_data, self.test_data]:
                    data[f"{col}_year"] = data[col].dt.year
                    data[f"{col}_month"] = data[col].dt.month
                    data[f"{col}_day"] = data[col].dt.day
                    data[f"{col}_dayofweek"] = data[col].dt.dayofweek
                
                if self.val_data is not None:
                    self.val_data[f"{col}_year"] = self.val_data[col].dt.year
                    self.val_data[f"{col}_month"] = self.val_data[col].dt.month
                    self.val_data[f"{col}_day"] = self.val_data[col].dt.day
                    self.val_data[f"{col}_dayofweek"] = self.val_data[col].dt.dayofweek
                
                # Add these new columns to feature list
                self.feature_columns.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])
                
                # Remove original date column from features
                if col in self.feature_columns:
                    self.feature_columns.remove(col)
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in self.train_data.columns and col in self.feature_columns:
                self.logger.info(f"One-hot encoding {col}")
                
                # Get all unique categories from both train and test
                all_categories = set(self.train_data[col].unique())
                all_categories.update(self.test_data[col].unique())
                if self.val_data is not None:
                    all_categories.update(self.val_data[col].unique())
                
                all_categories = list(all_categories)
                
                # Create one-hot encoded columns
                for category in all_categories:
                    if pd.isna(category):
                        continue  # Skip NaN categories
                        
                    new_col = f"{col}_{category}"
                    self.train_data[new_col] = (self.train_data[col] == category).astype(int)
                    self.test_data[new_col] = (self.test_data[col] == category).astype(int)
                    if self.val_data is not None:
                        self.val_data[new_col] = (self.val_data[col] == category).astype(int)
                    
                    # Add new column to features
                    self.feature_columns.append(new_col)
                
                # Remove original categorical column from features
                self.feature_columns.remove(col)
        
        # Normalize numeric features
        if normalize:
            numeric_cols = [col for col in self.feature_columns 
                          if col not in self.categorical_columns 
                          and pd.api.types.is_numeric_dtype(self.train_data[col])]
            
            self.logger.info(f"Normalizing {len(numeric_cols)} numeric features")
            
            for col in numeric_cols:
                # Calculate mean and std from training data
                mean_val = self.train_data[col].mean()
                std_val = self.train_data[col].std()
                
                # Avoid division by zero
                if std_val == 0:
                    std_val = 1
                
                # Store stats for later use
                self.column_stats[col] = {'mean': mean_val, 'std': std_val}
                
                # Apply normalization
                self.train_data[col] = (self.train_data[col] - mean_val) / std_val
                self.test_data[col] = (self.test_data[col] - mean_val) / std_val
                if self.val_data is not None:
                    self.val_data[col] = (self.val_data[col] - mean_val) / std_val
        
        self.is_preprocessed = True
        self.logger.info("Preprocessing complete")
    
    def get_features(self, split: str = 'train') -> np.ndarray:
        """
        Get feature matrix for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
        
        Returns:
            numpy array of features
        """
        # Ensure data is loaded
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get the correct dataframe
        if split == 'train':
            df = self.train_data
        elif split == 'test':
            df = self.test_data
        elif split == 'val':
            if self.val_data is None:
                raise ValueError("Validation data not available.")
            df = self.val_data
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', or 'val'.")
        
        # Extract features
        features = df[self.feature_columns].to_numpy()
        return features
    
    def get_labels(self, split: str = 'train') -> np.ndarray:
        """
        Get labels for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
        
        Returns:
            numpy array of labels
        """
        # Ensure data is loaded
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check if target column is specified
        if self.target_column is None:
            raise ValueError("Target column not specified.")
        
        # Get the correct dataframe
        if split == 'train':
            df = self.train_data
        elif split == 'val':
            if self.val_data is None:
                raise ValueError("Validation data not available.")
            df = self.val_data
        elif split == 'test':
            # Test data might not have labels
            if self.target_column not in self.test_data.columns:
                raise ValueError(f"Target column '{self.target_column}' not in test data.")
            df = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', or 'val'.")
        
        # Extract labels
        labels = df[self.target_column].to_numpy()
        return labels
    
    def get_ids(self, split: str = 'train') -> np.ndarray:
        """
        Get IDs for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
        
        Returns:
            numpy array of IDs
        """
        # Ensure data is loaded
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check if ID column is specified
        if self.id_column is None:
            raise ValueError("ID column not specified.")
        
        # Get the correct dataframe
        if split == 'train':
            df = self.train_data
        elif split == 'test':
            df = self.test_data
        elif split == 'val':
            if self.val_data is None:
                raise ValueError("Validation data not available.")
            df = self.val_data
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', or 'val'.")
        
        # Extract IDs
        ids = df[self.id_column].to_numpy()
        return ids
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features.
        
        Returns:
            List of feature names
        """
        return self.feature_columns
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary with dataset summary
        """
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'val_samples': len(self.val_data) if self.val_data is not None else 0,
            'features': len(self.feature_columns),
            'categorical_features': len(self.categorical_columns),
            'date_features': len(self.date_columns),
            'target_column': self.target_column,
            'id_column': self.id_column,
            'is_preprocessed': self.is_preprocessed,
            'feature_names': self.feature_columns,
        }
        
        # Include class distribution for classification problems
        if not self.is_regression and self.target_column is not None:
            class_counts = self.train_data[self.target_column].value_counts().to_dict()
            summary['class_distribution'] = class_counts
        
        return summary 