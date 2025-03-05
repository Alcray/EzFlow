import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BaseDataset(ABC):
    """
    Abstract base class for all datasets in the ezflow framework.
    
    This class defines the interface that all dataset implementations must adhere to.
    When creating a custom dataset for a specific hackathon, inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, data_dir: str, **kwargs):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory where the dataset files are stored
            **kwargs: Additional keyword arguments for dataset-specific initialization
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.meta_info = {}
        self._process_kwargs(**kwargs)
        
    def _process_kwargs(self, **kwargs):
        """Process additional keyword arguments. Override in subclasses if needed."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def load_data(self) -> None:
        """
        Load data from files and prepare initial dataframes or data structures.
        
        This method should populate self.train_data, self.test_data, and optionally self.val_data.
        """
        pass
    
    @abstractmethod
    def get_features(self, split: str = 'train') -> np.ndarray:
        """
        Get features for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
            
        Returns:
            numpy array of features
        """
        pass
    
    @abstractmethod
    def get_labels(self, split: str = 'train') -> np.ndarray:
        """
        Get labels for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
            
        Returns:
            numpy array of labels
        """
        pass
    
    def get_data(self, split: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get both features and labels for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
            
        Returns:
            Tuple of (features, labels) where labels may be None for test data
        """
        features = self.get_features(split)
        try:
            labels = self.get_labels(split)
        except (ValueError, AttributeError):
            self.logger.warning(f"Labels not available for {split} split, returning None")
            labels = None
        return features, labels
    
    def create_train_val_split(self, val_ratio: float = 0.2, random_state: int = 42) -> None:
        """
        Create a validation split from the training data.
        
        Args:
            val_ratio: Fraction of training data to use for validation
            random_state: Random seed for reproducibility
        """
        if self.train_data is None:
            raise ValueError("Train data must be loaded first. Call load_data() method.")
        
        # Default implementation using random splitting
        from sklearn.model_selection import train_test_split
        
        train_idx, val_idx = train_test_split(
            np.arange(len(self.train_data)),
            test_size=val_ratio,
            random_state=random_state
        )
        
        self.val_data = self.train_data.iloc[val_idx].reset_index(drop=True)
        self.train_data = self.train_data.iloc[train_idx].reset_index(drop=True)
        self.logger.info(f"Created validation split: {len(self.val_data)} samples")
        self.logger.info(f"Remaining training samples: {len(self.train_data)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the dataset.
        
        Returns:
            Dictionary with dataset metadata
        """
        meta = {
            "name": self.__class__.__name__,
            "train_samples": len(self.train_data) if self.train_data is not None else 0,
            "test_samples": len(self.test_data) if self.test_data is not None else 0,
            "val_samples": len(self.val_data) if self.val_data is not None else 0,
            **self.meta_info
        }
        return meta
    
    def save_split(self, output_dir: str, split: str = 'all') -> None:
        """
        Save the specified data split to CSV files.
        
        Args:
            output_dir: Directory to save the split data
            split: One of 'train', 'test', 'val', or 'all'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if split == 'all' or split == 'train':
            if self.train_data is not None:
                self.train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
                self.logger.info(f"Saved train split to {output_dir}/train.csv")
        
        if split == 'all' or split == 'test':
            if self.test_data is not None:
                self.test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
                self.logger.info(f"Saved test split to {output_dir}/test.csv")
        
        if split == 'all' or split == 'val':
            if self.val_data is not None:
                self.val_data.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
                self.logger.info(f"Saved validation split to {output_dir}/val.csv")
    
    def __len__(self) -> int:
        """Return the number of samples in the training set."""
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        return len(self.train_data) 