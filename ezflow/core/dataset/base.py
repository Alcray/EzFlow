from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import pandas as pd
import json
from pathlib import Path

class BaseDataset(ABC):
    """
    Abstract base class for all datasets in EZFlow.
    
    This class defines the interface and common functionality that all dataset implementations
    must provide. It handles basic data operations, validation, and transformations.
    """
    
    def __init__(
        self,
        name: str,
        data_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            name: Name of the dataset/competition
            data_dir: Directory containing the raw data
            cache_dir: Directory for storing processed data and cache
            metadata: Additional dataset-specific metadata
        """
        self.name = name
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(data_dir) / '.cache'
        self.metadata = metadata or {}
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    @abstractmethod
    def load_data(self) -> None:
        """
        Load the raw data into memory.
        This method should be implemented by child classes to handle dataset-specific loading logic.
        """
        pass
    
    @abstractmethod
    def preprocess(self) -> None:
        """
        Preprocess the raw data into a format suitable for training.
        This method should be implemented by child classes to handle dataset-specific preprocessing.
        """
        pass
    
    def to_jsonl(self, data: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """
        Convert and save data to JSONL format.
        
        Args:
            data: DataFrame to convert
            output_path: Path to save the JSONL file
        """
        output_path = Path(output_path)
        with output_path.open('w') as f:
            for record in data.to_dict('records'):
                f.write(json.dumps(record) + '\n')
    
    def from_jsonl(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from a JSONL file.
        
        Args:
            input_path: Path to the JSONL file
            
        Returns:
            DataFrame containing the loaded data
        """
        records = []
        input_path = Path(input_path)
        with input_path.open('r') as f:
            for line in f:
                records.append(json.loads(line.strip()))
        return pd.DataFrame.from_records(records)
    
    def split_data(
        self,
        val_size: float = 0.2,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Split the data into train/val/test sets.
        
        Args:
            val_size: Fraction of data to use for validation
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        from sklearn.model_selection import train_test_split
        
        if self.train_data is None:
            raise ValueError("Data must be loaded before splitting")
            
        if test_size:
            # First split into train+val and test
            train_val, self.test_data = train_test_split(
                self.train_data,
                test_size=test_size,
                random_state=random_state
            )
            
            # Then split train+val into train and val
            self.train_data, self.val_data = train_test_split(
                train_val,
                test_size=val_size/(1-test_size),  # Adjust val_size to account for test split
                random_state=random_state
            )
        else:
            # Split only into train and val
            self.train_data, self.val_data = train_test_split(
                self.train_data,
                test_size=val_size,
                random_state=random_state
            )
    
    def save_splits(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Save the train/val/test splits to JSONL files.
        
        Args:
            output_dir: Directory to save the splits. If None, uses cache_dir.
        """
        output_dir = Path(output_dir) if output_dir else self.cache_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.train_data is not None:
            self.to_jsonl(self.train_data, output_dir / 'train.jsonl')
        if self.val_data is not None:
            self.to_jsonl(self.val_data, output_dir / 'val.jsonl')
        if self.test_data is not None:
            self.to_jsonl(self.test_data, output_dir / 'test.jsonl')
    
    def load_splits(self, input_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Load the train/val/test splits from JSONL files.
        
        Args:
            input_dir: Directory containing the splits. If None, uses cache_dir.
        """
        input_dir = Path(input_dir) if input_dir else self.cache_dir
        
        train_path = input_dir / 'train.jsonl'
        val_path = input_dir / 'val.jsonl'
        test_path = input_dir / 'test.jsonl'
        
        if train_path.exists():
            self.train_data = self.from_jsonl(train_path)
        if val_path.exists():
            self.val_data = self.from_jsonl(val_path)
        if test_path.exists():
            self.test_data = self.from_jsonl(test_path)
    
    @abstractmethod
    def get_features(self, data: pd.DataFrame) -> Any:
        """
        Extract features from the data.
        This method should be implemented by child classes to handle dataset-specific feature extraction.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            Features in the format required by the model
        """
        pass
    
    @abstractmethod
    def get_labels(self, data: pd.DataFrame) -> Any:
        """
        Extract labels from the data.
        This method should be implemented by child classes to handle dataset-specific label extraction.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            Labels in the format required by the model
        """
        pass 