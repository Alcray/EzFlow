# Dataset Module

The Dataset module is a core component of EZFlow, providing a unified interface for loading, preprocessing, and manipulating different types of datasets.

## Table of Contents

- [Overview](#overview)
- [BaseDataset](#basedataset)
- [Specialized Datasets](#specialized-datasets)
  - [IrisDataset](#irisdataset)
  - [MolecularDataset](#moleculardataset)
- [Custom Datasets](#custom-datasets)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The Dataset module handles:

- Data loading from various sources
- Data preprocessing and feature engineering
- Data splitting into training, validation, and test sets
- Feature and label extraction
- Persistence of datasets and splits

This module is designed to be extendable, allowing users to easily implement custom datasets while maintaining a consistent interface.

## BaseDataset

`BaseDataset` is the abstract base class that all dataset implementations should inherit from.

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Name of the dataset |
| `data_dir` | str/Path | Directory where data files are stored |
| `cache_dir` | str/Path | Directory for caching processed data |
| `train_data` | pd.DataFrame | Training data |
| `val_data` | pd.DataFrame | Validation data |
| `test_data` | pd.DataFrame | Test data (optional) |
| `metadata` | Dict | Additional metadata about the dataset |

### Key Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `load_data` | - | None | Loads raw data into `train_data` |
| `split_data` | `val_size`, `random_state` | None | Splits `train_data` into `train_data` and `val_data` |
| `preprocess` | - | None | Applies preprocessing transformations |
| `get_features` | `data: pd.DataFrame` | np.ndarray | Extracts features from data |
| `get_labels` | `data: pd.DataFrame` | np.ndarray | Extracts labels from data |
| `save_splits` | - | None | Saves train/val splits to disk |
| `load_splits` | - | None | Loads train/val splits from disk |

### Usage Example

```python
from ezflow.core.dataset import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, data_dir):
        super().__init__("my_dataset", data_dir)
    
    def load_data(self):
        # Implementation specific to this dataset
        pass
    
    def preprocess(self):
        # Implementation specific to this dataset
        pass
    
    def get_features(self, data):
        # Implementation specific to this dataset
        pass
    
    def get_labels(self, data):
        # Implementation specific to this dataset
        pass
```

## Specialized Datasets

EZFlow provides several built-in dataset implementations for common use cases.

### IrisDataset

The `IrisDataset` class provides a wrapper for the popular Iris flower dataset, commonly used for classification tasks.

#### Initialization

```python
from ezflow.core.dataset import IrisDataset

# Initialize with a directory for storing data
dataset = IrisDataset(data_dir="./data")

# Load the data
dataset.load_data()

# Split into training and validation sets
dataset.split_data(val_size=0.2, random_state=42)
```

#### Features and Implementation Details

- Automatically loads the Iris dataset from scikit-learn
- Provides standard feature columns: sepal length, sepal width, petal length, petal width
- Target variable contains flower species (0, 1, 2)
- Implements stratified splitting to maintain class distribution
- Handles index preservation during split/save/load operations

### MolecularDataset

The `MolecularDataset` class is designed for working with chemical/molecular data, particularly useful in drug discovery and cheminformatics.

#### Initialization

```python
from ezflow.core.dataset import MolecularDataset

# Initialize with dataset details
dataset = MolecularDataset(
    name="my_molecular_dataset",
    data_dir="./data",
    smiles_col="SMILES",
    label_col="Activity",
    fingerprint_radius=2,
    fingerprint_bits=2048
)

# Load the data (expects CSV files with SMILES strings)
dataset.load_data()

# Split into training and validation sets
dataset.split_data(val_size=0.2, random_state=42)
```

#### Features and Implementation Details

- Handles SMILES (Simplified Molecular Input Line Entry System) strings
- Generates molecular fingerprints using RDKit
- Provides customizable fingerprint parameters (radius, bit length)
- Supports custom ID columns and metadata
- Includes functionality for generating submission files for competitions

## Custom Datasets

Creating a custom dataset involves inheriting from `BaseDataset` and implementing the required methods.

### Implementation Checklist

1. **Inherit from BaseDataset**: Extend the base class
2. **Implement load_data**: Define how to load your specific data
3. **Implement preprocess**: Define any preprocessing steps
4. **Implement get_features**: Define which columns/transformations to use as features
5. **Implement get_labels**: Define which column(s) to use as labels
6. **Implement save/load_splits**: Optional but recommended, to save/load the dataset state

### Example: Custom Image Dataset

```python
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from ezflow.core.dataset import BaseDataset

class ImageDataset(BaseDataset):
    def __init__(self, data_dir, image_size=(224, 224)):
        super().__init__("image_dataset", data_dir)
        self.image_size = image_size
        self.image_paths = []
        self.labels = []
    
    def load_data(self):
        # Create DataFrame with image paths and labels
        image_files = list(Path(self.data_dir).glob("**/*.jpg"))
        labels = [path.parent.name for path in image_files]
        
        self.train_data = pd.DataFrame({
            "image_path": [str(path) for path in image_files],
            "label": labels
        })
    
    def preprocess(self):
        # You might do label encoding here
        label_map = {label: idx for idx, label in enumerate(set(self.train_data["label"]))}
        self.train_data["label_encoded"] = self.train_data["label"].map(label_map)
        self.metadata["label_map"] = label_map
    
    def _load_and_preprocess_image(self, image_path):
        # Load and preprocess a single image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        return img / 255.0  # Normalize
    
    def get_features(self, data):
        # Load all images in the batch
        images = np.array([
            self._load_and_preprocess_image(path) 
            for path in data["image_path"]
        ])
        return images
    
    def get_labels(self, data):
        return data["label_encoded"].values
```

## Best Practices

For optimal results when working with datasets:

1. **Data Validation**: Always validate your data during loading to catch issues early
2. **Efficient Preprocessing**: Implement preprocessing operations that can be efficiently applied to large datasets
3. **Consistent Splitting**: Use a fixed random state for reproducible splits
4. **Feature Engineering**: Keep complex feature engineering in the dataset class for consistency
5. **Documentation**: Document data formats, expected columns, and special handling requirements
6. **Error Handling**: Provide clear error messages for missing files or invalid data
7. **Caching**: Use the cache directory for storing intermediate results to avoid recomputation
8. **Type Safety**: Return numpy arrays with the appropriate dtypes for features and labels

## API Reference

### BaseDataset

```python
class BaseDataset:
    """Abstract base class for all datasets in EZFlow."""
    
    def __init__(
        self,
        name: str,
        data_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        name : str
            Name of the dataset
        data_dir : str or Path
            Directory where data files are stored
        cache_dir : str or Path, optional
            Directory for caching processed data
        metadata : dict, optional
            Additional metadata about the dataset
        """
        
    def load_data(self) -> None:
        """
        Load the raw data into the dataset.
        
        This method should be implemented by subclasses to
        load data from files into the `train_data` attribute.
        """
        
    def split_data(self, val_size: float = 0.2, random_state: int = 42) -> None:
        """
        Split data into training and validation sets.
        
        Parameters
        ----------
        val_size : float
            Size of validation set as a fraction
        random_state : int
            Random seed for reproducibility
        """
        
    def preprocess(self) -> None:
        """
        Apply preprocessing transformations to the data.
        
        This method should be implemented by subclasses to
        perform any necessary preprocessing steps.
        """
        
    def get_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from the data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to extract features from
            
        Returns
        -------
        numpy.ndarray
            Features as a numpy array
        """
        
    def get_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract labels from the data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to extract labels from
            
        Returns
        -------
        numpy.ndarray
            Labels as a numpy array
        """
        
    def save_splits(self) -> None:
        """
        Save the current train/validation splits to disk.
        """
        
    def load_splits(self) -> None:
        """
        Load train/validation splits from disk.
        """
```

### IrisDataset

```python
class IrisDataset(BaseDataset):
    """Dataset implementation for the Iris flower dataset."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the Iris dataset.
        
        Parameters
        ----------
        data_dir : str or Path
            Directory for storing data files
        """
```

### MolecularDataset

```python
class MolecularDataset(BaseDataset):
    """Dataset implementation for molecular/chemical data."""
    
    def __init__(
        self,
        name: str,
        data_dir: Union[str, Path],
        smiles_col: str = 'smiles',
        label_col: str = 'activity',
        id_col: Optional[str] = 'id',
        cache_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048
    ):
        """
        Initialize the molecular dataset.
        
        Parameters
        ----------
        name : str
            Name of the dataset
        data_dir : str or Path
            Directory where data files are stored
        smiles_col : str
            Column name containing SMILES strings
        label_col : str
            Column name containing target values
        id_col : str, optional
            Column name containing molecule identifiers
        cache_dir : str or Path, optional
            Directory for caching processed data
        metadata : dict, optional
            Additional metadata about the dataset
        fingerprint_radius : int
            Radius parameter for Morgan fingerprints
        fingerprint_bits : int
            Number of bits in the fingerprint
        """
``` 