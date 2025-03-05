import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from ezflow.examples.iris_example import IrisDataset

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def test_iris_dataset_initialization(temp_dir):
    """Test IrisDataset initialization."""
    dataset = IrisDataset(data_dir=str(temp_dir))
    
    assert dataset.name == "iris"
    assert dataset.data_dir == temp_dir
    assert dataset.train_data is None
    assert dataset.val_data is None
    assert dataset.test_data is None

def test_load_data(temp_dir):
    """Test data loading."""
    dataset = IrisDataset(data_dir=str(temp_dir))
    dataset.load_data()
    
    assert isinstance(dataset.train_data, pd.DataFrame)
    assert len(dataset.train_data) == 150  # Iris dataset size
    assert dataset.train_data.shape[1] == 5  # 4 features + 1 target
    assert all(col in dataset.train_data.columns for col in [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)',
        'target'
    ])

def test_data_splitting(temp_dir):
    """Test data splitting functionality."""
    dataset = IrisDataset(data_dir=str(temp_dir))
    dataset.load_data()
    dataset.split_data(val_size=0.2, random_state=42)
    
    assert dataset.train_data is not None
    assert dataset.val_data is not None
    assert len(dataset.train_data) + len(dataset.val_data) == 150
    assert len(dataset.val_data) == 30  # 20% of 150

def test_get_features(temp_dir):
    """Test feature extraction."""
    dataset = IrisDataset(data_dir=str(temp_dir))
    dataset.load_data()
    
    features = dataset.get_features(dataset.train_data)
    assert isinstance(features, np.ndarray)
    assert features.shape == (150, 4)  # 150 samples, 4 features

def test_get_labels(temp_dir):
    """Test label extraction."""
    dataset = IrisDataset(data_dir=str(temp_dir))
    dataset.load_data()
    
    labels = dataset.get_labels(dataset.train_data)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (150,)  # 150 samples
    assert set(labels) == {0, 1, 2}  # Three classes in Iris

def test_save_load_splits(temp_dir):
    """Test saving and loading data splits."""
    dataset = IrisDataset(data_dir=str(temp_dir))
    dataset.load_data()
    dataset.split_data(val_size=0.2, random_state=42)
    
    # Save splits
    dataset.save_splits()
    
    # Create new dataset and load splits
    new_dataset = IrisDataset(data_dir=str(temp_dir))
    new_dataset.load_splits()
    
    assert new_dataset.train_data is not None
    assert new_dataset.val_data is not None
    
    # Compare data content without considering index
    pd.testing.assert_frame_equal(
        dataset.train_data.reset_index(drop=True),
        new_dataset.train_data.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        dataset.val_data.reset_index(drop=True),
        new_dataset.val_data.reset_index(drop=True)
    ) 