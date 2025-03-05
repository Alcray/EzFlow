import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ezflow.core.dataset.molecular import MolecularDataset

@pytest.fixture
def test_data():
    """Create test data for molecular dataset."""
    data = pd.DataFrame({
        'smiles': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C'  # Testosterone
        ],
        'activity': [1, 0, 1],
        'id': ['mol1', 'mol2', 'mol3']
    })
    return data

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def test_molecular_dataset_initialization(temp_dir):
    """Test MolecularDataset initialization."""
    dataset = MolecularDataset(
        name="test",
        data_dir=temp_dir,
        smiles_col="smiles",
        label_col="activity",
        id_col="id"
    )
    
    assert dataset.name == "test"
    assert dataset.data_dir == temp_dir
    assert dataset.smiles_col == "smiles"
    assert dataset.label_col == "activity"
    assert dataset.id_col == "id"
    assert dataset.fingerprint_radius == 2
    assert dataset.fingerprint_bits == 2048

def test_generate_fingerprint(test_data):
    """Test fingerprint generation."""
    dataset = MolecularDataset("test", "dummy_dir")
    
    # Test valid SMILES
    fp = dataset.generate_fingerprint(test_data['smiles'][0])
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (2048,)
    
    # Test invalid SMILES
    with pytest.raises(ValueError):
        dataset.generate_fingerprint("invalid_smiles")

def test_load_data(temp_dir, test_data):
    """Test data loading."""
    # Save test data
    test_data.to_csv(temp_dir / "train.csv", index=False)
    
    dataset = MolecularDataset("test", temp_dir)
    dataset.load_data()
    
    assert isinstance(dataset.train_data, pd.DataFrame)
    assert len(dataset.train_data) == len(test_data)
    assert all(col in dataset.train_data.columns for col in ['smiles', 'activity', 'id'])

def test_preprocess(temp_dir, test_data):
    """Test data preprocessing."""
    # Save test data
    test_data.to_csv(temp_dir / "train.csv", index=False)
    
    dataset = MolecularDataset("test", temp_dir)
    dataset.load_data()
    dataset.preprocess()
    
    assert dataset.has_fingerprints
    assert 'fingerprint' in dataset.train_data.columns
    assert isinstance(dataset.train_data['fingerprint'].iloc[0], np.ndarray)

def test_get_features(temp_dir, test_data):
    """Test feature extraction."""
    # Save test data
    test_data.to_csv(temp_dir / "train.csv", index=False)
    
    dataset = MolecularDataset("test", temp_dir)
    dataset.load_data()
    dataset.preprocess()
    
    features = dataset.get_features(dataset.train_data)
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(test_data), 2048)

def test_get_labels(temp_dir, test_data):
    """Test label extraction."""
    # Save test data
    test_data.to_csv(temp_dir / "train.csv", index=False)
    
    dataset = MolecularDataset("test", temp_dir)
    dataset.load_data()
    
    labels = dataset.get_labels(dataset.train_data)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(test_data),)
    assert set(labels) == {0, 1}

def test_data_splitting(temp_dir, test_data):
    """Test data splitting functionality."""
    # Save test data
    test_data.to_csv(temp_dir / "train.csv", index=False)
    
    dataset = MolecularDataset("test", temp_dir)
    dataset.load_data()
    dataset.split_data(val_size=0.33, random_state=42)
    
    assert dataset.train_data is not None
    assert dataset.val_data is not None
    assert len(dataset.train_data) + len(dataset.val_data) == len(test_data)

def test_save_load_splits(temp_dir, test_data):
    """Test saving and loading data splits."""
    # Save test data
    test_data.to_csv(temp_dir / "train.csv", index=False)
    
    dataset = MolecularDataset("test", temp_dir)
    dataset.load_data()
    dataset.split_data(val_size=0.33, random_state=42)
    
    # Save splits
    dataset.save_splits()
    
    # Create new dataset and load splits
    new_dataset = MolecularDataset("test", temp_dir)
    new_dataset.load_splits()
    
    assert new_dataset.train_data is not None
    assert new_dataset.val_data is not None
    pd.testing.assert_frame_equal(dataset.train_data, new_dataset.train_data)
    pd.testing.assert_frame_equal(dataset.val_data, new_dataset.val_data) 