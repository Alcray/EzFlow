from typing import Any, Dict, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import os

from .base import BaseDataset

class MolecularDataset(BaseDataset):
    """
    Dataset class for handling molecular data (SMILES, fingerprints, etc.).
    Implements molecular-specific functionality on top of BaseDataset.
    """
    
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
        
        Args:
            name: Name of the dataset/competition
            data_dir: Directory containing the raw data
            smiles_col: Name of the column containing SMILES strings
            label_col: Name of the column containing labels
            id_col: Name of the column containing molecule IDs
            cache_dir: Directory for storing processed data and cache
            metadata: Additional dataset-specific metadata
            fingerprint_radius: Radius for Morgan fingerprint generation
            fingerprint_bits: Number of bits in the fingerprint
        """
        super().__init__(name, data_dir, cache_dir, metadata)
        
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.id_col = id_col
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        
        # Additional molecular-specific attributes
        self.has_3d = False
        self.has_fingerprints = False
        
    def generate_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Generate Morgan fingerprint from SMILES string.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Numpy array containing the fingerprint
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.fingerprint_radius,
                nBits=self.fingerprint_bits
            )
            return np.array(fingerprint)
        except Exception as e:
            raise ValueError(f"Error generating fingerprint for SMILES {smiles}: {str(e)}")
    
    def load_data(self) -> None:
        """
        Load molecular data from CSV files in the data directory.
        Expects 'train.csv' and optionally 'test.csv'.
        """
        train_path = self.data_dir / 'train.csv'
        test_path = self.data_dir / 'test.csv'
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        self.train_data = pd.read_csv(train_path)
        if test_path.exists():
            self.test_data = pd.read_csv(test_path)
            
        # Validate required columns
        required_cols = [self.smiles_col]
        if self.label_col and self.train_data is not None:
            required_cols.append(self.label_col)
            
        for df in [self.train_data, self.test_data]:
            if df is not None and not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
    
    def preprocess(self) -> None:
        """
        Preprocess the molecular data:
        1. Generate fingerprints
        2. Clean SMILES strings
        3. Handle missing values
        """
        for df in [self.train_data, self.val_data, self.test_data]:
            if df is not None:
                # Clean SMILES and generate fingerprints
                df['fingerprint'] = df[self.smiles_col].apply(self.generate_fingerprint)
                
                # Canonicalize SMILES
                df[self.smiles_col] = df[self.smiles_col].apply(
                    lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
                )
                
        self.has_fingerprints = True
    
    def get_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get molecular fingerprints as features.
        
        Args:
            data: DataFrame containing the molecular data
            
        Returns:
            Numpy array of fingerprints
        """
        if not self.has_fingerprints:
            raise ValueError("Fingerprints not generated. Call preprocess() first.")
            
        return np.stack(data['fingerprint'].values)
    
    def get_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get labels from the data.
        
        Args:
            data: DataFrame containing the molecular data
            
        Returns:
            Numpy array of labels
        """
        if self.label_col not in data.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in data")
            
        return data[self.label_col].values
    
    def generate_submission(
        self,
        predictions: np.ndarray,
        output_path: Union[str, Path]
    ) -> None:
        """
        Generate a submission file with predictions.
        
        Args:
            predictions: Array of predictions
            output_path: Path to save the submission file
        """
        if self.test_data is None:
            raise ValueError("Test data not loaded")
            
        if self.id_col not in self.test_data.columns:
            raise ValueError(f"ID column '{self.id_col}' not found in test data")
            
        submission = pd.DataFrame({
            self.id_col: self.test_data[self.id_col],
            self.label_col: predictions
        })
        
        submission.to_csv(output_path, index=False)

    def save_splits(self):
        """Save train and validation splits to disk."""
        if self.train_data is not None:
            self.train_data.to_csv(os.path.join(self.data_dir, "train_split.csv"))
        if self.val_data is not None:
            self.val_data.to_csv(os.path.join(self.data_dir, "val_split.csv"))

    def load_splits(self):
        """Load train and validation splits from disk."""
        train_path = os.path.join(self.data_dir, "train_split.csv")
        val_path = os.path.join(self.data_dir, "val_split.csv")

        if os.path.exists(train_path):
            self.train_data = pd.read_csv(train_path, index_col=0)
        if os.path.exists(val_path):
            self.val_data = pd.read_csv(val_path, index_col=0) 