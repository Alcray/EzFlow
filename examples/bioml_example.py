#!/usr/bin/env python3
"""
Example of creating a custom dataset for bioinformatics data.

This example shows how to:
1. Create a custom dataset by inheriting from BaseDataset
2. Implement SMILES fingerprinting for molecular representation
3. Run experiments with different molecular features
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Import our ezflow framework
import ezflow as ez

# Set up logger
logger = ez.get_logger('bioml_example')

# --------------------------------
# Step 1: Create custom dataset for bioinformatics
# --------------------------------
class BioMLDataset(ez.BaseDataset):
    """
    Custom dataset implementation for bioinformatics data.
    This class handles SMILES data and molecular fingerprinting.
    """
    
    def __init__(
        self, 
        data_dir: str,
        smiles_column: str = 'smiles',
        target_column: str = 'activity',
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        **kwargs
    ):
        """
        Initialize the bioinformatics dataset.
        
        Args:
            data_dir: Directory containing the data files
            smiles_column: Name of the column containing SMILES strings
            target_column: Name of the target column
            fingerprint_radius: Radius for Morgan fingerprint
            fingerprint_bits: Number of bits for fingerprint
            **kwargs: Additional keyword arguments
        """
        super().__init__(data_dir, **kwargs)
        
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        
        # Containers for molecular data
        self.train_fingerprints = None
        self.test_fingerprints = None
        self.val_fingerprints = None
    
    def load_data(self) -> None:
        """
        Load the bioinformatics data files.
        """
        # Load training data if available
        train_path = os.path.join(self.data_dir, 'train.csv')
        if os.path.exists(train_path):
            self.train_data = pd.read_csv(train_path)
            logger.info(f"Loaded training data: {len(self.train_data)} molecules")
        
        # Load test data if available
        test_path = os.path.join(self.data_dir, 'test.csv')
        if os.path.exists(test_path):
            self.test_data = pd.read_csv(test_path)
            logger.info(f"Loaded test data: {len(self.test_data)} molecules")
        
        # Check if data was loaded
        if self.train_data is None and self.test_data is None:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")
        
        # Store metadata
        self.meta_info = {
            'train_molecules': len(self.train_data) if self.train_data is not None else 0,
            'test_molecules': len(self.test_data) if self.test_data is not None else 0,
            'fingerprint_radius': self.fingerprint_radius,
            'fingerprint_bits': self.fingerprint_bits
        }
    
    def generate_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Generate Morgan fingerprint from SMILES string.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            numpy array with molecular fingerprint
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return np.zeros(self.fingerprint_bits)
            
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fingerprint_radius, nBits=self.fingerprint_bits
            )
            return np.array(fingerprint)
        except Exception as e:
            logger.error(f"Error generating fingerprint for {smiles}: {e}")
            return np.zeros(self.fingerprint_bits)
    
    def prepare_fingerprints(self) -> None:
        """
        Prepare fingerprints for all datasets.
        """
        if self.train_data is not None and self.smiles_column in self.train_data.columns:
            logger.info("Generating fingerprints for training data...")
            self.train_fingerprints = np.array([
                self.generate_fingerprint(smiles) 
                for smiles in self.train_data[self.smiles_column]
            ])
        
        if self.test_data is not None and self.smiles_column in self.test_data.columns:
            logger.info("Generating fingerprints for test data...")
            self.test_fingerprints = np.array([
                self.generate_fingerprint(smiles) 
                for smiles in self.test_data[self.smiles_column]
            ])
        
        if self.val_data is not None and self.smiles_column in self.val_data.columns:
            logger.info("Generating fingerprints for validation data...")
            self.val_fingerprints = np.array([
                self.generate_fingerprint(smiles) 
                for smiles in self.val_data[self.smiles_column]
            ])
    
    def get_features(self, split: str = 'train') -> np.ndarray:
        """
        Get fingerprint features for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
            
        Returns:
            numpy array of fingerprint features
        """
        # Ensure fingerprints are generated
        if self.train_fingerprints is None and self.test_fingerprints is None:
            self.prepare_fingerprints()
        
        if split == 'train':
            if self.train_fingerprints is None:
                raise ValueError("Training fingerprints not available")
            return self.train_fingerprints
        elif split == 'test':
            if self.test_fingerprints is None:
                raise ValueError("Test fingerprints not available")
            return self.test_fingerprints
        elif split == 'val':
            if self.val_fingerprints is None:
                raise ValueError("Validation fingerprints not available")
            return self.val_fingerprints
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', or 'val'.")
    
    def get_labels(self, split: str = 'train') -> np.ndarray:
        """
        Get activity labels for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
            
        Returns:
            numpy array of activity labels
        """
        if split == 'train':
            if self.train_data is None or self.target_column not in self.train_data.columns:
                raise ValueError("Training labels not available")
            return self.train_data[self.target_column].to_numpy()
        elif split == 'val':
            if self.val_data is None or self.target_column not in self.val_data.columns:
                raise ValueError("Validation labels not available")
            return self.val_data[self.target_column].to_numpy()
        elif split == 'test':
            if self.test_data is None or self.target_column not in self.test_data.columns:
                raise ValueError("Test labels not available")
            return self.test_data[self.target_column].to_numpy()
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', or 'val'.")
    
    def get_smiles(self, split: str = 'train') -> list:
        """
        Get SMILES strings for the specified split.
        
        Args:
            split: One of 'train', 'test', or 'val'
            
        Returns:
            list of SMILES strings
        """
        if split == 'train':
            if self.train_data is None or self.smiles_column not in self.train_data.columns:
                raise ValueError("Training SMILES not available")
            return self.train_data[self.smiles_column].tolist()
        elif split == 'val':
            if self.val_data is None or self.smiles_column not in self.val_data.columns:
                raise ValueError("Validation SMILES not available")
            return self.val_data[self.smiles_column].tolist()
        elif split == 'test':
            if self.test_data is None or self.smiles_column not in self.test_data.columns:
                raise ValueError("Test SMILES not available")
            return self.test_data[self.smiles_column].tolist()
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', or 'val'.")


# --------------------------------
# Step 2: Create synthetic molecular data for demo
# --------------------------------

def create_synthetic_data():
    """Create synthetic SMILES data for demonstration."""
    
    # Example SMILES strings (simple compounds)
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CCO",  # Ethanol
        "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=C(C=C(C=C1)S(=O)(=O)N)CC(C(=O)O)N",  # Sulfanilamide
        "CCOC(=O)C1=CC=C(C=C1)O",  # Ethyl parahydroxybenzoate
        "C(C(=O)O)N",  # Glycine
        "C(C(C(=O)O)N)C(=O)O",  # Aspartic acid
        "CC(C(=O)O)N",  # Alanine
    ]
    
    # Generate more examples with slight variations
    extended_smiles = []
    for smiles in smiles_list:
        extended_smiles.append(smiles)
        # Add a methyl group to the original SMILES (simplified approach)
        extended_smiles.append("C" + smiles)
    
    # Duplicate and add more variations to get about 100 samples
    all_smiles = []
    for _ in range(5):
        all_smiles.extend(extended_smiles)
    
    # Create random activity values (1 = active, 0 = inactive)
    np.random.seed(42)
    activities = np.random.randint(0, 2, size=len(all_smiles))
    
    # Create dataframe
    df = pd.DataFrame({
        'smiles': all_smiles,
        'activity': activities
    })
    
    # Split into train and test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Create directories
    os.makedirs('bioml_data', exist_ok=True)
    
    # Save files
    train_df.to_csv('bioml_data/train.csv', index=False)
    test_df.to_csv('bioml_data/test.csv', index=False)
    
    logger.info(f"Created synthetic dataset with {len(train_df)} training samples and {len(test_df)} test samples")
    return train_df, test_df


# --------------------------------
# Step 3: Run the example
# --------------------------------

def main():
    """Run the BioML example."""
    
    print("BioML Example - Molecular Activity Prediction")
    print("=============================================")
    
    # Create synthetic data
    print("\nCreating synthetic molecular data...")
    create_synthetic_data()
    
    # Create dataset
    print("\nInitializing BioML dataset...")
    dataset = BioMLDataset(
        data_dir='bioml_data',
        smiles_column='smiles',
        target_column='activity',
        fingerprint_radius=2,
        fingerprint_bits=2048
    )
    
    # Load data
    dataset.load_data()
    
    # Create validation split
    dataset.create_train_val_split(val_ratio=0.2, random_state=42)
    
    # Prepare fingerprints
    print("\nGenerating molecular fingerprints...")
    dataset.prepare_fingerprints()
    
    # Get features and labels
    X_train = dataset.get_features('train')
    y_train = dataset.get_labels('train')
    X_val = dataset.get_features('val')
    y_val = dataset.get_labels('val')
    X_test = dataset.get_features('test')
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Print some examples
    train_smiles = dataset.get_smiles('train')
    print("\nExample molecules from training set:")
    for i in range(min(5, len(train_smiles))):
        print(f"  SMILES: {train_smiles[i]}, Activity: {y_train[i]}")
    
    # Create experiment
    print("\nRunning experiment...")
    experiment = ez.Experiment(
        name="bioml_rf",
        description="Random Forest for molecular activity prediction",
        tags=["bioml", "fingerprint", "random_forest"]
    )
    
    experiment.log_params({
        "fingerprint_radius": dataset.fingerprint_radius,
        "fingerprint_bits": dataset.fingerprint_bits,
        "n_estimators": 100,
        "max_depth": 10,
        "n_molecules_train": len(dataset.get_smiles('train')),
        "n_molecules_val": len(dataset.get_smiles('val')),
        "n_molecules_test": len(dataset.get_smiles('test')),
    })
    
    # Train model
    with experiment.train_phase():
        model = ez.Model(
            model=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            model_type="classifier",
            name="BioML_RF"
        )
        model.fit(X_train, y_train)
        
        # Save the model
        model_path = model.save(experiment.experiment_dir)
        experiment.log_artifact("model", model_path)
    
    # Evaluate model
    with experiment.eval_phase():
        # Evaluate on validation data
        metrics = model.evaluate(X_val, y_val)
        for metric_name, value in metrics.items():
            experiment.log_metric(metric_name, value)
        
        # Plot confusion matrix
        y_pred = model.predict(X_val)
        ez.plot_confusion_matrix(
            y_val, y_pred, 
            save_path=f"{experiment.experiment_dir}/confusion_matrix.png"
        )
        experiment.log_artifact("confusion_matrix", f"{experiment.experiment_dir}/confusion_matrix.png")
        
        # Generate classification report
        report = model.generate_classification_report(X_val, y_val)
        experiment.log_params({"classification_report": report})
    
    experiment.finish()
    print(f"Experiment completed: {experiment.id}")
    
    # Create dashboard
    dashboard_path = ez.create_experiment_dashboard(experiment.experiment_dir)
    print(f"Experiment dashboard created at: {dashboard_path}")
    
    # Generate predictions on test data
    print("\nGenerating predictions for test set...")
    test_predictions = model.predict(X_test)
    test_smiles = dataset.get_smiles('test')
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'smiles': test_smiles,
        'predicted_activity': test_predictions
    })
    
    # Save submission
    submission_path = 'bioml_data/predictions.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Predictions saved to: {submission_path}")
    
    print("\nBioML example completed successfully!")


if __name__ == "__main__":
    main() 