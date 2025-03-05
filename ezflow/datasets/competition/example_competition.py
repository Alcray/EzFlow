from typing import Optional, Union, Dict, Any
from pathlib import Path

from ...core.dataset.molecular import MolecularDataset

class ExampleCompetitionDataset(MolecularDataset):
    """
    Example competition dataset that demonstrates how to implement
    a competition-specific dataset using the MolecularDataset base class.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the example competition dataset.
        
        Args:
            data_dir: Directory containing the competition data
            cache_dir: Directory for storing processed data and cache
            metadata: Additional dataset-specific metadata
        """
        super().__init__(
            name="example_competition",
            data_dir=data_dir,
            smiles_col="SMILES",  # Competition-specific column name
            label_col="Target",   # Competition-specific column name
            id_col="MoleculeID", # Competition-specific column name
            cache_dir=cache_dir,
            metadata=metadata,
            # Competition-specific fingerprint parameters
            fingerprint_radius=3,
            fingerprint_bits=4096
        )
    
    def preprocess(self) -> None:
        """
        Add competition-specific preprocessing steps.
        """
        # First do the standard molecular preprocessing
        super().preprocess()
        
        # Add competition-specific preprocessing
        for df in [self.train_data, self.val_data, self.test_data]:
            if df is not None:
                # Example: Remove molecules with molecular weight > 500
                df['mol'] = df[self.smiles_col].apply(lambda s: Chem.MolFromSmiles(s))
                df['mw'] = df['mol'].apply(lambda m: Descriptors.ExactMolWt(m))
                df = df[df['mw'] <= 500].drop(['mol', 'mw'], axis=1)
                
                # Example: Normalize target values
                if self.label_col in df.columns:
                    df[self.label_col] = (df[self.label_col] - df[self.label_col].mean()) / df[self.label_col].std() 