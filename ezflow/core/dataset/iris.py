import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class IrisDataset(BaseDataset):
    def __init__(self, data_dir):
        super().__init__("iris", data_dir)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self):
        """Load the Iris dataset."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                          columns=iris['feature_names'] + ['target'])
        self.train_data = data

    def split_data(self, val_size=0.2, random_state=42):
        """Split data into train and validation sets."""
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get indices for splitting while preserving original data
        all_data = self.train_data.copy()
        indices = np.arange(len(all_data))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=random_state,
            stratify=all_data['target']
        )
        
        # Split the data using indices
        self.train_data = all_data.iloc[train_idx]
        self.val_data = all_data.iloc[val_idx]

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

    def get_features(self, data):
        """Extract features from data."""
        feature_cols = [col for col in data.columns if col != 'target']
        return data[feature_cols].values

    def get_labels(self, data):
        """Extract labels from data."""
        return data['target'].values 