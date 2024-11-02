import os
import pandas as pd
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

class BioMLDataHandler:
    def __init__(self, data_path: str = 'data', split_data_path: str = 'split_data'):
        """
        Initializes the data handler and loads datasets from specified directories.

        Parameters:
            data_path (str): Path to the 'data' directory containing train.csv and test.csv.
            split_data_path (str): Path to the 'split_data' directory containing train_split.csv and val_split.csv.
        """
        self.data_path = data_path
        self.split_data_path = split_data_path

        self.train = None
        self.test = None
        self.train_split = None
        self.val_split = None
        self.model = None

        # Load datasets
        self.load_data()

    def load_data(self):
        """
        Loads data from the specified directories into the handler.
        """
        try:
            # Load the main train and test datasets
            self.train = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
            self.test = pd.read_csv(os.path.join(self.data_path, 'test.csv'))
            print("Main train and test datasets loaded successfully.")

            # Load the train and validation splits
            self.train_split = pd.read_csv(os.path.join(self.split_data_path, 'train_split.csv'))
            self.val_split = pd.read_csv(os.path.join(self.split_data_path, 'val_split.csv'))
            print("Train and validation splits loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise

    @staticmethod
    def generate_fingerprint(smiles: str, radius=2, n_bits=2048) -> np.ndarray:
        """
        Converts a SMILES string to an ECFP4 fingerprint.

        Parameters:
            smiles (str): SMILES representation of a molecule.
            radius (int): Radius for Morgan fingerprint. Default is 2 for ECFP4.
            n_bits (int): Size of the fingerprint bit vector. Default is 2048.

        Returns:
            np.ndarray: Array representing the molecule's fingerprint.
        """
        mol = Chem.MolFromSmiles(smiles)
        generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fingerprint = generator.GetFingerprint(mol)
        return np.array(fingerprint)

    def get_train_data(self, representation: str = 'fingerprint') -> (np.ndarray, np.ndarray):
        """
        Prepares and returns the train split data.

        Parameters:
            representation (str): 'fingerprint' or 'smiles' for feature representation.

        Returns:
            tuple: Feature matrix (X_train) and labels (y_train).
        """
        self.prepare_train_data(representation)
        return self.X_train, self.y_train

    def get_original_train_data(self, representation: str = 'fingerprint') -> (np.ndarray, np.ndarray):
        """
        Prepares and returns the original full train data.

        Parameters:
            representation (str): 'fingerprint' or 'smiles' for feature representation.

        Returns:
            tuple: Feature matrix (X_full_train) and labels (y_full_train).
        """
        if self.train is None:
            raise ValueError("The full train dataset is not loaded.")
        
        self.X_full_train, self.y_full_train = self.prepare_data(self.train, representation)
        print("Original full training data prepared.")
        return self.X_full_train, self.y_full_train

    def get_test_data(self, representation: str = 'fingerprint') -> np.ndarray:
        """
        Prepares and returns the test data.

        Parameters:
            representation (str): 'fingerprint' or 'smiles' for feature representation.

        Returns:
            np.ndarray: Feature matrix (X_test).
        """
        self.prepare_test_data(representation)
        return self.X_test

    def get_val_data(self, representation: str = 'fingerprint') -> (np.ndarray, np.ndarray):
        """
        Prepares and returns the validation data.

        Parameters:
            representation (str): 'fingerprint' or 'smiles' for feature representation.

        Returns:
            tuple: Feature matrix (X_val) and labels (y_val).
        """
        self.prepare_validation_data(representation)
        return self.X_val, self.y_val

    def prepare_data(self, dataset: pd.DataFrame, representation: str = 'fingerprint') -> (np.ndarray, np.ndarray):
        """
        Generates features and labels for a given dataset based on the specified representation.

        Parameters:
            dataset (pd.DataFrame): DataFrame with 'smiles' and 'activity' columns.
            representation (str): 'fingerprint' or 'smiles' for feature representation.

        Returns:
            tuple: Feature matrix (X) and labels (y).
        """
        if representation == 'fingerprint':
            dataset['feature'] = dataset['smiles'].apply(self.generate_fingerprint)
            X = np.stack(dataset['feature'].values)
        elif representation == 'smiles':
            X = dataset['smiles'].values
        else:
            raise ValueError("Invalid representation. Choose 'fingerprint' or 'smiles'.")
        
        y = dataset['activity'].values
        return X, y

    def prepare_train_data(self, representation: str = 'fingerprint'):
        """
        Prepares the train split for model training.
        
        Parameters:
            representation (str): 'fingerprint' or 'smiles' for feature representation.
        """
        self.X_train, self.y_train = self.prepare_data(self.train_split, representation)
        print("Training data prepared.")

    def prepare_validation_data(self, representation: str = 'fingerprint'):
        """
        Prepares the validation split for model evaluation.
        
        Parameters:
            representation (str): 'fingerprint' or 'smiles' for feature representation.
        """
        self.X_val, self.y_val = self.prepare_data(self.val_split, representation)
        print("Validation data prepared.")

    def prepare_test_data(self, representation: str = 'fingerprint'):
        """
        Prepares the test dataset.
        
        Parameters:
            representation (str): 'fingerprint' or 'smiles' for feature representation.
        """
        if self.test is None:
            raise ValueError("Test data is not loaded.")
        
        if representation == 'fingerprint':
            self.test['feature'] = self.test['smiles'].apply(self.generate_fingerprint)
            self.X_test = np.stack(self.test['feature'].values)
        elif representation == 'smiles':
            self.X_test = self.test['smiles'].values
        else:
            raise ValueError("Invalid representation. Choose 'fingerprint' or 'smiles'.")

        print("Test data prepared.")

    def train_model(self):
        """
        Trains a logistic regression model using the training data fingerprints.
        """
        if isinstance(self.X_train[0], str):
            raise ValueError("Model training requires fingerprint representation, not SMILES strings.")
        
        self.model = LogisticRegression(solver='liblinear')
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def evaluate_model(self) -> float:
        """
        Evaluates the model using average precision score on validation data.

        Returns:
            float: The average precision score on validation data.
        """
        if not hasattr(self, 'X_val') or not hasattr(self, 'y_val'):
            raise ValueError("Validation data is not prepared.")
        
        val_preds = self.predict(self.X_val)
        score = average_precision_score(self.y_val, val_preds)
        print(f"Validation Average Precision Score: {score}")
        return score

    def predict(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Predicts activity probabilities for a given set of fingerprints.

        Parameters:
            fingerprints (np.ndarray): Feature matrix for predictions.

        Returns:
            np.ndarray: Array of probabilities for the positive class.
        """
        if not self.model:
            raise ValueError("Model is not trained.")
        return self.model.predict_proba(fingerprints)[:, 1]

    def generate_submission(self, filename: str = 'submission.csv'):
        """
        Generates a submission file with predictions for the test dataset.

        Parameters:
            filename (str): Name of the output submission file. Default is 'submission.csv'.
        """
        if not hasattr(self, 'X_test'):
            raise ValueError("Test data is not prepared.")
        
        test_preds = self.predict(self.X_test)
        self.test['activity'] = test_preds
        submission = self.test[['id', 'activity']]
        submission.to_csv(filename, index=False)
        print(f"Submission file '{filename}' generated successfully.")

    def export_to_jsonl(self, dataset: pd.DataFrame, filename: str):
        """
        Exports a dataset to JSONL format with 'id', 'smiles', and 'activity' fields.

        Parameters:
            dataset (pd.DataFrame): DataFrame to be exported.
            filename (str): Name of the output JSONL file.
        """
        with open(filename, 'w') as f:
            for _, row in dataset.iterrows():
                entry = {
                    "id": row.get("id", None),
                    "smiles": row["smiles"],
                    "activity": row.get("activity", None)
                }
                f.write(json.dumps(entry) + "\n")
        print(f"Data exported to '{filename}' in JSONL format.")
