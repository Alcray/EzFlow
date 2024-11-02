import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

class BioMLDataHandler:
    def __init__(self, data_path: str = 'data', split_data_path: str = 'splitted'):
        """
        Initializes the data handler and loads datasets from specified directories.

        Parameters:
            data_path (str): Path to the 'data' directory containing train.csv and test.csv.
            split_data_path (str): Path to the 'splitted' directory containing train_split.csv and validation_split.csv.
        """
        self.data_path = data_path
        self.split_data_path = split_data_path

        self.train = None
        self.test = None
        self.train_split = None
        self.validation_split = None
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
            self.validation_split = pd.read_csv(os.path.join(self.split_data_path, 'validation_split.csv'))
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

    def prepare_data(self, dataset: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Generates fingerprints and labels for a given dataset.

        Parameters:
            dataset (pd.DataFrame): DataFrame with 'smiles' and 'activity' columns.

        Returns:
            tuple: Feature matrix (X) and labels (y).
        """
        dataset['fingerprint'] = dataset['smiles'].apply(self.generate_fingerprint)
        X = np.stack(dataset['fingerprint'].values)
        y = dataset['activity'].values
        return X, y

    def prepare_train_data(self):
        """
        Prepares the train split for model training.
        """
        self.X_train, self.y_train = self.prepare_data(self.train_split)
        print("Training data prepared.")

    def prepare_validation_data(self):
        """
        Prepares the validation split for model evaluation.
        """
        self.X_val, self.y_val = self.prepare_data(self.validation_split)
        print("Validation data prepared.")

    def prepare_test_data(self):
        """
        Prepares the test dataset.
        """
        if self.test is None:
            raise ValueError("Test data is not loaded.")
        self.test['fingerprint'] = self.test['smiles'].apply(self.generate_fingerprint)
        self.X_test = np.stack(self.test['fingerprint'].values)
        print("Test data prepared.")

    def train_model(self):
        """
        Trains a logistic regression model using the training data fingerprints.
        """
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
