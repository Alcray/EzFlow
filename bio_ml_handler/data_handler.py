import os
import kagglehub
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

class BioMLDataHandler:
    def __init__(self, bio_ml_path: str = None):
        """
        Initializes the data handler and optionally loads the dataset if the path is provided.

        Parameters:
            bio_ml_path (str): Path to the bio-ml data directory, if already downloaded.
        """
        self.bio_ml_path = bio_ml_path
        self.train = None
        self.test = None
        self.sample_submission = None
        self.model = None

        # Load data if path is provided
        if bio_ml_path:
            self.load_data(bio_ml_path)

    def download_data(self, competition_name: str = 'bio-ml', destination: str = './bio_ml_data'):
        """
        Downloads data from Kaggle using kagglehub, and sets the path to the downloaded data.

        Parameters:
            competition_name (str): Kaggle competition name.
            destination (str): Destination directory to download the data.

        Returns:
            str: Path to the downloaded data directory.
        """
        # Authenticate and download the data
        kagglehub.login()
        bio_ml_path = kagglehub.competition_download(competition_name, path=destination)
        
        print('Data source import complete.')
        print(f'Data downloaded to: {bio_ml_path}')

        # Set the data path and load the data
        self.bio_ml_path = bio_ml_path
        self.load_data(bio_ml_path)
        
        return bio_ml_path

    def load_data(self, path: str):
        """
        Loads data from the specified path into the handler.

        Parameters:
            path (str): Path to the directory containing train.csv, test.csv, and sample_submission.csv.
        """
        self.train = pd.read_csv(os.path.join(path, 'train.csv'))
        self.test = pd.read_csv(os.path.join(path, 'test.csv'))
        self.sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
        print("Data loaded successfully.")

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

    def prepare_train_data(self):
        """
        Generates fingerprints for the training dataset.
        """
        if self.train is None:
            raise ValueError("Training data is not loaded.")
        self.train['fingerprint'] = self.train['smiles'].apply(self.generate_fingerprint)
        self.X_train = np.stack(self.train['fingerprint'].values)
        self.y_train = self.train['activity'].values

    def prepare_test_data(self):
        """
        Generates fingerprints for the test dataset.
        """
        if self.test is None:
            raise ValueError("Test data is not loaded.")
        self.test['fingerprint'] = self.test['smiles'].apply(self.generate_fingerprint)
        self.X_test = np.stack(self.test['fingerprint'].values)

    def train_model(self):
        """
        Trains a logistic regression model using the training data fingerprints.
        """
        self.model = LogisticRegression(solver='liblinear')
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self) -> float:
        """
        Evaluates the model using average precision score on training data.

        Returns:
            float: The average precision score.
        """
        train_preds = self.predict(self.X_train)
        return average_precision_score(self.y_train, train_preds)

    def predict(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Predicts activity probabilities for a given set of fingerprints.

        Parameters:
            fingerprints (np.ndarray): Feature matrix for predictions.

        Returns:
            np.ndarray: Array of probabilities for the positive class.
        """
        return self.model.predict_proba(fingerprints)[:, 1]

    def generate_submission(self, filename: str = 'submission.csv'):
        """
        Generates a submission file for Kaggle.

        Parameters:
            filename (str): Name of the output submission file. Default is 'submission.csv'.
        """
        test_preds = self.predict(self.X_test)
        self.test['activity'] = test_preds
        submission = self.test[['id', 'activity']]
        submission.to_csv(filename, index=False)
