"""
XGBoost model implementation for the ezflow framework.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import joblib
import xgboost as xgb

from ezflow.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """
    XGBoost model for tabular data.
    Handles data from manifest.jsonl where each key-value pair is treated as a feature,
    except for the target key which is used as the label.
    """
    
    def __init__(self, params: Dict[str, Any], target_key: str = "target"):
        """
        Initialize XGBoost model.
        
        Args:
            params (Dict[str, Any]): XGBoost parameters
            target_key (str): Key in manifest.jsonl that contains the target value
        """
        super().__init__(params)
        self.target_key = target_key
        
        # Default parameters for XGBoost
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        
        # Update default params with provided params
        self.params = {**default_params, **params}
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names = None
    
    def load_data(self, manifest_path: str) -> tuple:
        """
        Load and prepare data from manifest.jsonl.
        
        Args:
            manifest_path (str): Path to manifest.jsonl file
            
        Returns:
            tuple: (X, y) where X is features DataFrame and y is target Series
        """
        # Read manifest file
        data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Separate features and target
        if self.target_key not in df.columns:
            raise ValueError(f"Target key '{self.target_key}' not found in manifest")
        
        y = df[self.target_key]
        X = df.drop(columns=[self.target_key])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, manifest_path: str, eval_manifest: Optional[str] = None) -> None:
        """
        Train the model on data from manifest file.
        
        Args:
            manifest_path (str): Path to training manifest.jsonl
            eval_manifest (Optional[str]): Path to evaluation manifest.jsonl
        """
        logger.info("Loading training data...")
        X_train, y_train = self.load_data(manifest_path)
        
        eval_set = None
        if eval_manifest:
            logger.info("Loading evaluation data...")
            X_eval, y_eval = self.load_data(eval_manifest)
            eval_set = [(X_eval, y_eval)]
        
        logger.info("Training XGBoost model...")
        
        # Check if we need to modify the objective based on the target data
        if len(np.unique(y_train)) > 2:
            # For multiclass classification
            self.params['objective'] = 'multi:softprob'
            self.params['num_class'] = len(np.unique(y_train))
            
        # Reinitialize the model with updated params
        self.model = xgb.XGBClassifier(**self.params)
        
        # Fit the model
        if eval_set:
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        logger.info("Training completed")
    
    def predict(self, manifest_path: str) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            manifest_path (str): Path to manifest.jsonl with features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
        
        # Load and prepare data
        X, _ = self.load_data(manifest_path)
        
        # Ensure all feature columns are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in prediction data: {missing_features}")
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_proba(self, manifest_path: str) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            manifest_path (str): Path to manifest.jsonl with features
            
        Returns:
            np.ndarray: Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
        
        # Load and prepare data
        X, _ = self.load_data(manifest_path)
        
        # Ensure all feature columns are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in prediction data: {missing_features}")
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if not self.is_fitted:
            logger.warning("Saving an untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'target_key': self.target_key,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model and metadata
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.target_key = model_data['target_key']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {path}")
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances.
        
        Returns:
            pd.DataFrame: Feature importances sorted by importance
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
        
        # Get feature importances
        importance = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return importance_df 