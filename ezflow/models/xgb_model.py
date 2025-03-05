"""
XGBoost model implementation for the ezflow framework.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
import joblib
import xgboost as xgb

from ezflow.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """
    XGBoost model implementation for the ezflow framework.
    
    This class provides a wrapper around the XGBoost model with
    support for classification and regression.
    """
    
    def __init__(self, params: Dict[str, Any], model_type: str = 'classifier'):
        """
        Initialize the XGBoost model.
        
        Args:
            params (Dict[str, Any]): XGBoost parameters.
            model_type (str): Model type ('classifier' or 'regressor').
            
        Raises:
            ValueError: If model_type is not 'classifier' or 'regressor'.
        """
        self.params = params
        self.model_type = model_type.lower()
        self.is_fitted = False
        
        # Default parameters for XGBoost
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic' if self.model_type == 'classifier' else 'reg:squarederror',
            'random_state': 42
        }
        
        # Update default params with provided params
        self.params = {**default_params, **self.params}
        
        # Initialize model
        if self.model_type == 'classifier':
            self.model = xgb.XGBClassifier(**self.params)
        elif self.model_type == 'regressor':
            self.model = xgb.XGBRegressor(**self.params)
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'classifier' or 'regressor'")
        
        logger.info(f"Initialized XGBoostModel of type {self.model_type}")
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
              eval_set: Optional[list] = None, early_stopping_rounds: Optional[int] = None) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
            eval_set (Optional[list]): Validation set for early stopping.
            early_stopping_rounds (Optional[int]): Number of rounds for early stopping.
        """
        logger.info("Training XGBoost model...")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = None
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Set feature names if available
        if feature_names:
            self.model.feature_names = feature_names
        
        # Train the model
        fit_params = {}
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        
        logger.info(f"XGBoost model trained successfully with {self.model.n_estimators} trees")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the XGBoost model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            
        Returns:
            np.ndarray: Predictions.
            
        Raises:
            ValueError: If the model is not trained.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probability predictions with the XGBoost classifier.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            
        Returns:
            np.ndarray: Probability predictions.
            
        Raises:
            ValueError: If the model is not trained or not a classifier.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
        
        if self.model_type != 'classifier':
            raise ValueError("predict_proba is only available for classifiers")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """
        Save the XGBoost model to disk.
        
        Args:
            path (str): Path to save the model.
        """
        if not self.is_fitted:
            logger.warning("Saving an untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        joblib.dump(self, path)
        logger.info(f"XGBoost model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the XGBoost model from disk.
        
        Args:
            path (str): Path to load the model from.
            
        Raises:
            FileNotFoundError: If the model file doesn't exist.
            ValueError: If the loaded object is not an XGBoostModel.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        loaded_model = joblib.load(path)
        
        if not isinstance(loaded_model, XGBoostModel):
            raise ValueError(f"Loaded object is not an XGBoostModel")
        
        # Update attributes
        self.model = loaded_model.model
        self.params = loaded_model.params
        self.model_type = loaded_model.model_type
        self.is_fitted = loaded_model.is_fitted
        
        logger.info(f"XGBoost model loaded from {path}")
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances from the XGBoost model.
        
        Returns:
            pd.DataFrame: Feature importances sorted by importance.
            
        Raises:
            ValueError: If the model is not trained.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
        
        # Get feature importances
        feature_importance = self.model.feature_importances_
        
        # Get feature names
        feature_names = getattr(self.model, 'feature_names', None)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return importance_df 