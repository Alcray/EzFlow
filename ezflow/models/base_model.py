"""
Base model interface for the ezflow framework.

This module defines the base interface for all models in the framework.
"""

from abc import ABC, abstractmethod
import os
import logging
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all models in the ezflow framework.
    
    All model implementations should inherit from this class and implement
    its abstract methods.
    """
    
    @abstractmethod
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the model with parameters.
        
        Args:
            params (Dict[str, Any]): Model-specific parameters.
        """
        self.params = params
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Train the model on the given data.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            
        Returns:
            np.ndarray: Predictions.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probability predictions on new data (for classification models).
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            
        Returns:
            np.ndarray: Probability predictions.
            
        Note:
            This method should be implemented for classification models.
            For regression models, it can raise NotImplementedError.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path (str): Path to load the model from.
        """
        pass
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances if the model supports it.
        
        Returns:
            pd.DataFrame: Feature importances sorted by importance.
            
        Raises:
            NotImplementedError: If the model doesn't support feature importances.
        """
        raise NotImplementedError("Feature importance not implemented for this model type")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters.
        """
        return self.params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set the model parameters.
        
        Args:
            params (Dict[str, Any]): New model parameters.
        """
        self.params = params
        # Re-initialize the model with new parameters if needed
        # This may need to be implemented by the specific model class
    
    def __str__(self) -> str:
        """String representation of the model."""
        model_type = self.__class__.__name__
        return f"{model_type}(params={self.params}, fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__() 