from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.base import BaseEstimator, TransformerMixin

from .base import Pipeline

class SklearnPipelineWrapper(Pipeline):
    """
    Scikit-learn pipeline implementation.
    Wraps sklearn.pipeline.Pipeline with additional functionality.
    """
    
    def __init__(
        self,
        steps: List[Tuple[str, Union[str, Any], Optional[Dict[str, Any]]]]
    ):
        """
        Initialize the scikit-learn pipeline.
        
        Args:
            steps: List of (name, transformer, params) tuples
        """
        super().__init__(steps, framework="sklearn")
        
        # Create sklearn Pipeline
        self.pipeline = SklearnPipeline(self.steps)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'SklearnPipelineWrapper':
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformations to the data.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        # If the last step is an estimator, use predict
        if hasattr(self.pipeline.steps[-1][1], 'predict'):
            return self.predict(X)
        return self.pipeline.transform(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates.
        
        Args:
            X: Input data
            
        Returns:
            Probability estimates
        """
        if hasattr(self.pipeline.steps[-1][1], 'predict_proba'):
            return self.pipeline.predict_proba(X)
        raise AttributeError("Final estimator does not have predict_proba method")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the score on test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Score value
        """
        return self.pipeline.score(X, y)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the pipeline to disk using joblib.
        
        Args:
            path: Path to save the pipeline
        """
        path = Path(path)
        joblib.dump(self.pipeline, path)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the pipeline from disk using joblib.
        
        Args:
            path: Path to load the pipeline from
        """
        path = Path(path)
        self.pipeline = joblib.load(path)
        self.steps = self.pipeline.steps
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get feature names after transformation.
        
        Returns:
            List of feature names
        """
        if hasattr(self.pipeline, 'get_feature_names_out'):
            return self.pipeline.get_feature_names_out()
        raise AttributeError("Pipeline does not support feature names") 