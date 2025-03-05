from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import numpy as np
from pathlib import Path
import importlib
from abc import ABC, abstractmethod

class Pipeline(ABC):
    """
    Abstract base class for ML pipelines.
    Supports multiple ML frameworks and custom transformations.
    """
    
    def __init__(
        self,
        steps: List[Tuple[str, Union[str, Any], Optional[Dict[str, Any]]]],
        framework: str = "sklearn"
    ):
        """
        Initialize the pipeline.
        
        Args:
            steps: List of (name, transformer, params) tuples where:
                  - name is a string
                  - transformer is either a string (module path) or an object
                  - params is an optional dict of parameters
            framework: ML framework to use ("sklearn", "torch", "tensorflow")
        """
        self.steps = []
        self.framework = framework.lower()
        
        for name, transformer, params in steps:
            if isinstance(transformer, str):
                # Import the transformer class
                module_path, class_name = transformer.rsplit(".", 1)
                module = importlib.import_module(module_path)
                transformer_class = getattr(module, class_name)
                
                # Initialize with params if provided
                if params:
                    transformer = transformer_class(**params)
                else:
                    transformer = transformer_class()
            
            self.steps.append((name, transformer))
    
    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None) -> 'Pipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, X: Any) -> Any:
        """
        Apply transformations to the data.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        pass
    
    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Fit the pipeline and transform the data.
        
        Args:
            X: Input data
            y: Labels
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the pipeline to disk.
        
        Args:
            path: Path to save the pipeline
        """
        raise NotImplementedError("Saving not implemented for this pipeline")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the pipeline from disk.
        
        Args:
            path: Path to load the pipeline from
        """
        raise NotImplementedError("Loading not implemented for this pipeline")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get pipeline parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = {}
        for name, transformer in self.steps:
            if hasattr(transformer, 'get_params'):
                params[name] = transformer.get_params()
        return params
    
    def set_params(self, **params) -> 'Pipeline':
        """
        Set pipeline parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self
        """
        for name, transformer in self.steps:
            if name in params and hasattr(transformer, 'set_params'):
                transformer.set_params(**params[name])
        return self 