"""
ModelFactory for the ezflow framework.

This module provides a robust factory for creating model instances.
"""

import logging
import importlib
from typing import Dict, Any, Type, List, Optional, Union, Callable

from ezflow.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating model instances.
    
    This factory supports:
    1. Registry of model types
    2. Dynamic model instantiation
    3. Parameter validation
    4. Default parameter handling
    """
    
    _registry = {}  # Map of model_type: model_class
    _problem_types = ['classification', 'regression']  # Supported problem types
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class with the factory.
        
        Args:
            name (str): Name to register the model under
            model_class (Type[BaseModel]): Class to register
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class must inherit from BaseModel, got {model_class}")
            
        logger.debug(f"Registering model type: {name}")
        cls._registry[name.lower()] = model_class
    
    @classmethod
    def register_from_module(cls, module_path: str) -> None:
        """
        Register all model classes from a module.
        
        Args:
            module_path (str): Import path of the module
        """
        try:
            module = importlib.import_module(module_path)
            
            # Find all BaseModel subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseModel) and 
                    attr is not BaseModel):
                    # Register with lowercase class name
                    name = attr.__name__.lower()
                    if name.endswith('model'):
                        name = name[:-5]  # Remove 'model' suffix if present
                    cls.register(name, attr)
                    
            logger.info(f"Registered models from module: {module_path}")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to register models from {module_path}: {str(e)}")
    
    @classmethod
    def create(cls, model_type: str, problem_type: str = 'classification',
              params: Dict[str, Any] = None, **kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type (str): Type of model to create
            problem_type (str): Problem type ('classification' or 'regression')
            params (Dict[str, Any]): Model parameters
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            BaseModel: Initialized model instance
            
        Raises:
            ValueError: If model_type is not registered or problem_type is invalid
        """
        model_type = model_type.lower()
        
        # Check if model type exists
        if model_type not in cls._registry:
            available_models = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")
            
        # Check problem type
        if problem_type not in cls._problem_types:
            valid_types = ", ".join(cls._problem_types)
            raise ValueError(f"Invalid problem_type: {problem_type}. Must be one of: {valid_types}")
        
        # Get model class
        model_class = cls._registry[model_type]
        
        # Initialize parameters
        model_params = params or {}
        
        # Add problem type to kwargs
        kwargs['problem_type'] = problem_type
        
        try:
            # Create model instance
            logger.info(f"Creating {model_type} model for {problem_type}")
            model = model_class(params=model_params, **kwargs)
            return model
        except Exception as e:
            logger.error(f"Failed to create {model_type} model: {str(e)}")
            raise
    
    @classmethod
    def available_models(cls) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List[str]: List of registered model types
        """
        return list(cls._registry.keys())
    
    @classmethod
    def validate_params(cls, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for a specific model type.
        
        Args:
            model_type (str): Type of model
            params (Dict[str, Any]): Parameters to validate
            
        Returns:
            Dict[str, Any]: Validated parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        model_type = model_type.lower()
        
        # Check if model type exists
        if model_type not in cls._registry:
            available_models = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")
            
        # Get model class for param validation (if it has validation method)
        model_class = cls._registry[model_type]
        
        # Check if model class has validate_params method
        if hasattr(model_class, 'validate_params') and callable(getattr(model_class, 'validate_params')):
            return model_class.validate_params(params)
        
        # Default validation (just return the params)
        return params
    
    @classmethod
    def get_default_params(cls, model_type: str, problem_type: str = 'classification') -> Dict[str, Any]:
        """
        Get default parameters for a model type.
        
        Args:
            model_type (str): Type of model
            problem_type (str): Problem type ('classification' or 'regression')
            
        Returns:
            Dict[str, Any]: Default parameters for the model
        """
        model_type = model_type.lower()
        
        # Check if model type exists
        if model_type not in cls._registry:
            available_models = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")
        
        # Get model class
        model_class = cls._registry[model_type]
        
        # Check if model class has get_default_params method
        if hasattr(model_class, 'get_default_params') and callable(getattr(model_class, 'get_default_params')):
            return model_class.get_default_params(problem_type)
        
        # Return empty dict if no default params method
        return {}
    
    @classmethod
    def get_param_search_space(cls, model_type: str, problem_type: str = 'classification') -> Dict[str, Any]:
        """
        Get default hyperparameter search space for a model type.
        
        Args:
            model_type (str): Type of model
            problem_type (str): Problem type ('classification' or 'regression')
            
        Returns:
            Dict[str, Any]: Default hyperparameter search space
        """
        model_type = model_type.lower()
        
        # Check if model type exists
        if model_type not in cls._registry:
            available_models = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")
        
        # Get model class
        model_class = cls._registry[model_type]
        
        # Check if model class has get_param_search_space method
        if hasattr(model_class, 'get_param_search_space') and callable(getattr(model_class, 'get_param_search_space')):
            return model_class.get_param_search_space(problem_type)
        
        # Return empty dict if no search space method
        return {} 