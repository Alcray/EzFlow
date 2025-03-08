from ezflow.models.base_model import BaseModel
from ezflow.models.xgb_model import XGBoostModel
from ezflow.models.model_factory import ModelFactory

# Register available models with the factory
ModelFactory.register_from_module('ezflow.models')

def get_model(model_type, **kwargs):
    """
    Factory function to get the appropriate model instance
    
    Args:
        model_type (str): Type of model to create
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        BaseModel: An instance of a model that inherits from BaseModel
    """
    # Use ModelFactory to create model
    return ModelFactory.create(model_type, **kwargs)
