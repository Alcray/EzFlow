from ezflow.models.base_model import BaseModel
from ezflow.models.xgb_model import XGBoostModel

def get_model(model_type, **kwargs):
    """
    Factory function to get the appropriate model instance
    
    Args:
        model_type (str): Type of model to create
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        BaseModel: An instance of a model that inherits from BaseModel
    """
    if model_type.lower() == 'xgboost':
        # Initialize with empty params if none provided
        if 'params' not in kwargs:
            kwargs['params'] = {}
        return XGBoostModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")