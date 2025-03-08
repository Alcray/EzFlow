from ezflowx.models.base_model import BaseModel
from ezflowx.models.xgb_model import XGBoostModel
from ezflowx.models.model_factory import ModelFactory

# Register available models with the factory
ModelFactory.register_from_module('ezflowx.models')

