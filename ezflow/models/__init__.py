from ezflow.models.base_model import BaseModel
from ezflow.models.xgb_model import XGBoostModel
from ezflow.models.model_factory import ModelFactory

# Register available models with the factory
ModelFactory.register_from_module('ezflow.models')

