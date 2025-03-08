# Version information
__version__ = "0.1.0" 

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import core components to make them available via the main module
from ezflow.models import ModelFactory
from ezflow.models.base_model import BaseModel

# Make key classes and functions available at the top level
__all__ = ['ModelFactory', 'BaseModel']

# Check for MLflow availability
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not found. Experiment tracking will be limited.") 