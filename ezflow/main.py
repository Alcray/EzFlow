"""
Core functionality for the ezflow framework, simplified for notebook environments.
This module initializes the framework and provides basic utility functions.
"""

import logging
import os
from typing import List, Optional

from ezflow.models import ModelFactory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ezflow")

# Check for MLflow availability
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.debug("MLflow not installed. Install with: pip install mlflow")

def list_available_models() -> List[str]:
    """
    List all available models registered with ModelFactory.
    
    Returns:
        List[str]: List of available model names
    """
    return ModelFactory.available_models()

def configure_mlflow(tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None) -> bool:
    """
    Configure MLflow for experiment tracking.
    
    Args:
        tracking_uri (Optional[str]): MLflow tracking URI
        experiment_name (Optional[str]): Experiment name
        
    Returns:
        bool: True if MLflow is available and configured, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Install with: pip install mlflow")
        return False
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    return True

# Version information
__version__ = "0.1.0" 