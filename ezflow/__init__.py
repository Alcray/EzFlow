"""
ezflow: A streamlined framework for running and tracking ML experiments for hackathons.
"""

__version__ = "0.1.0"

# Import core components for easier access
from ezflow.core.dataset import BaseDataset
from ezflow.core.model import Model
from ezflow.experiments.experiment import Experiment

# Import common dataset implementations
from ezflow.datasets.tabular import TabularDataset

# Import helper utilities
from ezflow.utils.helpers import (
    timed, 
    load_config, 
    save_config, 
    load_pickle, 
    save_pickle,
    setup_logger,
    create_submission_file,
    load_dataset_from_config
)

# Import visualization tools
from ezflow.visualization.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_metric_comparison,
    create_experiment_dashboard
)

# Set up a default logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Convenience function to get default logger
def get_logger(name: str = 'ezflow'):
    """Get a configured logger instance."""
    return logging.getLogger(name)
