"""
EZFlow: A Simple, Modular ML Experimentation Pipeline
"""

__version__ = "0.1.0"

from ezflow.core.dataset.base import BaseDataset
from ezflow.core.pipeline.base import Pipeline
from ezflow.core.experiment.tracker import ExperimentTracker, ExperimentConfig

__all__ = [
    "BaseDataset",
    "Pipeline",
    "ExperimentTracker",
    "ExperimentConfig",
]
