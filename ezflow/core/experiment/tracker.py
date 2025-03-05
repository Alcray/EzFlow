from typing import Any, Dict, Optional, Union, List
import os
from pathlib import Path
import mlflow
import json
import datetime
from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    model_params: Dict[str, Any]
    dataset_params: Dict[str, Any]
    training_params: Dict[str, Any]
    tags: Optional[Dict[str, str]] = None

class ExperimentTracker:
    """
    Experiment tracking with MLflow integration.
    Handles logging of parameters, metrics, and artifacts.
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifacts_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking server
            artifacts_dir: Directory to store artifacts
        """
        self.experiment_name = experiment_name
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path("mlruns")
        
        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=str(self.artifacts_dir)
            )
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            
        self.current_run = None
    
    def start_run(
        self,
        config: ExperimentConfig,
        run_name: Optional[str] = None,
        nested: bool = False
    ) -> None:
        """
        Start a new experiment run.
        
        Args:
            config: Experiment configuration
            run_name: Optional name for the run
            nested: Whether this is a nested run
        """
        if not run_name:
            run_name = f"{config.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested
        )
        
        # Log configuration
        mlflow.log_params(config.model_params)
        mlflow.log_params(config.dataset_params)
        mlflow.log_params(config.training_params)
        
        if config.tags:
            mlflow.set_tags(config.tags)
            
        # Save full config as JSON artifact
        config_path = self.artifacts_dir / f"{run_name}_config.json"
        with config_path.open('w') as f:
            json.dump(asdict(config), f, indent=2)
        mlflow.log_artifact(str(config_path))
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """
        Log a local file or directory as an artifact.
        
        Args:
            local_path: Path to the file or directory
        """
        mlflow.log_artifact(str(local_path))
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        conda_env: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a model artifact.
        
        Args:
            model: The model object
            artifact_path: Destination path within the run's artifact directory
            conda_env: Optional Conda environment
        """
        mlflow.pyfunc.log_model(
            artifact_path,
            python_model=model,
            conda_env=conda_env
        )
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.
        
        Args:
            status: Run status ("FINISHED", "FAILED", etc.)
        """
        if self.current_run:
            mlflow.end_run(status=status)
            self.current_run = None
    
    def get_run_info(self) -> Dict[str, Any]:
        """
        Get information about the current run.
        
        Returns:
            Dictionary containing run information
        """
        if not self.current_run:
            raise ValueError("No active run")
            
        return {
            "run_id": self.current_run.info.run_id,
            "experiment_id": self.experiment_id,
            "status": self.current_run.info.status,
            "start_time": self.current_run.info.start_time,
            "artifact_uri": self.current_run.info.artifact_uri
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status) 