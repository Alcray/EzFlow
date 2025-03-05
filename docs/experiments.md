# Experiment Module

The Experiment module provides tools for configuring, tracking, and comparing machine learning experiments.

## Table of Contents

- [Overview](#overview)
- [ExperimentConfig](#experimentconfig)
- [ExperimentTracker](#experimenttracker)
- [Integration with MLflow](#integration-with-mlflow)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The Experiment module is designed to help data scientists and researchers manage their machine learning experiments efficiently. Key features include:

- **Configuration Management**: Store and manage experiment parameters in a structured format
- **Metrics Tracking**: Log and compare metrics across different runs
- **Artifact Management**: Save and organize models, plots, and other artifacts
- **Run History**: Maintain a history of experiment runs for comparison
- **Reproducibility**: Enable reproducible experiments through consistent tracking

## ExperimentConfig

The `ExperimentConfig` class provides a structured way to define and manage experiment parameters.

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Name of the experiment |
| `model_params` | Dict | Model hyperparameters |
| `dataset_params` | Dict | Dataset configuration parameters |
| `training_params` | Dict | Training configuration parameters |
| `metadata` | Dict | Additional metadata about the experiment |

### Usage Example

```python
from ezflow.core.experiment import ExperimentConfig

# Create a configuration for an experiment
config = ExperimentConfig(
    name="random_forest_classifier",
    model_params={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    dataset_params={
        "val_size": 0.2,
        "stratify": True
    },
    training_params={
        "cv_folds": 5,
        "scoring": "accuracy"
    },
    metadata={
        "author": "John Doe",
        "description": "Random forest classifier for Iris dataset"
    }
)

# Convert to dictionary for serialization
config_dict = config.to_dict()

# Save to file
config.save("./experiments/rf_config.json")

# Load from file
loaded_config = ExperimentConfig.load("./experiments/rf_config.json")
```

### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `to_dict` | - | Dict | Convert configuration to a dictionary |
| `from_dict` | `config_dict: Dict` | ExperimentConfig | Create configuration from a dictionary |
| `save` | `path: str/Path` | None | Save configuration to a file |
| `load` | `path: str/Path` | ExperimentConfig | Load configuration from a file |

## ExperimentTracker

The `ExperimentTracker` class provides functionality for tracking experiments, including metrics, parameters, and artifacts.

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `experiment_name` | str | Name of the experiment |
| `artifacts_dir` | str/Path | Directory for storing artifacts |
| `backend` | str | Backend tracking system ("mlflow", "tensorboard", etc.) |
| `active_run` | Any | Current active run |

### Usage Example

```python
from ezflow.core.experiment import ExperimentTracker, ExperimentConfig
from ezflow.core.pipeline import SklearnPipelineWrapper
from ezflow.core.dataset import IrisDataset
from sklearn.metrics import accuracy_score

# Initialize dataset
dataset = IrisDataset(data_dir="./data")
dataset.load_data()
dataset.split_data(val_size=0.2, random_state=42)

# Get features and labels
X_train = dataset.get_features(dataset.train_data)
y_train = dataset.get_labels(dataset.train_data)
X_val = dataset.get_features(dataset.val_data)
y_val = dataset.get_labels(dataset.val_data)

# Create pipeline
pipeline = SklearnPipelineWrapper([
    ("scaler", "sklearn.preprocessing.StandardScaler", {}),
    ("classifier", "sklearn.ensemble.RandomForestClassifier", {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    })
])

# Create experiment config
config = ExperimentConfig(
    name="iris_classification",
    model_params={"n_estimators": 100, "max_depth": 5},
    dataset_params={"val_size": 0.2},
    training_params={"random_state": 42}
)

# Initialize experiment tracker
tracker = ExperimentTracker(
    experiment_name="iris_example",
    artifacts_dir="./experiments"
)

# Track experiment
with tracker:
    # Start run with config
    tracker.start_run(config)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    # Log metrics
    tracker.log_metrics({
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    })
    
    # Save model as artifact
    pipeline.save("./models/iris_rf.joblib")
    tracker.log_artifact("./models/iris_rf.joblib")
```

### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `start_run` | `config: ExperimentConfig` | None | Start a new tracked run |
| `end_run` | - | None | End the current run |
| `log_metric` | `name: str`, `value: float` | None | Log a single metric |
| `log_metrics` | `metrics: Dict[str, float]` | None | Log multiple metrics |
| `log_parameter` | `name: str`, `value: Any` | None | Log a single parameter |
| `log_parameters` | `params: Dict[str, Any]` | None | Log multiple parameters |
| `log_artifact` | `artifact_path: str/Path` | None | Log a file as an artifact |
| `get_artifact` | `artifact_path: str/Path` | str/Path | Retrieve an artifact |

## Integration with MLflow

EZFlow's experiment tracking is built on top of MLflow, a popular open-source platform for managing the ML lifecycle.

### MLflow Backend

By default, `ExperimentTracker` uses MLflow as its backend tracking system. This provides several advantages:

- **Web UI**: Access a web-based dashboard for experiment comparison
- **Artifact Storage**: Efficient storage and retrieval of models and other files
- **Run Comparison**: Easy comparison of different experiment runs
- **Metadata Search**: Search and filter experiments based on parameters

### Tracking Server Configuration

You can configure the MLflow tracking server in several ways:

1. **Local Tracking (Default)**:
   ```python
   # Uses ./mlruns directory by default
   tracker = ExperimentTracker("my_experiment")
   ```

2. **Custom Local Directory**:
   ```python
   # Specify a custom directory for MLflow data
   tracker = ExperimentTracker(
       "my_experiment", 
       backend_config={"tracking_uri": "file:/path/to/mlruns"}
   )
   ```

3. **Remote Tracking Server**:
   ```python
   # Use a remote MLflow tracking server
   tracker = ExperimentTracker(
       "my_experiment", 
       backend_config={"tracking_uri": "http://my-mlflow-server:5000"}
   )
   ```

### Viewing Experiment Results

To view experiment results using the MLflow UI:

```bash
# Start the MLflow UI
mlflow ui --backend-store-uri ./experiments/mlruns

# If using a remote server
mlflow ui --backend-store-uri http://my-mlflow-server:5000
```

This will start a web server (default: http://localhost:5000) where you can view and compare experiments.

## Best Practices

For optimal results when tracking experiments:

1. **Consistent Naming**: Use consistent naming conventions for experiments, runs, and metrics
2. **Parameter Tracking**: Track all relevant hyperparameters for reproducibility
3. **Granular Metrics**: Log metrics at appropriate intervals (not just final results)
4. **Artifact Management**: Save models and other important artifacts with clear versioning
5. **Contextual Metadata**: Include contextual information like dataset versions, environment details
6. **Run Tags**: Use tags to categorize and filter experiment runs
7. **Experiment Organization**: Group related experiments logically

## API Reference

### ExperimentConfig

```python
class ExperimentConfig:
    """Configuration for machine learning experiments."""
    
    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any] = None,
        dataset_params: Dict[str, Any] = None,
        training_params: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize experiment configuration.
        
        Parameters
        ----------
        name : str
            Name of the experiment
        model_params : dict, optional
            Model hyperparameters
        dataset_params : dict, optional
            Dataset configuration parameters
        training_params : dict, optional
            Training configuration parameters
        metadata : dict, optional
            Additional metadata about the experiment
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the configuration
        """
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create configuration from a dictionary.
        
        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration values
            
        Returns
        -------
        ExperimentConfig
            Configuration object
        """
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a file.
        
        Parameters
        ----------
        path : str or Path
            Path to save the configuration to
        """
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """
        Load configuration from a file.
        
        Parameters
        ----------
        path : str or Path
            Path to load the configuration from
            
        Returns
        -------
        ExperimentConfig
            Configuration object
        """
```

### ExperimentTracker

```python
class ExperimentTracker:
    """Tracker for machine learning experiments."""
    
    def __init__(
        self,
        experiment_name: str,
        artifacts_dir: Union[str, Path] = None,
        backend: str = "mlflow",
        backend_config: Dict[str, Any] = None
    ):
        """
        Initialize experiment tracker.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment
        artifacts_dir : str or Path, optional
            Directory for storing artifacts
        backend : str
            Backend tracking system ("mlflow", "tensorboard", etc.)
        backend_config : dict, optional
            Configuration for the backend tracking system
        """
    
    def __enter__(self) -> 'ExperimentTracker':
        """
        Enter context manager.
        
        Returns
        -------
        ExperimentTracker
            Self, for context manager usage
        """
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager and end run if active.
        
        Parameters
        ----------
        exc_type : type
            Exception type, if any
        exc_val : Exception
            Exception value, if any
        exc_tb : traceback
            Exception traceback, if any
        """
    
    def start_run(self, config: ExperimentConfig = None) -> None:
        """
        Start a new tracked run.
        
        Parameters
        ----------
        config : ExperimentConfig, optional
            Configuration for the run
        """
    
    def end_run(self) -> None:
        """End the current run."""
    
    def log_metric(self, name: str, value: float) -> None:
        """
        Log a single metric.
        
        Parameters
        ----------
        name : str
            Name of the metric
        value : float
            Value of the metric
        """
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log multiple metrics.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of metric names and values
        """
    
    def log_parameter(self, name: str, value: Any) -> None:
        """
        Log a single parameter.
        
        Parameters
        ----------
        name : str
            Name of the parameter
        value : any
            Value of the parameter
        """
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary of parameter names and values
        """
    
    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """
        Log a file as an artifact.
        
        Parameters
        ----------
        artifact_path : str or Path
            Path to the file to log as an artifact
        """
    
    def get_artifact(self, artifact_path: Union[str, Path]) -> Union[str, Path]:
        """
        Retrieve an artifact.
        
        Parameters
        ----------
        artifact_path : str or Path
            Path to the artifact
            
        Returns
        -------
        str or Path
            Path to the retrieved artifact
        """
``` 