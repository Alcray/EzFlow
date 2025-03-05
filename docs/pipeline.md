# Pipeline Module

The Pipeline module provides a unified interface for creating, training, and deploying machine learning models with different backend frameworks.

## Table of Contents

- [Overview](#overview)
- [Base Pipeline](#base-pipeline)
- [Pipeline Implementations](#pipeline-implementations)
  - [SklearnPipelineWrapper](#sklearnpipelinewrapper)
  - [TorchPipelineWrapper](#torchpipelinewrapper)
- [Custom Pipelines](#custom-pipelines)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The Pipeline module is a core component of EZFlow, designed to abstract away the differences between machine learning frameworks and provide a consistent interface for model development. Key features include:

- Unified API for different ML frameworks
- Consistent methods for training, prediction, and evaluation
- Model serialization and deserialization
- Support for complex pipelines with preprocessing steps
- Extensibility for custom model implementations

## Base Pipeline

The `Pipeline` abstract base class defines the interface that all pipeline implementations must follow.

### Key Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `fit` | `X: np.ndarray`, `y: np.ndarray` | None | Train the model on the provided data |
| `predict` | `X: np.ndarray` | np.ndarray | Generate predictions for the input data |
| `score` | `X: np.ndarray`, `y: np.ndarray` | float | Evaluate the model on the provided data |
| `save` | `path: str/Path` | None | Save the model to disk |
| `load` | `path: str/Path` | None | Load the model from disk |

### Usage Example

```python
from ezflow.core.pipeline import Pipeline

# Create a pipeline using a concrete implementation
pipeline = SomePipelineImplementation(params)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
score = pipeline.score(X_test, y_test)

# Save and load the model
pipeline.save("./models/my_model.pkl")
pipeline.load("./models/my_model.pkl")
```

## Pipeline Implementations

EZFlow provides several concrete implementations of the `Pipeline` interface for popular machine learning frameworks.

### SklearnPipelineWrapper

The `SklearnPipelineWrapper` class provides an implementation of the pipeline interface for scikit-learn models.

#### Features

- Support for scikit-learn's pipeline API
- Dynamic creation of pipeline steps from string references
- Flexible parameter specification
- Consistent serialization/deserialization

#### Initialization

```python
from ezflow.core.pipeline import SklearnPipelineWrapper

# Create a pipeline with preprocessing and a model
pipeline = SklearnPipelineWrapper([
    ("scaler", "sklearn.preprocessing.StandardScaler", {}),
    ("pca", "sklearn.decomposition.PCA", {"n_components": 10}),
    ("classifier", "sklearn.ensemble.RandomForestClassifier", {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })
])
```

#### Parameters

- `steps`: A list of tuples, each containing:
  - Step name (string)
  - Fully qualified class path (string) or actual class
  - Parameters dictionary for the step

### TorchPipelineWrapper

The `TorchPipelineWrapper` class provides an implementation for PyTorch models.

#### Features

- Support for PyTorch models and training loops
- Configurable training parameters
- Automatic device selection (CPU/GPU)
- Proper model serialization/deserialization

#### Initialization

```python
import torch.nn as nn
from ezflow.core.pipeline import TorchPipelineWrapper

# Define a PyTorch model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create a pipeline with the model
model = SimpleNN(input_dim=10, hidden_dim=50, output_dim=3)
pipeline = TorchPipelineWrapper(
    model=model,
    loss_fn="torch.nn.CrossEntropyLoss",
    optimizer="torch.optim.Adam",
    optimizer_params={"lr": 0.001},
    device="auto"  # Automatically select CPU or GPU
)
```

#### Parameters

- `model`: PyTorch nn.Module or string reference to a model class
- `loss_fn`: Loss function (string reference or actual function)
- `optimizer`: Optimizer class (string reference or actual class)
- `optimizer_params`: Dictionary of optimizer parameters
- `device`: "cpu", "cuda", or "auto" for automatic selection

## Custom Pipelines

Creating a custom pipeline involves inheriting from the `Pipeline` base class and implementing the required methods.

### Example: Custom TensorFlow Pipeline

```python
import numpy as np
import tensorflow as tf
from ezflow.core.pipeline import Pipeline

class TensorFlowPipelineWrapper(Pipeline):
    def __init__(self, model=None, model_config=None, compile_params=None):
        super().__init__()
        self.model_config = model_config or {}
        self.compile_params = compile_params or {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"]
        }
        
        # Create or use provided model
        if model is not None:
            if isinstance(model, str):
                # Import the model class dynamically
                parts = model.split(".")
                module_name = ".".join(parts[:-1])
                class_name = parts[-1]
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                self.model = model_class(**self.model_config)
            else:
                self.model = model
        else:
            raise ValueError("Model must be provided")
        
        # Compile the model
        self.model.compile(**self.compile_params)
    
    def fit(self, X, y, **kwargs):
        """Train the model on the provided data."""
        default_params = {
            "epochs": 10,
            "batch_size": 32,
            "validation_split": 0.1,
            "verbose": 1
        }
        # Update defaults with any provided kwargs
        params = {**default_params, **kwargs}
        history = self.model.fit(X, y, **params)
        return history
    
    def predict(self, X):
        """Generate predictions for the input data."""
        return self.model.predict(X)
    
    def score(self, X, y):
        """Evaluate the model on the provided data."""
        scores = self.model.evaluate(X, y, verbose=0)
        if isinstance(scores, list):
            # Return the main metric (usually loss is first, acc is second)
            return scores[1] if len(scores) > 1 else scores[0]
        return scores
    
    def save(self, path):
        """Save the model to disk."""
        self.model.save(path)
    
    def load(self, path):
        """Load the model from disk."""
        self.model = tf.keras.models.load_model(path)
        return self
```

### Using the Custom Pipeline

```python
# Create a TensorFlow pipeline
pipeline = TensorFlowPipelineWrapper(
    model="tensorflow.keras.Sequential",
    model_config={
        "layers": [
            tf.keras.layers.Dense(128, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax")
        ]
    },
    compile_params={
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"]
    }
)

# Train and use like any other pipeline
pipeline.fit(X_train, y_train, epochs=20, batch_size=64)
predictions = pipeline.predict(X_test)
```

## Best Practices

For optimal results when working with pipelines:

1. **Consistent Interfaces**: Ensure all pipeline implementations follow the standard interface
2. **Error Handling**: Add informative error messages for common issues
3. **Resource Management**: Clean up resources properly, especially with GPU-based models
4. **Serialization**: Test model saving and loading thoroughly
5. **Performance Monitoring**: Include ways to monitor training performance
6. **Parameter Validation**: Validate parameters early to catch configuration errors
7. **Documentation**: Document expected input formats and model-specific behaviors

## API Reference

### Pipeline (Abstract Base Class)

```python
class Pipeline:
    """Abstract base class for all pipeline implementations in EZFlow."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.pipeline = None
        self.pipeline_config = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """
        Train the model on the provided data.
        
        Parameters
        ----------
        X : numpy.ndarray
            Features for training
        y : numpy.ndarray
            Target values for training
        **kwargs : dict
            Additional arguments for the specific implementation
            
        Returns
        -------
        Any
            Implementation-specific return value, often training history
        """
        raise NotImplementedError
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        Parameters
        ----------
        X : numpy.ndarray
            Features to generate predictions for
        **kwargs : dict
            Additional arguments for the specific implementation
            
        Returns
        -------
        numpy.ndarray
            Predictions
        """
        raise NotImplementedError
    
    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Evaluate the model on the provided data.
        
        Parameters
        ----------
        X : numpy.ndarray
            Features for evaluation
        y : numpy.ndarray
            Target values for evaluation
        **kwargs : dict
            Additional arguments for the specific implementation
            
        Returns
        -------
        float
            Score value (higher is better)
        """
        raise NotImplementedError
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Parameters
        ----------
        path : str or Path
            Path to save the model to
        """
        raise NotImplementedError
    
    def load(self, path: Union[str, Path]) -> 'Pipeline':
        """
        Load the model from disk.
        
        Parameters
        ----------
        path : str or Path
            Path to load the model from
            
        Returns
        -------
        Pipeline
            Self, for method chaining
        """
        raise NotImplementedError
```

### SklearnPipelineWrapper

```python
class SklearnPipelineWrapper(Pipeline):
    """Pipeline implementation for scikit-learn models."""
    
    def __init__(self, steps: List[Tuple[str, Union[str, Any], Dict[str, Any]]]):
        """
        Initialize the scikit-learn pipeline.
        
        Parameters
        ----------
        steps : list of tuples
            List of (name, estimator, params) tuples defining the pipeline steps.
            Estimator can be a string (module path) or an actual estimator class.
        """
```

### TorchPipelineWrapper

```python
class TorchPipelineWrapper(Pipeline):
    """Pipeline implementation for PyTorch models."""
    
    def __init__(
        self,
        model: Union[str, nn.Module],
        loss_fn: Union[str, Callable],
        optimizer: Union[str, Type[optim.Optimizer]],
        optimizer_params: Dict[str, Any] = None,
        device: str = "auto"
    ):
        """
        Initialize the PyTorch pipeline.
        
        Parameters
        ----------
        model : str or torch.nn.Module
            PyTorch model or string reference to a model class
        loss_fn : str or callable
            Loss function or string reference
        optimizer : str or torch.optim.Optimizer
            Optimizer class or string reference
        optimizer_params : dict
            Dictionary of optimizer parameters
        device : str
            "cpu", "cuda", or "auto" for automatic selection
        """
``` 