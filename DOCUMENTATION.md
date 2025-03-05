# ezflow Framework Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Adding Custom Models](#adding-custom-models)
4. [Command Line Interface](#command-line-interface)
5. [Configuration](#configuration)
6. [Examples](#examples)

## Overview

ezflow is a modular machine learning framework designed for rapid prototyping and hackathons. The framework consists of several core components that work together to provide a complete ML pipeline.

## Core Components

### 1. DataLoader (`ezflow/src/data_loader.py`)
The DataLoader handles all data ingestion and preprocessing tasks:
- **Key Functions**:
  - `load_data(mode)`: Loads data from various file formats (CSV, Excel, JSON, Parquet)
  - `clean_data(df)`: Performs basic data cleaning operations
  - `split_features_target(df, target_col)`: Separates features and target variables
  - `train_test_split(df, target_col)`: Creates train-test splits
  - `merge_data(df1, df2, on)`: Merges multiple datasets

Example usage:
```python
from ezflow.src.data_loader import DataLoader

loader = DataLoader(config)
train_data = loader.load_data("train")
X_train, X_test, y_train, y_test = loader.train_test_split(train_data, "target")
```

### 2. FeatureEngineer (`ezflow/src/feature_engineering.py`)
Handles all feature engineering and transformation tasks:
- **Key Functions**:
  - `create_features(df)`: Creates new features from existing ones
  - `setup_preprocessing(numeric_features, categorical_features)`: Sets up preprocessing pipelines
  - `fit_transform(df)`: Fits and transforms the data
  - `transform(df)`: Applies learned transformations to new data
  - `encode_categorical(df, columns)`: Handles categorical encoding
  - `scale_features(df, columns)`: Scales numeric features

Example usage:
```python
from ezflow.src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(config)
transformed_data = engineer.fit_transform(train_data)
```

### 3. ModelTrainer (`ezflow/src/trainer.py`)
Manages model training and basic evaluation:
- **Key Functions**:
  - `train(X_train, y_train)`: Trains the model
  - `predict(X)`: Makes predictions
  - `evaluate(X, y)`: Evaluates model performance
  - `cross_validate(X, y)`: Performs cross-validation
  - `save_model(path)`: Saves trained model
  - `load_model(path)`: Loads trained model

Example usage:
```python
from ezflow.src.trainer import ModelTrainer
from ezflow.models.xgb_model import XGBoostModel

model = XGBoostModel(params)
trainer = ModelTrainer(model, config)
trainer.train(X_train, y_train)
predictions = trainer.predict(X_test)
```

### 4. HyperparamTuner (`ezflow/src/hyperparam_tuning.py`)
Handles hyperparameter optimization:
- **Supported Methods**:
  - Optuna
  - Hyperopt
  - Grid Search
  - Random Search
- **Key Functions**:
  - `tune(X, y)`: Performs hyperparameter tuning
  - `create_best_model()`: Creates model with best parameters
  - `get_results()`: Returns tuning results

Example usage:
```python
from ezflow.src.hyperparam_tuning import HyperparamTuner

tuner = HyperparamTuner(XGBoostModel, config)
best_params = tuner.tune(X_train, y_train)
best_model = tuner.create_best_model()
```

### 5. Evaluator (`ezflow/src/evaluator.py`)
Provides comprehensive model evaluation:
- **Key Functions**:
  - `calculate_metrics(y_true, y_pred)`: Calculates various metrics
  - `confusion_matrix(y_true, y_pred)`: Generates confusion matrix
  - `roc_curve(y_true, y_proba)`: Plots ROC curve
  - `feature_importance_plot(importance_df)`: Visualizes feature importance
  - `plot_regression_results(y_true, y_pred)`: Plots regression results

Example usage:
```python
from ezflow.src.evaluator import Evaluator

evaluator = Evaluator(config)
metrics = evaluator.calculate_metrics(y_test, predictions)
evaluator.confusion_matrix(y_test, predictions, save_path="confusion_matrix.png")
```

### 6. Pipeline (`ezflow/src/pipeline.py`)
Orchestrates the entire ML workflow:
- **Key Functions**:
  - `run()`: Executes the complete pipeline
  - `cross_validate()`: Performs cross-validation
  - `predict(data)`: Makes predictions on new data
  - `save_pipeline(path)`: Saves the entire pipeline
  - `load_pipeline(path)`: Loads a saved pipeline

Example usage:
```python
from ezflow.src.pipeline import Pipeline

pipeline = Pipeline(config, model)
results = pipeline.run()
```

## Adding Custom Models

To add a custom model to ezflow:

1. Create a new file in `ezflow/models/` (e.g., `custom_model.py`)
2. Inherit from `BaseModel` class:
```python
from ezflow.models.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        # Initialize your model
        
    def train(self, X, y):
        # Implement training logic
        
    def predict(self, X):
        # Implement prediction logic
        
    def predict_proba(self, X):
        # Implement probability predictions
        
    def save(self, path):
        # Implement model saving
        
    def load(self, path):
        # Implement model loading
```

## Command Line Interface

ezflow provides a simple CLI interface:

```bash
# Initialize new project
ez init my_project

# Train a model
ez train --config config.yaml

# Make predictions
ez predict --input data.csv --model model.pkl --output predictions.csv

# Deploy model
ez deploy --model model.pkl --type api
```

## Configuration

Configuration can be defined in Python, JSON, or YAML format. Example:

```python
class Config:
    DATA_PATHS = {
        "train": "data/raw/train.csv",
        "test": "data/raw/test.csv"
    }
    
    MODEL = {
        "type": "xgb",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1
        }
    }
    
    FEATURE_ENGINEERING = {
        "scaling_method": "standard",
        "categorical_encoding": "onehot"
    }
    
    HYPERPARAMETER_TUNING = {
        "method": "optuna",
        "n_trials": 100
    }
```

## Examples

### Basic Training Pipeline
```python
from ezflow.src.pipeline import Pipeline
from ezflow.models.xgb_model import XGBoostModel
from ezflow.utils.config import Config

# Load configuration
config = Config()

# Create model and pipeline
model = XGBoostModel(config.MODEL['params'])
pipeline = Pipeline(config, model)

# Run pipeline
results = pipeline.run()
```

### Hyperparameter Tuning
```python
from ezflow.src.hyperparam_tuning import HyperparamTuner

# Create tuner
tuner = HyperparamTuner(XGBoostModel, config)

# Tune hyperparameters
best_params = tuner.tune(X_train, y_train)
best_model = tuner.create_best_model()

# Train with best parameters
pipeline = Pipeline(config, best_model)
results = pipeline.run()
```

### Model Deployment
```python
# Deploy as API
ez deploy --model models/best_model.pkl --type api

# Deploy as dashboard
ez deploy --model models/best_model.pkl --type dashboard
``` 