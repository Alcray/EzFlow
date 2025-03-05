# EZFlow: A Machine Learning Framework

<p align="center">
  <a href="https://github.com/Alcray/ezflow/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Alcray/ezflow.svg" alt="License"></a>
  <a href="https://github.com/Alcray/ezflow/stargazers"><img src="https://img.shields.io/github/stars/Alcray/ezflow.svg" alt="GitHub stars"></a>
</p>

EZFlow is a flexible, modular machine learning framework designed to streamline the development and deployment of ML pipelines. It provides a unified API for working with different types of datasets, models, and experiment tracking tools.

## Features

- **Simple Dataset API**: Easily load, preprocess, and split various types of data
- **Unified Pipeline Interface**: Work with scikit-learn, PyTorch, and TensorFlow models using the same interface
- **Experiment Tracking**: Track metrics, hyperparameters, and artifacts with MLflow integration
- **Reproducibility**: Ensure experiment reproducibility with configuration management
- **CLI Tools**: Command-line utilities for project creation and management

## Installation

### Using Conda (Recommended)

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate ezflow

# Install the package in development mode
pip install -e .
```

### Development Installation

```bash
# Install in development mode from source
git clone https://github.com/Alcray/ezflow.git
cd ezflow
pip install -e .
```

## Quick Start

### Creating a New Project

```bash
# Create a new EZFlow project
ezflow create my_project
cd my_project
```

### Running the Iris Example

```python
from ezflow.core.dataset import IrisDataset
from ezflow.core.pipeline import SklearnPipelineWrapper
from ezflow.core.experiment import ExperimentTracker, ExperimentConfig

# Create and load dataset
dataset = IrisDataset(data_dir="./data")
dataset.load_data()
dataset.split_data(val_size=0.2)

# Create pipeline
pipeline = SklearnPipelineWrapper([
    ("scaler", "sklearn.preprocessing.StandardScaler", {}),
    ("classifier", "sklearn.ensemble.RandomForestClassifier", {"n_estimators": 100})
])

# Train and evaluate
X_train = dataset.get_features(dataset.train_data)
y_train = dataset.get_labels(dataset.train_data)
pipeline.fit(X_train, y_train)

# Save model
pipeline.save("./models/iris_model.joblib")
```

Or run the included example:

```bash
python -m ezflow.examples.iris_example
```

## Data Management

EZFlow works with data stored in the `data/` directory. This directory is included in `.gitignore` to prevent uploading datasets to GitHub. When using the framework, place your datasets in this directory, and they will be automatically used by the dataset classes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
