# EZFlow: A Machine Learning Framework

<p align="center">
  <img src="docs/images/ezflow_logo.png" alt="EZFlow Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/yourusername/ezflow/actions"><img src="https://github.com/yourusername/ezflow/workflows/tests/badge.svg" alt="Tests Status"></a>
  <a href="https://pypi.org/project/ezflow-ml/"><img src="https://img.shields.io/pypi/v/ezflow-ml.svg" alt="PyPI version"></a>
  <a href="https://github.com/yourusername/ezflow/blob/main/LICENSE"><img src="https://img.shields.io/github/license/yourusername/ezflow.svg" alt="License"></a>
  <a href="https://github.com/yourusername/ezflow/stargazers"><img src="https://img.shields.io/github/stars/yourusername/ezflow.svg" alt="GitHub stars"></a>
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

### Using Pip

```bash
# Install from PyPI
pip install ezflow-ml

# Or install in development mode from source
git clone https://github.com/yourusername/ezflow.git
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

## Documentation

For more detailed documentation, please see:

- [Architecture Overview](docs/architecture.md)
- [Dataset Module](docs/dataset.md)
- [Pipeline Module](docs/pipeline.md)
- [Experiment Tracking](docs/experiments.md)
- [Command-line Interface](docs/cli.md)
- [Examples and Tutorials](docs/examples.md)
- [API Reference](docs/api_reference.md)

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
