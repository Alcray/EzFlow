# ezflow: Hackathon ML Experiment Framework

A streamlined framework for conducting, tracking, and visualizing machine learning experiments, specifically designed for hackathons where time and clarity are critical.

## Features

- **Standardized Dataset Interface**: Base dataset class with common functionality for different data types
- **Experiment Tracking**: Comprehensive tracking of metrics, models, and metadata
- **Visualization Tools**: Built-in visualization for model performance and comparison
- **Easy Reporting**: HTML dashboards for sharing results with teammates and judges
- **Rapid Iteration**: Tools to quickly iterate on models and track improvements
- **Unified API**: Consistent interface for various ML tasks

## Installation

```bash
# Install from the current directory
pip install .

# Or install directly from GitHub
pip install git+https://github.com/Alcray/BioML.git@ezflow
```

## Quick Start

### 1. Dataset Setup

```python
import ezflow as ez
import pandas as pd

# Create a tabular dataset
dataset = ez.TabularDataset(
    data_dir='data',
    train_file='train.csv',
    test_file='test.csv',
    id_column='id',
    target_column='target',
    categorical_columns=['category1', 'category2'],
    date_columns=['date_created']
)

# Load and preprocess the data
dataset.load_data()
dataset.preprocess(normalize=True)

# Get features and labels
X_train = dataset.get_features('train')
y_train = dataset.get_labels('train')
X_test = dataset.get_features('test')

# Print dataset summary
print(dataset.get_summary())
```

### 2. Experiment Tracking

```python
import ezflow as ez
from sklearn.ensemble import RandomForestClassifier

# Create an experiment
experiment = ez.Experiment(
    name="rf_baseline",
    description="Random Forest baseline model",
    tags=["baseline", "random_forest"]
)

# Log parameters
experiment.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
})

# Training phase
with experiment.train_phase():
    # Create and train a model
    model = ez.Model(
        model=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        feature_names=dataset.get_feature_names()
    )
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = model.save(experiment.experiment_dir)
    experiment.log_artifact("model", model_path)

# Evaluation phase
with experiment.eval_phase():
    # Evaluate on validation data
    X_val = dataset.get_features('val')
    y_val = dataset.get_labels('val')
    
    metrics = model.evaluate(X_val, y_val)
    for metric_name, value in metrics.items():
        experiment.log_metric(metric_name, value)
    
    # Create and log visualizations
    if model.model_type == "classifier":
        y_pred = model.predict(X_val)
        ez.plot_confusion_matrix(
            y_val, y_pred, 
            save_path=f"{experiment.experiment_dir}/confusion_matrix.png"
        )
        experiment.log_artifact("confusion_matrix", f"{experiment.experiment_dir}/confusion_matrix.png")

# Finish the experiment
experiment.finish()

# Create a dashboard for the experiment
dashboard_path = ez.create_experiment_dashboard(experiment.experiment_dir)
print(f"Experiment dashboard created at: {dashboard_path}")
```

### 3. Model Comparison

```python
import ezflow as ez
import matplotlib.pyplot as plt

# List all experiments
experiments = ez.Experiment.list_experiments()
print(f"Found {len(experiments)} experiments")

# Get the experiment IDs
exp_ids = [exp['id'] for exp in experiments]

# Compare experiments
comparison = ez.Experiment.compare_experiments(
    experiment_ids=exp_ids,
    metric_names=['accuracy', 'f1', 'precision', 'recall']
)

# Display comparison
print(comparison)

# Plot comparison of a specific metric
ez.plot_metric_comparison(
    experiment_dirs=[f"experiments/{exp['name']}_{exp['id']}" for exp in experiments],
    metric_name='accuracy',
    title='Accuracy Comparison'
)
plt.show()
```

### 4. Submission Generation

```python
import ezflow as ez

# Load the best model
best_model = ez.Model.load('experiments/best_model.pkl')

# Get test data
X_test = dataset.get_features('test')
test_ids = dataset.get_ids('test')

# Generate predictions
predictions = best_model.predict(X_test)

# Create submission file
ez.create_submission_file(
    predictions=predictions,
    ids=test_ids,
    submission_path='submission.csv'
)
```

## Creating Custom Datasets

You can easily create custom dataset implementations by inheriting from the `BaseDataset` class:

```python
import ezflow as ez
import pandas as pd
import numpy as np

class MyCompetitionDataset(ez.BaseDataset):
    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, **kwargs)
        # Add competition-specific initialization
        
    def load_data(self) -> None:
        # Implement data loading logic
        self.train_data = pd.read_csv(f"{self.data_dir}/train.csv")
        self.test_data = pd.read_csv(f"{self.data_dir}/test.csv")
        
    def get_features(self, split: str = 'train') -> np.ndarray:
        # Implement feature extraction
        if split == 'train':
            return self.train_data[['feature1', 'feature2']].values
        elif split == 'test':
            return self.test_data[['feature1', 'feature2']].values
        
    def get_labels(self, split: str = 'train') -> np.ndarray:
        # Implement label extraction
        if split == 'train':
            return self.train_data['target'].values
        else:
            raise ValueError(f"Labels not available for {split}")
```

## Documentation

For complete documentation, see the [API reference](docs/api_reference.md) and [examples](examples/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
