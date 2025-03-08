# 🚀 ezflow: Machine Learning Framework for Hackathons

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ezflow.git
cd ezflow

# Install in development mode
pip install -e .
```


### Using in Google Colab or Jupyter Notebooks

You can use ezflow directly in notebook environments without project initialization:

```python
# Import key components
from ezflow.models import ModelFactory

# Create a model
model = ModelFactory.create('xgboost', problem_type='classification')

# Train on your data (DataFrame format)
model.train(X_train, y_train)

# Evaluate performance
metrics = model.evaluate(X_test, y_test, is_classification=True)
print(f"Metrics: {metrics}")

# Make predictions
predictions = model.predict(X_test)

# Save model for later use
model.save("my_model.pkl")
```

### Experiment Tracking

ezflow includes built-in experiment tracking capabilities powered by MLflow. This lets you track metrics, parameters, and artifacts across multiple runs:

```python
from ezflow.models import ModelFactory

# Create your model
model = ModelFactory.create('xgboost', problem_type='classification')

# Start tracking run with an experiment name
model.start_run(experiment_name="my_classification_experiment")

# Train the model - metrics will be automatically logged
model.train(X_train, y_train)

# Log evaluation metrics
metrics = model.evaluate(X_test, y_test, is_classification=True)

# Perform cross-validation with automatic metric logging
cv_results = model.cross_validate(X, y, cv=5, is_classification=True)
print(f"Cross-validation results: {cv_results}")

# Hyperparameter optimization with tracking
param_space = model.get_param_search_space('xgboost')
best_params = model.search_hyperparams(
    X_train, y_train, 
    param_space=param_space,
    n_trials=20,
    cv=3
)
print(f"Best parameters: {best_params}")

# End the tracking run
model.end_run()

# The results will be available in your MLflow UI
# Run `mlflow ui` in your terminal to view experiments
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.