# 🚀 ezflow: Machine Learning Framework for Hackathons

ezflow is a flexible and easy-to-use machine learning framework designed for hackathons. It allows data scientists to focus on data preparation and feature engineering, while automating model training, hyperparameter optimization, and evaluation.

# typical workflow
```python
from ezflow.models import ModelFactory

# Register models (run once at startup, usually at application initialization)
ModelFactory.register_from_module('ezflow.models')

# Step 1: Instantiate model via ModelFactory
model = ModelFactory.create(
    model_type='xgboost',                 # model type name from registry
    problem_type='classification',        # classification or regression
    params={"max_depth": 5}               # initial parameters
)

# Step 2: Start experiment tracking (MLflow, optional)
model.start_run(experiment_name="xgb_classification")

# Step 3: Train the model
model.train(X_train, y_train)

# Step 4: Evaluate performance
eval_metrics = model.evaluate(X_test, y_test, is_classification=True)
print(f"Evaluation metrics: {eval_metrics}")

# Step 5: Plot training history (e.g., accuracy over time)
model.plot_training_history(metric='accuracy', save_path='accuracy_plot.png')

# Step 6: Hyperparameter tuning with Optuna (optional)
best_params = model.search_hyperparams(
    X_train, y_train,
    param_space={'max_depth': ('int', 3, 10)},
    n_trials=10,
    cv=3,
    is_classification=True,
    metric='accuracy'
)

print(f"Best hyperparameters found: {best_params}")

# Step 7: Re-train with best hyperparameters (optional)
model.set_params(best_params)
model.train(X_train, y_train)

# Step 8: Save the final trained model to disk
model.save("model.joblib")

# Step 9: End experiment tracking
model.end_run()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.