#!/usr/bin/env python3
"""
Basic example of using the ezflow framework for a machine learning experiment.

This example:
1. Creates a synthetic dataset
2. Trains multiple models with different parameters
3. Compares the models and visualizes the results
4. Generates a final submission
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Import our ezflow framework
import ezflow as ez

# Set up directories
os.makedirs('data', exist_ok=True)
os.makedirs('experiments', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

# --------------------------------
# Step 1: Create synthetic dataset
# --------------------------------
print("Creating synthetic dataset...")

# Generate synthetic classification data
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=10, 
    n_redundant=5, 
    n_classes=2, 
    random_state=42
)

# Create IDs for each sample
ids = np.arange(1000)

# Split into train and test sets
train_mask = np.random.RandomState(42).rand(1000) < 0.8
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
ids_train, ids_test = ids[train_mask], ids[~train_mask]

# Create column names
feature_names = [f'feature_{i}' for i in range(20)]

# Create pandas dataframes
train_df = pd.DataFrame(
    data=np.column_stack([ids_train, X_train, y_train]),
    columns=['id'] + feature_names + ['target']
)

test_df = pd.DataFrame(
    data=np.column_stack([ids_test, X_test]),
    columns=['id'] + feature_names
)

# Save the datasets
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# --------------------------------
# Step 2: Create a TabularDataset
# --------------------------------
print("Creating dataset object...")

dataset = ez.TabularDataset(
    data_dir='data',
    train_file='train.csv',
    test_file='test.csv',
    id_column='id',
    target_column='target'
)

# Load and preprocess the data
dataset.load_data()
dataset.create_train_val_split(val_ratio=0.2, random_state=42)
dataset.preprocess(normalize=True)

# Get features and labels
X_train = dataset.get_features('train')
y_train = dataset.get_labels('train')
X_val = dataset.get_features('val')
y_val = dataset.get_labels('val')
X_test = dataset.get_features('test')

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Print dataset summary
print("\nDataset Summary:")
summary = dataset.get_summary()
for key, value in summary.items():
    if key != 'feature_names':  # Skip the long list of feature names
        print(f"  {key}: {value}")

# --------------------------------
# Step 3: Run experiments
# --------------------------------
print("\nRunning experiments...")

# Train a Random Forest model
rf_experiment = ez.Experiment(
    name="random_forest",
    description="Random Forest classifier with default parameters",
    tags=["baseline", "random_forest"]
)

with rf_experiment.train_phase():
    rf_model = ez.Model(
        model=RandomForestClassifier(random_state=42),
        feature_names=dataset.get_feature_names()
    )
    rf_model.fit(X_train, y_train)
    
    # Save the model
    model_path = rf_model.save(rf_experiment.experiment_dir)
    rf_experiment.log_artifact("model", model_path)

with rf_experiment.eval_phase():
    # Evaluate on validation data
    rf_metrics = rf_model.evaluate(X_val, y_val)
    for metric_name, value in rf_metrics.items():
        rf_experiment.log_metric(metric_name, value)
    
    # Log feature importance
    importance = rf_model.get_feature_importance()
    importances = [importance[name] for name in dataset.get_feature_names()]
    
    ez.plot_feature_importance(
        feature_importances=importances,
        feature_names=dataset.get_feature_names(),
        save_path=f"{rf_experiment.experiment_dir}/feature_importance.png"
    )
    rf_experiment.log_artifact("feature_importance", f"{rf_experiment.experiment_dir}/feature_importance.png")
    
    # Plot confusion matrix
    y_pred = rf_model.predict(X_val)
    ez.plot_confusion_matrix(
        y_val, y_pred, 
        save_path=f"{rf_experiment.experiment_dir}/confusion_matrix.png"
    )
    rf_experiment.log_artifact("confusion_matrix", f"{rf_experiment.experiment_dir}/confusion_matrix.png")
    
    # Generate classification report
    report = rf_model.generate_classification_report(X_val, y_val)
    rf_experiment.log_params({"classification_report": report})

rf_experiment.finish()
print(f"Random Forest experiment completed: {rf_experiment.id}")

# Train a Logistic Regression model
lr_experiment = ez.Experiment(
    name="logistic_regression",
    description="Logistic Regression classifier with L2 regularization",
    tags=["baseline", "logistic_regression"]
)

with lr_experiment.train_phase():
    lr_model = ez.Model(
        model=LogisticRegression(random_state=42, max_iter=1000),
        feature_names=dataset.get_feature_names()
    )
    lr_model.fit(X_train, y_train)
    
    # Save the model
    model_path = lr_model.save(lr_experiment.experiment_dir)
    lr_experiment.log_artifact("model", model_path)

with lr_experiment.eval_phase():
    # Evaluate on validation data
    lr_metrics = lr_model.evaluate(X_val, y_val)
    for metric_name, value in lr_metrics.items():
        lr_experiment.log_metric(metric_name, value)
    
    # Log feature importance (coefficients)
    importance = lr_model.get_feature_importance()
    importances = [importance[name] for name in dataset.get_feature_names()]
    
    ez.plot_feature_importance(
        feature_importances=importances,
        feature_names=dataset.get_feature_names(),
        save_path=f"{lr_experiment.experiment_dir}/feature_importance.png"
    )
    lr_experiment.log_artifact("feature_importance", f"{lr_experiment.experiment_dir}/feature_importance.png")
    
    # Plot ROC curve
    y_proba = lr_model.predict_proba(X_val)
    ez.plot_roc_curve(
        y_val, y_proba, 
        save_path=f"{lr_experiment.experiment_dir}/roc_curve.png"
    )
    lr_experiment.log_artifact("roc_curve", f"{lr_experiment.experiment_dir}/roc_curve.png")

lr_experiment.finish()
print(f"Logistic Regression experiment completed: {lr_experiment.id}")

# --------------------------------
# Step 4: Compare experiments
# --------------------------------
print("\nComparing experiments...")

# List all experiments
experiments = ez.Experiment.list_experiments()
print(f"Found {len(experiments)} experiments")

# Get the experiment IDs
exp_ids = [exp['id'] for exp in experiments]

# Compare experiments
comparison = ez.Experiment.compare_experiments(
    experiment_ids=exp_ids,
    metric_names=['accuracy', 'precision', 'recall', 'f1']
)

print("\nExperiment Comparison:")
print(comparison)

# --------------------------------
# Step 5: Create final submission
# --------------------------------
print("\nCreating final submission...")

# Determine the best model
best_model_id = None
best_accuracy = 0

for i, row in comparison.iterrows():
    if 'accuracy_best' in row and row['accuracy_best'] > best_accuracy:
        best_accuracy = row['accuracy_best']
        best_model_id = row['id']

print(f"Best model: {best_model_id} with accuracy {best_accuracy:.4f}")

# Load the best experiment
best_exp_dir = None
for exp in experiments:
    if exp['id'] == best_model_id:
        best_exp_dir = os.path.join("experiments", f"{exp['name']}_{exp['id']}")
        break

if best_exp_dir:
    # Load the best model
    best_model = ez.Model.load(os.path.join(best_exp_dir, "model.pkl"))
    
    # Generate predictions on test set
    test_ids = dataset.get_ids('test')
    test_predictions = best_model.predict(X_test)
    
    # Create submission file
    submission_path = 'submissions/final_submission.csv'
    ez.create_submission_file(
        predictions=test_predictions,
        ids=test_ids,
        submission_path=submission_path,
        columns=['id', 'prediction']
    )
    
    print(f"Submission file created: {submission_path}")
    
    # Create experiment dashboard
    for exp_id in exp_ids:
        for exp in experiments:
            if exp['id'] == exp_id:
                exp_dir = os.path.join("experiments", f"{exp['name']}_{exp['id']}")
                dashboard_path = ez.create_experiment_dashboard(exp_dir)
                print(f"Dashboard created: {dashboard_path}")

print("\nDemo completed successfully!") 