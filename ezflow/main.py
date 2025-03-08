#!/usr/bin/env python
"""
CLI entry point for the ezflow framework.
"""

import argparse
import os
import logging
import json
import tempfile
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import yaml
from pprint import pformat

from ezflow.models import get_model, ModelFactory
from ezflow.data.processor import load_manifest, save_manifest, process_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ezflow")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")

def init_project(project_name: str):
    """
    Create basic project structure.
    
    Args:
        project_name (str): Name of the project directory to create
    """
    if os.path.exists(project_name):
        logger.error(f"Directory {project_name} already exists.")
        return False
    
    logger.info(f"Creating project structure in: {project_name}")
    
    # Create basic directory structure
    project_dirs = [
        "",
        "data",
        "data/raw",
        "data/processed",
        "models",
        "plots",
        "logs"
    ]
    
    for directory in project_dirs:
        dir_path = os.path.join(project_name, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    # Create example manifest.jsonl
    example_manifest = """{"feature1": 1.0, "feature2": "value", "target": 0}
{"feature1": 2.0, "feature2": "other", "target": 1}"""
    
    with open(os.path.join(project_name, "data/raw/example_manifest.jsonl"), "w") as f:
        f.write(example_manifest)
    
    # Create example preprocessing steps
    example_steps = [
        {"name": "drop_duplicates"},
        {"name": "filter_by_value", "filters": {"feature1": 1.0}},
        {"name": "split_manifest", "splits": {"train": 0.8, "val": 0.2}}
    ]
    
    with open(os.path.join(project_name, "data/raw/preprocess_steps.json"), "w") as f:
        json.dump(example_steps, f, indent=2)
    
    # Create example hyperparameter search config
    example_hyperparam_config = {
        "model": "xgboost",
        "problem_type": "classification",
        "param_space": {
            "n_estimators": ["int", 50, 500],
            "learning_rate": ["loguniform", 0.01, 0.3],
            "max_depth": ["int", 3, 10]
        },
        "n_trials": 10,
        "cv": 3,
        "metric": "accuracy"
    }
    
    with open(os.path.join(project_name, "models/hyperparam_search_example.yaml"), "w") as f:
        yaml.dump(example_hyperparam_config, f, default_flow_style=False)
    
    # Create .gitignore
    gitignore_content = """# Data
data/raw/*
data/processed/*

# Models
models/*.pkl
models/*.json
models/*.yaml

# Logs
logs/*
*.log

# Plots
plots/*

# Python
__pycache__/
*.py[cod]
*$py.class

# Keep examples
!data/raw/example_manifest.jsonl
!data/raw/preprocess_steps.json
!models/hyperparam_search_example.yaml
"""
    
    with open(os.path.join(project_name, ".gitignore"), "w") as f:
        f.write(gitignore_content)
    
    # Create example README.md
    readme_content = """# {project_name}

This project was created using ezflow.

## Getting Started

1. Prepare your manifest.jsonl file:
   - Place your data in the `data/raw/` directory.
   - See `data/raw/example_manifest.jsonl` for the expected format.

2. Train a model:
   ```bash
   ez train --model xgboost --manifest data/raw/manifest.jsonl
   ```

3. Find optimal hyperparameters:
   ```bash
   ez hyperparams --config models/hyperparam_search_example.yaml --manifest data/raw/manifest.jsonl
   ```

4. Make predictions:
   ```bash
   ez predict --model models/xgboost_model.pkl --manifest data/raw/test.jsonl --output predictions.jsonl
   ```

## Available Models

{available_models}

## Documentation

For more information, visit [GitHub repository](https://github.com/yourusername/ezflow).
"""

    available_models_info = "- " + "\n- ".join(ModelFactory.available_models())
    readme_content = readme_content.format(
        project_name=project_name,
        available_models=available_models_info
    )
    
    with open(os.path.join(project_name, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Create an mlflow.yml config if MLflow is available
    if MLFLOW_AVAILABLE:
        mlflow_config = {
            "tracking_uri": "./mlruns",
            "experiment_name": project_name
        }
        
        with open(os.path.join(project_name, "mlflow.yml"), "w") as f:
            yaml.dump(mlflow_config, f, default_flow_style=False)
    
    logger.info(f"""
Project initialized successfully!
    
To get started:
1. Create your manifest.jsonl file in data/raw/
   Example format is in data/raw/example_manifest.jsonl
2. Train your model:
   ez train --model xgboost --manifest data/raw/manifest.jsonl
3. Find optimal hyperparameters:
   ez hyperparams --config models/hyperparam_search_example.yaml --manifest data/raw/manifest.jsonl
""")
    return True

def train_model(model_type: str, manifest_path: str, problem_type: str = 'classification', 
                preprocess_steps: Optional[str] = None, model_params: Optional[Dict] = None, 
                target_key: str = "target", output_path: Optional[str] = None, 
                cross_validate: bool = False, cv_folds: int = 5):
    """
    Train a model using manifest data.
    
    Args:
        model_type (str): Type of model to train (e.g., xgboost)
        manifest_path (str): Path to manifest.jsonl
        problem_type (str): Problem type (classification or regression)
        preprocess_steps (Optional[str]): Path to preprocessing steps config
        model_params (Optional[Dict]): Model-specific parameters
        target_key (str): Key in manifest containing the target
        output_path (Optional[str]): Path to save the trained model
        cross_validate (bool): Whether to perform cross-validation
        cv_folds (int): Number of folds for cross-validation
    
    Returns:
        bool: Success flag
    """
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found: {manifest_path}")
        return False
    
    # Process manifest if needed
    processed_manifest_path = manifest_path
    if preprocess_steps:
        try:
            # This would need to be implemented to process according to the config
            logger.info(f"Preprocessing manifest using steps from: {preprocess_steps}")
            # For now, we'll continue using the original manifest
            logger.warning("Using original manifest, processed manifest path unknown.")
        except Exception as e:
            logger.error(f"Failed to process manifest: {str(e)}")
            return False
    
    # Create model
    try:
        logger.info(f"Creating {model_type} model for {problem_type} problem")
        model = get_model(
            model_type=model_type, 
            problem_type=problem_type, 
            params=model_params or {}, 
            target_key=target_key
        )
        
        # Load data for cross-validation
        if cross_validate:
            logger.info(f"Loading data for cross-validation with {cv_folds} folds")
            # Load manifest
            data = []
            with open(manifest_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            
            # Separate features and target
            if target_key not in df.columns:
                logger.error(f"Target key '{target_key}' not found in manifest")
                return False
            
            y = df[target_key]
            X = df.drop(columns=[target_key])
            
            # Run cross-validation
            logger.info("Starting cross-validation")
            cv_metrics = model.cross_validate(
                X=X, 
                y=y, 
                cv=cv_folds, 
                is_classification=(problem_type == 'classification')
            )
            
            logger.info("Cross-validation results:")
            for metric, value in cv_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Train on full dataset
        logger.info("Training model on full dataset")
        model.train(processed_manifest_path)
        logger.info("Training completed successfully!")
        
        # Save model if path provided
        if output_path:
            model.save(output_path)
            logger.info(f"Model saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

def search_hyperparams(config_path: str, manifest_path: str):
    """
    Search for optimal hyperparameters.
    
    Args:
        config_path (str): Path to hyperparameter search config YAML file
        manifest_path (str): Path to manifest.jsonl
        
    Returns:
        bool: Success flag
    """
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load hyperparameter search config: {str(e)}")
        return False
    
    # Extract configuration
    model_type = config.get('model', 'xgboost')
    problem_type = config.get('problem_type', 'classification')
    param_space = config.get('param_space', {})
    n_trials = config.get('n_trials', 10)
    cv = config.get('cv', 3)
    metric = config.get('metric', None)
    target_key = config.get('target_key', 'target')
    
    # Validate config
    if not param_space:
        logger.error("No parameter space defined in config")
        return False
    
    # Format param_space if needed
    formatted_param_space = {}
    for param_name, param_spec in param_space.items():
        if isinstance(param_spec, list) and len(param_spec) >= 2:
            if param_spec[0] in ['int', 'float', 'loguniform'] and len(param_spec) == 3:
                formatted_param_space[param_name] = tuple(param_spec)
            else:
                formatted_param_space[param_name] = param_spec
        else:
            logger.warning(f"Invalid parameter specification for {param_name}, skipping")
    
    # Load data
    try:
        # Load manifest
        data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Separate features and target
        if target_key not in df.columns:
            logger.error(f"Target key '{target_key}' not found in manifest")
            return False
        
        y = df[target_key]
        X = df.drop(columns=[target_key])
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return False
    
    # Create model with default parameters
    try:
        logger.info(f"Creating {model_type} model for hyperparameter search")
        model = get_model(
            model_type=model_type, 
            problem_type=problem_type, 
            params={}, 
            target_key=target_key
        )
        
        # Use custom logger for hyperparameter search
        search_log_path = os.path.join("logs", f"hyperparam_search_{model_type}_{problem_type}.log")
        os.makedirs(os.path.dirname(search_log_path), exist_ok=True)
        
        search_logger = logging.getLogger(f"ezflow.hyperparams.{model_type}")
        search_logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(search_log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        search_logger.addHandler(file_handler)
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(f"hyperparam_search_{model_type}")
            mlflow.start_run(run_name=f"{model_type}_{problem_type}")
            mlflow.log_params({
                "model_type": model_type,
                "problem_type": problem_type,
                "n_trials": n_trials,
                "cv": cv,
                "metric": metric,
                "target_key": target_key,
                "param_space": str(formatted_param_space)
            })
        
        # Perform hyperparameter search
        logger.info(f"Starting hyperparameter search with {n_trials} trials")
        best_params = model.search_hyperparams(
            X=X,
            y=y,
            param_space=formatted_param_space,
            n_trials=n_trials,
            cv=cv,
            is_classification=(problem_type == 'classification'),
            metric=metric
        )
        
        # Log results
        logger.info("Best parameters found:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        # Save best parameters to file
        output_dir = os.path.dirname(config_path)
        output_base = os.path.splitext(os.path.basename(config_path))[0]
        output_path = os.path.join(output_dir, f"{output_base}_best_params.json")
        
        with open(output_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"Best parameters saved to: {output_path}")
        
        # End MLflow run if available
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
        
        # Train final model with best parameters
        final_model_path = os.path.join(output_dir, f"{output_base}_best_model.pkl")
        
        logger.info(f"Training final model with best parameters")
        final_model = get_model(
            model_type=model_type, 
            problem_type=problem_type, 
            params=best_params, 
            target_key=target_key
        )
        
        final_model.train(X, y)
        final_model.save(final_model_path)
        
        logger.info(f"Final model saved to: {final_model_path}")
        
        return True
    except Exception as e:
        logger.error(f"Hyperparameter search failed: {str(e)}")
        return False

def predict(model_path: str, manifest_path: str, output_path: str):
    """
    Make predictions using a trained model.
    
    Args:
        model_path (str): Path to trained model file
        manifest_path (str): Path to manifest.jsonl with features
        output_path (str): Path to save predictions
        
    Returns:
        bool: Success flag
    """
    # Load model
    try:
        # Get model type from filename if possible, default to xgboost
        model_name = os.path.basename(model_path).split('_')[0]
        model_type = model_name if model_name in ModelFactory.available_models() else "xgboost"
        
        model = get_model(model_type)
        model.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False
    
    # Make predictions
    try:
        # Check if manifest file exists
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest file not found: {manifest_path}")
            return False
            
        # Make predictions
        predictions = model.predict(manifest_path)
        
        # Create directory for output if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's a directory part
            os.makedirs(output_dir, exist_ok=True)
        
        # Save as manifest
        result_manifest = [{"prediction": float(p)} for p in predictions]
        
        # Write directly to file
        with open(output_path, 'w') as f:
            for item in result_manifest:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Predictions saved to {output_path}")
        
        # Try to save prediction probabilities if it's a classification model
        try:
            proba_predictions = model.predict_proba(manifest_path)
            
            # For binary classification
            if proba_predictions.shape[1] == 2:
                proba_output_path = f"{os.path.splitext(output_path)[0]}_proba.jsonl"
                proba_manifest = [{"prediction_proba": float(p[1])} for p in proba_predictions]
                
                with open(proba_output_path, 'w') as f:
                    for item in proba_manifest:
                        f.write(json.dumps(item) + '\n')
                
                logger.info(f"Prediction probabilities saved to {proba_output_path}")
        except Exception as e:
            logger.debug(f"Could not save probabilities: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return False

def evaluate_model(model_path: str, manifest_path: str, target_key: str = "target",
                  problem_type: str = "classification", output_dir: Optional[str] = None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path (str): Path to trained model file
        manifest_path (str): Path to manifest.jsonl with features and ground truth
        target_key (str): Key in manifest containing the target
        problem_type (str): Problem type (classification or regression)
        output_dir (Optional[str]): Directory to save evaluation results
        
    Returns:
        bool: Success flag
    """
    # Load model
    try:
        # Get model type from filename if possible, default to xgboost
        model_name = os.path.basename(model_path).split('_')[0]
        model_type = model_name if model_name in ModelFactory.available_models() else "xgboost"
        
        model = get_model(model_type)
        model.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False
    
    # Load test data
    try:
        # Load manifest
        data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Separate features and target
        if target_key not in df.columns:
            logger.error(f"Target key '{target_key}' not found in manifest")
            return False
        
        y = df[target_key]
        X = df.drop(columns=[target_key])
        
        # Evaluate model
        is_classification = problem_type == 'classification'
        metrics = model.evaluate(X, y, is_classification=is_classification)
        
        # Display metrics
        logger.info("Evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save metrics and plots if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metrics to JSON
            metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Metrics saved to {metrics_path}")
            
            # Save confusion matrix for classification
            if is_classification:
                try:
                    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                    
                    y_pred = model.predict(X)
                    cm = confusion_matrix(y, y_pred)
                    
                    plt.figure(figsize=(8, 6))
                    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
                    plt.title('Confusion Matrix')
                    
                    cm_path = os.path.join(output_dir, "confusion_matrix.png")
                    plt.savefig(cm_path)
                    
                    logger.info(f"Confusion matrix saved to {cm_path}")
                except Exception as e:
                    logger.debug(f"Could not create confusion matrix: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return False

def list_models():
    """
    List all available model types and their default parameters.
    """
    available_models = ModelFactory.available_models()
    
    logger.info("Available models:")
    for model_type in available_models:
        logger.info(f"  - {model_type}")
        
        # Show default parameters for classification
        try:
            default_params = ModelFactory.get_default_params(model_type, 'classification')
            logger.info(f"    Default classification parameters:")
            for param, value in default_params.items():
                logger.info(f"      {param}: {value}")
        except Exception:
            pass
        
        # Show hyperparameter search space
        try:
            param_space = ModelFactory.get_param_search_space(model_type, 'classification')
            if param_space:
                logger.info(f"    Hyperparameter search space:")
                for param, spec in param_space.items():
                    logger.info(f"      {param}: {spec}")
        except Exception:
            pass

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="ez - ML framework for hackathons")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init project command
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument("project_name", help="Name of the project to initialize")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", required=True, help="Model type (e.g., xgboost)")
    train_parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    train_parser.add_argument("--problem-type", default="classification", 
                            choices=["classification", "regression"], 
                            help="Problem type")
    train_parser.add_argument("--preprocess", help="Path to preprocessing steps config")
    train_parser.add_argument("--target-key", default="target", help="Key for target variable")
    train_parser.add_argument("--output", help="Path to save model")
    train_parser.add_argument("--params", type=json.loads, help="Model parameters as JSON string")
    train_parser.add_argument("--cross-validate", action="store_true", help="Perform cross-validation")
    train_parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    predict_parser.add_argument("--output", required=True, help="Path to save predictions")
    
    # Hyperparameter search command
    hyperparams_parser = subparsers.add_parser("hyperparams", help="Search for optimal hyperparameters")
    hyperparams_parser.add_argument("--config", required=True, help="Path to hyperparameter search config YAML")
    hyperparams_parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model on test data")
    evaluate_parser.add_argument("--model", required=True, help="Path to trained model")
    evaluate_parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl with ground truth")
    evaluate_parser.add_argument("--target-key", default="target", help="Key for target variable")
    evaluate_parser.add_argument("--problem-type", default="classification", 
                              choices=["classification", "regression"], 
                              help="Problem type")
    evaluate_parser.add_argument("--output-dir", help="Directory to save evaluation results")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available model types")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_project(args.project_name)
    elif args.command == "train":
        output_path = args.output or f"models/{args.model}_model.pkl"
        train_model(args.model, args.manifest, args.problem_type, args.preprocess, 
                   args.params, args.target_key, output_path, args.cross_validate,
                   args.cv_folds)
    elif args.command == "predict":
        predict(args.model, args.manifest, args.output)
    elif args.command == "hyperparams":
        search_hyperparams(args.config, args.manifest)
    elif args.command == "evaluate":
        evaluate_model(args.model, args.manifest, args.target_key, args.problem_type, args.output_dir)
    elif args.command == "list-models":
        list_models()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 