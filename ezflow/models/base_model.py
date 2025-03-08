"""
Base model interface for the ezflow framework.

This module defines the base interface for all models in the framework.
"""

from abc import ABC, abstractmethod
import os
import logging
import time
import json
import datetime
from typing import Dict, Any, Union, Optional, List, Callable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, r2_score
import joblib

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all models in the ezflow framework.
    
    All model implementations should inherit from this class and implement
    its abstract methods.
    """
    
    @abstractmethod
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the model with parameters.
        
        Args:
            params (Dict[str, Any]): Model-specific parameters.
        """
        self.params = params
        self.model = None
        self.is_fitted = False
        self.training_history = {}
        self.feature_names = None
        self.target_key = None
        self.experiment_id = None
        self.run_id = None
    
    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Train the model on the given data.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            
        Returns:
            np.ndarray: Predictions.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probability predictions on new data (for classification models).
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            
        Returns:
            np.ndarray: Probability predictions.
            
        Note:
            This method should be implemented for classification models.
            For regression models, it can raise NotImplementedError.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path (str): Path to load the model from.
        """
        pass
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances if the model supports it.
        
        Returns:
            pd.DataFrame: Feature importances sorted by importance.
            
        Raises:
            NotImplementedError: If the model doesn't support feature importances.
        """
        raise NotImplementedError("Feature importance not implemented for this model type")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters.
        """
        return self.params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set the model parameters.
        
        Args:
            params (Dict[str, Any]): New model parameters.
        """
        self.params = params
        # Re-initialize the model with new parameters if needed
        # This may need to be implemented by the specific model class
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to tracking system and internal history.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            step (Optional[int]): Step number for tracking
        """
        # Add timestamp to metrics
        timestamp = time.time()
        metrics_with_time = {**metrics, "timestamp": timestamp}
        
        # Update internal training history
        if step is not None:
            self.training_history[step] = metrics_with_time
        else:
            step = len(self.training_history)
            self.training_history[step] = metrics_with_time
            
        # Log to stdout
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step}: {metrics_str}")
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
    
    def start_run(self, experiment_name: Optional[str] = None) -> None:
        """
        Start a tracking run for this model training session.
        
        Args:
            experiment_name (Optional[str]): Name of experiment for tracking
        """
        self.training_history = {}
        
        if MLFLOW_AVAILABLE:
            # Set experiment
            if experiment_name:
                mlflow.set_experiment(experiment_name)
                
            # Start run
            mlflow.start_run()
            self.run_id = mlflow.active_run().info.run_id
            self.experiment_id = mlflow.active_run().info.experiment_id
            
            # Log parameters
            mlflow.log_params(self.get_params())
            
            logger.info(f"Started MLflow run: {self.run_id}")
        else:
            logger.info("MLflow not available. Install with: pip install mlflow")
    
    def end_run(self) -> None:
        """End the current tracking run."""
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.end_run()
            logger.info("Ended MLflow run")
    
    def plot_training_history(self, metric: str = "loss", save_path: Optional[str] = None) -> None:
        """
        Plot training history for a specific metric.
        
        Args:
            metric (str): Name of metric to plot
            save_path (Optional[str]): Path to save the plot
        """
        if not self.training_history:
            logger.warning("No training history available to plot")
            return
            
        steps = []
        values = []
        
        for step, metrics in sorted(self.training_history.items()):
            if metric in metrics:
                steps.append(step)
                values.append(metrics[metric])
        
        if not steps:
            logger.warning(f"Metric '{metric}' not found in training history")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values)
        plt.title(f"Training {metric.capitalize()}")
        plt.xlabel("Step")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved training history plot to {save_path}")
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_artifact(save_path)
        else:
            plt.show()
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
                is_classification: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features
            y (Union[np.ndarray, pd.Series]): Ground truth labels/values
            is_classification (bool): Whether this is a classification task
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
            
        y_pred = self.predict(X)
        metrics = {}
        
        if is_classification:
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['f1'] = f1_score(y, y_pred, average='weighted')
            metrics['precision'] = precision_score(y, y_pred, average='weighted')
            metrics['recall'] = recall_score(y, y_pred, average='weighted')
            
            try:
                y_proba = self.predict_proba(X)
                if y_proba.shape[1] == 2:  # Binary classification
                    metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
            except (NotImplementedError, ValueError, AttributeError):
                logger.debug("AUC calculation failed, predict_proba may not be implemented")
        else:
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y, y_pred)
            
        # Log metrics
        self.log_metrics(metrics)
        
        return metrics
    
    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
                      cv: int = 5, is_classification: bool = True) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features
            y (Union[np.ndarray, pd.Series]): Ground truth labels/values
            cv (int): Number of cross-validation folds
            is_classification (bool): Whether this is a classification task
            
        Returns:
            Dict[str, float]: Dictionary with cross-validation metrics (mean and std)
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Select appropriate cross-validation strategy
        if is_classification:
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = 'neg_mean_squared_error'
            
        # This is a placeholder since we need the model's internal API for proper CV
        # Child classes should implement more comprehensive CV if needed
        fold_metrics = []
        
        for i, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Reset model state
            self.set_params(self.get_params())  # Reset with same parameters
            
            # Train on this fold
            self.train(X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate(X_val, y_val, is_classification=is_classification)
            
            logger.info(f"Fold {i+1}/{cv}: {metrics}")
            fold_metrics.append(metrics)
            
        # Compute mean and std of metrics
        result = {}
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            result[f"{metric}_mean"] = np.mean(values)
            result[f"{metric}_std"] = np.std(values)
            
        # Log aggregate metrics
        self.log_metrics(result)
            
        return result
    
    def search_hyperparams(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
                         param_space: Dict[str, Any], n_trials: int = 20, 
                         cv: int = 3, is_classification: bool = True,
                         metric: str = None) -> Dict[str, Any]:
        """
        Search for optimal hyperparameters.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features
            y (Union[np.ndarray, pd.Series]): Ground truth labels/values
            param_space (Dict[str, Any]): Parameter search space
            n_trials (int): Number of trials to run
            cv (int): Number of cross-validation folds
            is_classification (bool): Whether this is a classification task
            metric (str): Metric to optimize (if None, uses accuracy for classification, neg_mse for regression)
            
        Returns:
            Dict[str, Any]: Best parameters found
            
        Raises:
            ImportError: If optuna is not installed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter search. Install with: pip install optuna")
            
        logger.info(f"Starting hyperparameter search with {n_trials} trials")
        
        # Default metric based on problem type
        if metric is None:
            metric = 'accuracy' if is_classification else 'neg_mean_squared_error'
            
        # Define objective function for optimization
        def objective(trial):
            # Sample parameters from the search space
            trial_params = {}
            for param_name, param_spec in param_space.items():
                if isinstance(param_spec, tuple) and len(param_spec) == 3:
                    param_type, low, high = param_spec
                    if param_type == 'int':
                        trial_params[param_name] = trial.suggest_int(param_name, low, high)
                    elif param_type == 'float':
                        trial_params[param_name] = trial.suggest_float(param_name, low, high)
                    elif param_type == 'loguniform':
                        trial_params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                elif isinstance(param_spec, list):
                    trial_params[param_name] = trial.suggest_categorical(param_name, param_spec)
                else:
                    raise ValueError(f"Invalid parameter specification for {param_name}")
                    
            # Update model parameters
            current_params = self.get_params()
            self.set_params({**current_params, **trial_params})
            
            # Perform cross-validation
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
                
            scores = []
            for train_idx, val_idx in cv_splitter.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train on this fold
                self.set_params({**current_params, **trial_params})  # Reset with trial parameters
                self.train(X_train, y_train)
                
                # Evaluate based on the metric
                val_metrics = self.evaluate(X_val, y_val, is_classification=is_classification)
                scores.append(val_metrics.get(metric.replace('neg_', ''), 0))
                
            # Return mean score across folds
            mean_score = np.mean(scores)
            
            # For negative metrics (like neg_mean_squared_error), flip the sign for optuna
            if metric.startswith('neg_'):
                mean_score = -mean_score
                
            return mean_score
        
        # Create and run the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_value = study.best_value
        
        # Log results
        logger.info(f"Best {metric}: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric(f"best_{metric}", best_value)
            
        return best_params
    
    def __str__(self) -> str:
        """String representation of the model."""
        model_type = self.__class__.__name__
        return f"{model_type}(params={self.params}, fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__() 