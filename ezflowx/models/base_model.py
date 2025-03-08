"""
Base model interface for the ezflowx framework.

This module defines the base interface for all models in the framework.
"""

from abc import ABC, abstractmethod
import os
import logging
import time
import json
import datetime
from typing import Dict, Any, Union, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    r2_score,
)
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
    Abstract base class for all models in the ezflowx framework.
    
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
        
        # By default, assume the target column is 'target'
        self.target_key = "target"
        
        self.experiment_id = None
        self.run_id = None
        self.feature_names = None

    # -------------------------------------------------------------------------
    #                            DATA PREP UTILITIES
    # -------------------------------------------------------------------------
    def _load_file_as_df(
        self, file_path: str, target_key: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Internal helper to load .jsonl or .csv data into a DataFrame, 
        split off the target column, and return (X, y).
        
        Args:
            file_path (str): Path to the data file (.jsonl or .csv).
            target_key (str): Name of the target column.
            
        Returns:
            (X, y): DataFrame (features), Series (target).
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".jsonl":
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(
                f"Unsupported file format {ext}. Only .jsonl or .csv are supported."
            )
        
        if target_key not in df.columns:
            raise ValueError(f"Target key '{target_key}' not found in {file_path}")

        y = df[target_key]
        X = df.drop(columns=[target_key])
        
        return X, y

    def _prepare_data(
        self,
        X_or_path: Union[str, pd.DataFrame, np.ndarray],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target_key: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Central utility to handle data ingestion. 
        If X_or_path is a string, assumes it's a .jsonl or .csv path.
        Otherwise, treats X_or_path as features (Pandas/DataFrame or NumPy array).
        
        Args:
            X_or_path (Union[str, pd.DataFrame, np.ndarray]): The input data or path.
            y (Optional[Union[np.ndarray, pd.Series]]): Targets, if X_or_path is not a file path.
            target_key (Optional[str]): Overrides the model's default self.target_key.
            
        Returns:
            (X, y): DataFrame of features, Series of targets.
        """
        # Determine which target column to use
        tkey = target_key if target_key else self.target_key
        
        if isinstance(X_or_path, str):
            # Load from file
            X, y_ = self._load_file_as_df(X_or_path, tkey)
            return X, y_
        else:
            # X_or_path is already data
            if y is None:
                raise ValueError(
                    "If X_or_path is not a filename, you must provide the y array."
                )
            # Convert X to DataFrame if necessary
            if not isinstance(X_or_path, pd.DataFrame):
                X = pd.DataFrame(X_or_path)
            else:
                X = X_or_path
            # Convert y to Series if necessary
            if not isinstance(y, pd.Series):
                y = pd.Series(y)

            return X, y

    # -------------------------------------------------------------------------
    #                            ABSTRACT METHODS
    # -------------------------------------------------------------------------
    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame, str],
              y: Optional[Union[np.ndarray, pd.Series]] = None) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame, str]) -> np.ndarray:
        """Predict with the model."""
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, str]) -> np.ndarray:
        """Predict probabilities (for classification)."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass

    # -------------------------------------------------------------------------
    #                      COMMON LOGGING & TRACKING METHODS
    # -------------------------------------------------------------------------
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
        """Get the model parameters."""
        return self.params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the model parameters (child classes can override)."""
        self.params = params

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to tracking system and internal history.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            step (Optional[int]): Step number for tracking
        """
        timestamp = time.time()
        metrics_with_time = {**metrics, "timestamp": timestamp}
        
        if step is not None:
            self.training_history[step] = metrics_with_time
        else:
            step = len(self.training_history)
            self.training_history[step] = metrics_with_time
            
        # Log to stdout
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != "timestamp"])
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
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            self.run_id = mlflow.active_run().info.run_id
            self.experiment_id = mlflow.active_run().info.experiment_id
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
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_artifact(save_path)
        else:
            plt.show()

    def evaluate(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first.")
            
        y_pred = self.predict(X)
        metrics = {}
        
        if is_classification:
            metrics["accuracy"] = accuracy_score(y, y_pred)
            metrics["f1"] = f1_score(y, y_pred, average="weighted")
            metrics["precision"] = precision_score(y, y_pred, average="weighted")
            metrics["recall"] = recall_score(y, y_pred, average="weighted")
            
            try:
                y_proba = self.predict_proba(X)
                if y_proba.shape[1] == 2:  # binary classification
                    metrics["auc"] = roc_auc_score(y, y_proba[:, 1])
            except (NotImplementedError, ValueError, AttributeError):
                logger.debug("AUC calculation failed; predict_proba not implemented or invalid output.")
        else:
            metrics["mse"] = mean_squared_error(y, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["r2"] = r2_score(y, y_pred)
            
        self.log_metrics(metrics)
        return metrics

    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv: int = 5,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        if is_classification:
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        fold_metrics = []
        
        for i, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.set_params(self.get_params())  # reset parameters
            self.train(X_train, y_train)
            metrics = self.evaluate(X_val, y_val, is_classification=is_classification)
            
            logger.info(f"Fold {i+1}/{cv} metrics: {metrics}")
            fold_metrics.append(metrics)
            
        result = {}
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            result[f"{metric}_mean"] = np.mean(values)
            result[f"{metric}_std"] = np.std(values)
        
        self.log_metrics(result)
        return result

    def search_hyperparams(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        param_space: Dict[str, Any],
        n_trials: int = 20,
        cv: int = 3,
        is_classification: bool = True,
        metric: str = None
    ) -> Dict[str, Any]:
        """
        Search for optimal hyperparameters using Optuna.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Please install via pip install optuna.")

        logger.info(f"Starting hyperparameter search with {n_trials} trials")

        if metric is None:
            metric = "accuracy" if is_classification else "neg_mean_squared_error"

        import optuna

        def objective(trial):
            trial_params = {}
            for param_name, param_spec in param_space.items():
                if isinstance(param_spec, tuple) and len(param_spec) == 3:
                    param_type, low, high = param_spec
                    if param_type == "int":
                        trial_params[param_name] = trial.suggest_int(param_name, low, high)
                    elif param_type == "float":
                        trial_params[param_name] = trial.suggest_float(param_name, low, high)
                    elif param_type == "loguniform":
                        trial_params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    else:
                        raise ValueError(f"Unsupported param_type: {param_type}")
                elif isinstance(param_spec, list):
                    # categorical
                    trial_params[param_name] = trial.suggest_categorical(param_name, param_spec)
                else:
                    raise ValueError(f"Invalid parameter specification for {param_name}")
            
            current_params = self.get_params()
            self.set_params({**current_params, **trial_params})

            # Cross-validation
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

            scores = []
            for train_idx, val_idx in cv_splitter.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                self.set_params({**current_params, **trial_params})
                self.train(X_train, y_train)
                val_metrics = self.evaluate(X_val, y_val, is_classification=is_classification)
                # retrieve the metric we want to optimize
                # if metric is "neg_mean_squared_error", we need to get "mse" from val_metrics 
                # then negate it
                if metric.startswith("neg_"):
                    # e.g. "neg_mean_squared_error" => we want -val_metrics["mse"]
                    raw_metric_name = metric.replace("neg_", "")
                    score_value = val_metrics.get(raw_metric_name, None)
                    if score_value is None:
                        score_value = 0
                    # negate because "neg_" means we want to maximize the negative MSE
                    score_value = -score_value
                else:
                    score_value = val_metrics.get(metric, None)
                    if score_value is None:
                        score_value = 0
                scores.append(score_value)

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_value = study.best_value
        logger.info(f"Best {metric}: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric(f"best_{metric}", best_value)

        return best_params

    def __str__(self) -> str:
        model_type = self.__class__.__name__
        return f"{model_type}(params={self.params}, fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        return self.__str__()
