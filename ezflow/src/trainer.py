"""
Trainer module for the ezflow framework.

This module contains the ModelTrainer class for training, evaluating, and predicting with models.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Type

from ezflow.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    ModelTrainer class for training, evaluating, and predicting with models.
    
    This class handles the entire train-predict-evaluate lifecycle for a model.
    """
    
    def __init__(self, model: BaseModel, config: Dict):
        """
        Initialize the ModelTrainer with a model and configuration.
        
        Args:
            model (BaseModel): Model instance.
            config (Dict): Configuration dictionary.
        """
        self.model = model
        self.config = config
        self.metrics_history = []
        
        logger.info(f"ModelTrainer initialized with {type(model).__name__}")
    
    def train(self, X_train: Union[np.ndarray, pd.DataFrame], 
             y_train: Union[np.ndarray, pd.Series],
             X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
             y_val: Optional[Union[np.ndarray, pd.Series]] = None) -> None:
        """
        Train the model on the given data.
        
        Args:
            X_train (Union[np.ndarray, pd.DataFrame]): Training features.
            y_train (Union[np.ndarray, pd.Series]): Training target values.
            X_val (Optional[Union[np.ndarray, pd.DataFrame]]): Validation features.
            y_val (Optional[Union[np.ndarray, pd.Series]]): Validation target values.
        """
        logger.info("Starting model training...")
        
        # Prepare validation data if provided
        eval_set = None
        early_stopping_rounds = None
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            early_stopping_rounds = self.config.get('EARLY_STOPPING_ROUNDS', 10)
            logger.info(f"Using validation set for early stopping with {early_stopping_rounds} rounds")
        
        # Train with or without validation data
        if eval_set is not None:
            self.model.train(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)
        else:
            self.model.train(X_train, y_train)
        
        logger.info("Model training completed")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            
        Returns:
            np.ndarray: Predictions.
        """
        logger.info("Making predictions...")
        
        predictions = self.model.predict(X)
        
        logger.info(f"Predictions made with shape {predictions.shape}")
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probability predictions with the trained model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            
        Returns:
            np.ndarray: Probability predictions.
        """
        try:
            logger.info("Making probability predictions...")
            
            proba_predictions = self.model.predict_proba(X)
            
            logger.info(f"Probability predictions made with shape {proba_predictions.shape}")
            return proba_predictions
        except (ValueError, NotImplementedError) as e:
            logger.error(f"Could not make probability predictions: {str(e)}")
            raise
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series], 
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
            metrics (Optional[List[str]]): List of metrics to calculate.
            
        Returns:
            Dict[str, float]: Dictionary of metric name to value.
        """
        from sklearn import metrics as sk_metrics
        
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Convert y to numpy if needed
        if isinstance(y, pd.Series):
            y = y.values
        
        # Default metrics
        if metrics is None:
            # Auto-detect if classification or regression
            unique_values = np.unique(y)
            if len(unique_values) <= 10:  # Classification (arbitrary threshold)
                metrics = ['accuracy', 'precision', 'recall', 'f1']
            else:  # Regression
                metrics = ['mse', 'rmse', 'mae', 'r2']
        
        # Calculate metrics
        results = {}
        
        try:
            if 'accuracy' in metrics:
                results['accuracy'] = sk_metrics.accuracy_score(y, y_pred)
            
            if 'precision' in metrics:
                results['precision'] = sk_metrics.precision_score(y, y_pred, average='weighted')
            
            if 'recall' in metrics:
                results['recall'] = sk_metrics.recall_score(y, y_pred, average='weighted')
            
            if 'f1' in metrics:
                results['f1'] = sk_metrics.f1_score(y, y_pred, average='weighted')
            
            if 'roc_auc' in metrics:
                # Only for binary classification
                if len(np.unique(y)) == 2:
                    try:
                        y_proba = self.predict_proba(X)
                        # Get probabilities for positive class
                        if y_proba.shape[1] == 2:  # Binary classification
                            results['roc_auc'] = sk_metrics.roc_auc_score(y, y_proba[:, 1])
                    except (ValueError, NotImplementedError):
                        logger.warning("Could not calculate ROC AUC score: predict_proba not available")
            
            if 'mse' in metrics:
                results['mse'] = sk_metrics.mean_squared_error(y, y_pred)
            
            if 'rmse' in metrics:
                results['rmse'] = np.sqrt(sk_metrics.mean_squared_error(y, y_pred))
            
            if 'mae' in metrics:
                results['mae'] = sk_metrics.mean_absolute_error(y, y_pred)
            
            if 'r2' in metrics:
                results['r2'] = sk_metrics.r2_score(y, y_pred)
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
        
        # Log results
        logger.info(f"Evaluation results: {results}")
        
        # Store metrics history
        self.metrics_history.append(results)
        
        return results
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model.
        """
        logger.info(f"Saving model to {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to load the model from.
        """
        logger.info(f"Loading model from {path}")
        
        # Load the model
        self.model.load(path)
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importances from the model if supported.
        
        Returns:
            Optional[pd.DataFrame]: Feature importances or None if not supported.
        """
        try:
            return self.model.feature_importance()
        except (ValueError, NotImplementedError) as e:
            logger.warning(f"Feature importance not available: {str(e)}")
            return None
    
    def cross_validate(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        n_splits: int = 5,
        metrics: Optional[List[str]] = None,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the model.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
            n_splits (int): Number of folds.
            metrics (Optional[List[str]]): List of metrics to calculate.
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            Dict[str, List[float]]: Dictionary of metric name to list of values for each fold.
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        # Create empty results dictionary
        cv_results = {}
        
        # Auto-detect if classification or regression
        unique_values = np.unique(y)
        is_classification = len(unique_values) <= 10  # Arbitrary threshold
        
        # Default metrics
        if metrics is None:
            if is_classification:
                metrics = ['accuracy', 'precision', 'recall', 'f1']
            else:
                metrics = ['mse', 'rmse', 'mae', 'r2']
        
        # Initialize results for each metric
        for metric in metrics:
            cv_results[metric] = []
        
        # Choose appropriate CV splitter
        if is_classification:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            # Split data
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            
            if isinstance(y, pd.Series):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Train the model on this fold
            # Create a fresh model instance to avoid data leakage
            model_class = type(self.model)
            model_params = self.model.get_params()
            fold_model = model_class(model_params)
            fold_trainer = ModelTrainer(fold_model, self.config)
            
            # Train and evaluate
            fold_trainer.train(X_train, y_train)
            fold_results = fold_trainer.evaluate(X_val, y_val, metrics)
            
            # Store results for this fold
            for metric, value in fold_results.items():
                cv_results[metric].append(value)
        
        # Calculate mean and std for each metric
        cv_summary = {}
        for metric, values in cv_results.items():
            cv_summary[f"{metric}_mean"] = np.mean(values)
            cv_summary[f"{metric}_std"] = np.std(values)
        
        # Log summary
        logger.info(f"Cross-validation results: {cv_summary}")
        
        return cv_results 