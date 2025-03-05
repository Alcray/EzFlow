import os
import pickle
import logging
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


class Model:
    """
    Model wrapper for tracking and managing machine learning models.
    
    This class provides a unified interface for training, evaluating, and making predictions
    with different types of machine learning models. It also handles model serialization,
    metric tracking, and feature importance.
    """
    
    def __init__(
        self, 
        model: BaseEstimator,
        name: Optional[str] = None,
        model_type: str = "classifier",
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model: The machine learning model (sklearn-compatible with fit/predict methods)
            name: Name of the model
            model_type: Type of model ('classifier' or 'regressor')
            feature_names: Names of the features
            class_names: Names of the classes (for classification models)
            metadata: Additional metadata to store with the model
        """
        self.model = model
        self.name = name or model.__class__.__name__
        self.model_type = model_type
        self.feature_names = feature_names
        self.class_names = class_names
        self.metadata = metadata or {}
        self.history = {}
        self._is_fitted = False
        
        logger.info(f"Initialized {self.model_type} model: {self.name}")
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'Model':
        """
        Fit the model to the training data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            Self
        """
        logger.info(f"Fitting model {self.name} to data with shape {X.shape}")
        fit_start_time = datetime.now()
        
        # Fit the model
        self.model.fit(X, y, **kwargs)
        
        fit_end_time = datetime.now()
        fit_duration = (fit_end_time - fit_start_time).total_seconds()
        
        # Update history and metadata
        self.history['fit_timestamp'] = fit_end_time.isoformat()
        self.history['fit_duration'] = fit_duration
        self.history['n_samples'] = X.shape[0]
        self.history['n_features'] = X.shape[1]
        
        if hasattr(self.model, 'feature_importances_'):
            self.history['feature_importances'] = self.model.feature_importances_.tolist()
        
        self._is_fitted = True
        logger.info(f"Model {self.name} fitted in {fit_duration:.2f} seconds")
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        logger.info(f"Making predictions with model {self.name} on data with shape {X.shape}")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities for classification models.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if self.model_type != "classifier":
            raise ValueError("predict_proba is only available for classification models")
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"Model {self.name} does not have predict_proba method")
        
        logger.info(f"Getting prediction probabilities with model {self.name} on data with shape {X.shape}")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before evaluating.")
        
        logger.info(f"Evaluating model {self.name} on data with shape {X.shape}")
        
        y_pred = self.predict(X)
        
        # Calculate metrics based on model type
        metrics = {}
        
        if self.model_type == "classifier":
            # For classification models
            metrics['accuracy'] = accuracy_score(y, y_pred)
            
            # Handle binary vs multiclass differently
            if len(np.unique(y)) == 2:
                # Binary classification
                metrics['precision'] = precision_score(y, y_pred, average='binary')
                metrics['recall'] = recall_score(y, y_pred, average='binary')
                metrics['f1'] = f1_score(y, y_pred, average='binary')
                
                # Calculate ROC AUC if predict_proba is available
                if hasattr(self.model, 'predict_proba'):
                    try:
                        y_proba = self.predict_proba(X)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y, y_proba)
                    except (IndexError, ValueError):
                        logger.warning("Could not calculate ROC AUC score")
            else:
                # Multiclass classification
                metrics['precision_macro'] = precision_score(y, y_pred, average='macro')
                metrics['recall_macro'] = recall_score(y, y_pred, average='macro')
                metrics['f1_macro'] = f1_score(y, y_pred, average='macro')
                
                # Weighted metrics
                metrics['precision_weighted'] = precision_score(y, y_pred, average='weighted')
                metrics['recall_weighted'] = recall_score(y, y_pred, average='weighted')
                metrics['f1_weighted'] = f1_score(y, y_pred, average='weighted')
        
        else:
            # For regression models
            metrics['mean_squared_error'] = mean_squared_error(y, y_pred)
            metrics['mean_absolute_error'] = mean_absolute_error(y, y_pred)
            metrics['r2'] = r2_score(y, y_pred)
            
            # RMSE
            metrics['root_mean_squared_error'] = np.sqrt(metrics['mean_squared_error'])
        
        # Save metrics to history
        if 'eval_metrics' not in self.history:
            self.history['eval_metrics'] = []
        
        self.history['eval_metrics'].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'n_samples': X.shape[0]
        })
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importances if available.
        
        Returns:
            Dictionary mapping feature names to importance values, or None if not available
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting feature importances.")
        
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # If feature names are provided, return a dictionary
            if self.feature_names is not None:
                if len(self.feature_names) != len(importances):
                    logger.warning(f"Feature importance length ({len(importances)}) " +
                                 f"doesn't match feature_names length ({len(self.feature_names)})")
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                else:
                    feature_names = self.feature_names
                
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
            
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
        
        # Check if model has coef_ attribute (linear models)
        elif hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            
            # Handle multiclass models (shape: n_classes, n_features)
            if coefs.ndim > 1:
                coefs = np.abs(coefs).mean(axis=0)  # Average absolute coefficients across classes
            
            # If feature names are provided, return a dictionary
            if self.feature_names is not None:
                if len(self.feature_names) != len(coefs):
                    logger.warning(f"Coefficient length ({len(coefs)}) " +
                                 f"doesn't match feature_names length ({len(self.feature_names)})")
                    feature_names = [f"feature_{i}" for i in range(len(coefs))]
                else:
                    feature_names = self.feature_names
                
                return {name: float(coef) for name, coef in zip(feature_names, coefs)}
            
            return {f"feature_{i}": float(coef) for i, coef in enumerate(coefs)}
        
        logger.warning(f"Model {self.name} does not support feature importance extraction")
        return None
    
    def save(self, directory: str, filename: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            directory: Directory to save the model
            filename: Optional filename (default: {model_name}_{timestamp}.pkl)
            
        Returns:
            Path to the saved model file
        """
        os.makedirs(directory, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_{timestamp}.pkl"
        
        file_path = os.path.join(directory, filename)
        
        # Prepare model data with metadata
        model_data = {
            'model': self.model,
            'name': self.name,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'metadata': self.metadata,
            'history': self.history,
            'is_fitted': self._is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model {self.name} saved to {file_path}")
        return file_path
    
    @classmethod
    def load(cls, file_path: str) -> 'Model':
        """
        Load a model from a file.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Loaded Model instance
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            model=model_data['model'],
            name=model_data['name'],
            model_type=model_data['model_type'],
            feature_names=model_data['feature_names'],
            class_names=model_data['class_names'],
            metadata=model_data['metadata']
        )
        
        model.history = model_data['history']
        model._is_fitted = model_data['is_fitted']
        
        logger.info(f"Loaded model {model.name} from {file_path}")
        return model
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report about the model.
        
        Returns:
            Dictionary containing model information and performance metrics
        """
        report = {
            'name': self.name,
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__,
            'is_fitted': self._is_fitted,
            'history': self.history,
            'metadata': self.metadata,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'n_classes': len(self.class_names) if self.class_names else None,
            'feature_importance': self.get_feature_importance() if self._is_fitted else None,
        }
        
        # Add model parameters if available
        if hasattr(self.model, 'get_params'):
            report['parameters'] = self.model.get_params()
        
        return report
    
    def generate_classification_report(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Generate a detailed classification report.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary with detailed classification metrics
        """
        if self.model_type != "classifier":
            raise ValueError("Classification report is only available for classification models")
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before generating a classification report.")
        
        y_pred = self.predict(X)
        
        # Generate the scikit-learn classification report
        target_names = self.class_names if self.class_names else None
        sklearn_report = classification_report(y, y_pred, target_names=target_names, output_dict=True)
        
        # Get confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Calculate ROC AUC for binary classification if predict_proba is available
        roc_auc = None
        if len(np.unique(y)) == 2 and hasattr(self.model, 'predict_proba'):
            try:
                y_proba = self.predict_proba(X)[:, 1]
                roc_auc = roc_auc_score(y, y_proba)
            except (IndexError, ValueError):
                logger.warning("Could not calculate ROC AUC score")
        
        # Combine results
        report = {
            'classification_report': sklearn_report,
            'confusion_matrix': cm.tolist(),
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc,
            'n_samples': X.shape[0],
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def get_cross_validation_results(self, cv_results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Process cross-validation results from sklearn's GridSearchCV or similar.
        
        Args:
            cv_results: Cross-validation results from GridSearchCV.cv_results_
            
        Returns:
            Processed cross-validation metrics
        """
        # Extract test scores
        test_scores = {}
        for key in cv_results.keys():
            if key.startswith('mean_test_'):
                metric_name = key.replace('mean_test_', '')
                test_scores[metric_name] = {
                    'mean': cv_results[key],
                    'std': cv_results[f'std_test_{metric_name}']
                }
        
        # Extract best parameters
        best_index = np.argmax(cv_results['mean_test_score'])
        param_names = [k for k in cv_results.keys() if k.startswith('param_')]
        best_params = {k.replace('param_', ''): cv_results[k][best_index] for k in param_names}
        
        # Create summary
        cv_summary = {
            'best_score': cv_results['mean_test_score'][best_index],
            'best_params': best_params,
            'test_scores': test_scores,
            'n_splits': cv_results['n_splits'][0] if 'n_splits' in cv_results else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return cv_summary 