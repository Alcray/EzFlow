"""
XGBoost model implementation for the ezflow framework.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from ezflow.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """
    XGBoost model for tabular data.
    Handles data from manifest.jsonl where each key-value pair is treated as a feature,
    except for the target key which is used as the label.
    """
    
    def __init__(self, params: Dict[str, Any], problem_type: str = 'classification', target_key: str = "target"):
        """
        Initialize XGBoost model.
        
        Args:
            params (Dict[str, Any]): XGBoost parameters
            problem_type (str): Problem type ('classification' or 'regression')
            target_key (str): Key in manifest.jsonl that contains the target value
        """
        super().__init__(params)
        self.target_key = target_key
        self.problem_type = problem_type
        
        # Default parameters for XGBoost based on problem type
        if problem_type == 'classification':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': 42
            }
        else:  # regression
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        
        # Update default params with provided params
        self.params = {**default_params, **params}
        
        # Initialize model based on problem type
        if problem_type == 'classification':
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
            
        self.feature_names = None
    
    def load_data(self, manifest_path: str) -> tuple:
        """
        Load and prepare data from manifest.jsonl.
        
        Args:
            manifest_path (str): Path to manifest.jsonl file
            
        Returns:
            tuple: (X, y) where X is features DataFrame and y is target Series
        """
        # Read manifest file
        data = []
        with open(manifest_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Separate features and target
        if self.target_key not in df.columns:
            raise ValueError(f"Target key '{self.target_key}' not found in manifest")
        
        y = df[self.target_key]
        X = df.drop(columns=[self.target_key])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X: Union[np.ndarray, pd.DataFrame, str], 
             y: Optional[Union[np.ndarray, pd.Series]] = None,
             eval_set: Optional[List[tuple]] = None) -> None:
        """
        Train the model on data.
        
        Args:
            X: Features or path to manifest.jsonl
            y: Target values (ignored if X is a manifest path)
            eval_set: Optional evaluation set for early stopping
        """
        # Start tracking run
        self.start_run(experiment_name=f"xgboost_{self.problem_type}")
        
        # Log training parameters
        logger.info(f"Training XGBoost model with params: {self.params}")
        
        # Handle manifest path input
        if isinstance(X, str) and X.endswith('.jsonl'):
            logger.info(f"Loading training data from manifest: {X}")
            X_train, y_train = self.load_data(X)
        else:
            # Use provided X and y directly
            X_train, y_train = X, y
            
            if y is None:
                raise ValueError("Target values (y) must be provided when X is not a manifest path")
        
        # Check data types and convert to DataFrame if needed
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
            
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Additional setup for classification problems
        if self.problem_type == 'classification':
            # For multiclass classification
            if len(np.unique(y_train)) > 2:
                self.params['objective'] = 'multi:softprob'
                self.params['num_class'] = len(np.unique(y_train))
                
                # Reinitialize the model with updated params
                self.model = xgb.XGBClassifier(**self.params)
        
        # Setup for training with eval set if provided
        fit_params = {}
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['eval_metric'] = 'logloss' if self.problem_type == 'classification' else 'rmse'
            fit_params['early_stopping_rounds'] = 10
            fit_params['verbose'] = True
            
        # Train the model with callback for metric tracking
        class MetricCallback(xgb.callback.TrainingCallback):
            def __init__(self, model_instance):
                self.model = model_instance
                
            def after_iteration(self, model, epoch, evals_log):
                # Extract metrics from evals_log and log them
                metrics = {}
                for data_name, metric_values in evals_log.items():
                    for metric_name, values in metric_values.items():
                        metrics[f"{data_name}_{metric_name}"] = values[-1]
                
                # Log metrics using our tracking system
                self.model.log_metrics(metrics, step=epoch)
                return False
                
        # Add callback
        fit_params['callbacks'] = [MetricCallback(self)]
            
        # Fit the model
        logger.info("Starting model training...")
        self.model.fit(X_train, y_train, **fit_params)
        
        self.is_fitted = True
        logger.info("Training completed successfully")
        
        # Plot feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = self.feature_importance()
            
            # Create plot directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png')
            
            # End the tracking run
            self.end_run()
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, str]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features or path to manifest.jsonl
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
        
        # Handle manifest path input
        if isinstance(X, str) and X.endswith('.jsonl'):
            X_test, _ = self.load_data(X)
        else:
            # Use X directly
            X_test = X
            
        # Check data type and convert to DataFrame if needed
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
            
        # Ensure all feature columns are present and in the right order
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X_test.columns)
            if missing_features:
                raise ValueError(f"Missing features in prediction data: {missing_features}")
            
            # Reorder columns to match training data
            X_test = X_test[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, str]) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Features or path to manifest.jsonl
            
        Returns:
            np.ndarray: Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
            
        if self.problem_type != 'classification':
            raise NotImplementedError("predict_proba is only available for classification problems")
        
        # Handle manifest path input
        if isinstance(X, str) and X.endswith('.jsonl'):
            X_test, _ = self.load_data(X)
        else:
            # Use X directly
            X_test = X
            
        # Check data type and convert to DataFrame if needed
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
            
        # Ensure all feature columns are present and in the right order
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X_test.columns)
            if missing_features:
                raise ValueError(f"Missing features in prediction data: {missing_features}")
            
            # Reorder columns to match training data
            X_test = X_test[self.feature_names]
        
        # Make probability predictions
        proba_predictions = self.model.predict_proba(X_test)
        
        return proba_predictions
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if not self.is_fitted:
            logger.warning("Saving an untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'target_key': self.target_key,
            'problem_type': self.problem_type,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model and metadata
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.target_key = model_data['target_key']
        self.problem_type = model_data.get('problem_type', 'classification')  # Default for backward compatibility
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {path}")
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances.
        
        Returns:
            pd.DataFrame: Feature importances sorted by importance
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first")
            
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("This model doesn't support feature importance")
        
        # Get feature importances
        importance = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    @staticmethod
    def get_default_params(problem_type: str = 'classification') -> Dict[str, Any]:
        """
        Get default parameters for XGBoost.
        
        Args:
            problem_type (str): Problem type ('classification' or 'regression')
            
        Returns:
            Dict[str, Any]: Default parameters
        """
        if problem_type == 'classification':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': 42
            }
        else:
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
    
    @staticmethod
    def get_param_search_space(problem_type: str = 'classification') -> Dict[str, Any]:
        """
        Get hyperparameter search space for XGBoost.
        
        Args:
            problem_type (str): Problem type ('classification' or 'regression')
            
        Returns:
            Dict[str, Any]: Hyperparameter search space
        """
        # Common hyperparameters for both classification and regression
        param_space = {
            'n_estimators': ('int', 50, 500),
            'learning_rate': ('loguniform', 0.01, 0.3),
            'max_depth': ('int', 3, 10),
            'min_child_weight': ('int', 1, 10),
            'subsample': ('float', 0.5, 1.0),
            'colsample_bytree': ('float', 0.5, 1.0),
            'gamma': ('float', 0, 5)
        }
        
        # Problem-specific hyperparameters
        if problem_type == 'classification':
            param_space.update({
                'scale_pos_weight': ('float', 0.1, 10),
            })
        
        return param_space 