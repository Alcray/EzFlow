"""
Hyperparameter tuning module for the ezflow framework.

This module contains the HyperparamTuner class for optimizing model parameters.
It supports various hyperparameter tuning frameworks like Optuna and Hyperopt.
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, Type, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Optional imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

from ezflow.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class HyperparamTuner:
    """
    HyperparamTuner class for optimizing model parameters.
    
    This class provides methods for hyperparameter tuning using various frameworks.
    """
    
    def __init__(self, model_class: Type[BaseModel], config: Dict[str, Any]):
        """
        Initialize the HyperparamTuner with a model class and configuration.
        
        Args:
            model_class (Type[BaseModel]): Class of the model to tune.
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.model_class = model_class
        self.config = config
        self.tuning_config = config.get('HYPERPARAMETER_TUNING', {})
        self.best_params = {}
        self.best_score = None
        self.best_trial = None
        
        # Get tuning method
        self.method = self.tuning_config.get('method', 'optuna').lower()
        
        # Validate tuning method
        if self.method == 'optuna' and not OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed. Please install it with 'pip install optuna'")
            raise ImportError("Optuna not installed")
        elif self.method == 'hyperopt' and not HYPEROPT_AVAILABLE:
            logger.warning("Hyperopt not installed. Please install it with 'pip install hyperopt'")
            raise ImportError("Hyperopt not installed")
        elif self.method not in ['optuna', 'hyperopt', 'grid_search', 'random_search']:
            raise ValueError(f"Unsupported tuning method: {self.method}")
        
        logger.info(f"HyperparamTuner initialized with method: {self.method}")
    
    def _get_param_space(self, tuner_type: str) -> Dict[str, Any]:
        """
        Get parameter space for the tuner.
        
        Args:
            tuner_type (str): Type of tuner ('optuna', 'hyperopt', 'sklearn').
            
        Returns:
            Dict[str, Any]: Parameter space for the tuner.
        """
        param_space = self.tuning_config.get('param_space', {})
        
        if not param_space:
            raise ValueError("Parameter space not defined in configuration")
        
        # Convert parameter space to the format required by the tuner
        if tuner_type == 'optuna':
            # Optuna param space is created in the objective function
            return param_space
        elif tuner_type == 'hyperopt':
            hyperopt_space = {}
            
            for param_name, param_config in param_space.items():
                if 'low' in param_config and 'high' in param_config:
                    if param_config.get('type') == 'int' or 'step' in param_config:
                        # Integer parameter
                        step = param_config.get('step', 1)
                        hyperopt_space[param_name] = hp.quniform(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            step
                        )
                    elif param_config.get('log', False):
                        # Log-uniform parameter
                        hyperopt_space[param_name] = hp.loguniform(
                            param_name,
                            np.log(param_config['low']),
                            np.log(param_config['high'])
                        )
                    else:
                        # Uniform parameter
                        hyperopt_space[param_name] = hp.uniform(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                elif 'choices' in param_config:
                    # Categorical parameter
                    hyperopt_space[param_name] = hp.choice(
                        param_name,
                        param_config['choices']
                    )
            
            return hyperopt_space
        elif tuner_type == 'sklearn':
            sklearn_space = {}
            
            for param_name, param_config in param_space.items():
                if 'low' in param_config and 'high' in param_config and 'step' in param_config:
                    # Integer or float parameter with step
                    sklearn_space[param_name] = np.arange(
                        param_config['low'],
                        param_config['high'] + param_config['step'],
                        param_config['step']
                    )
                elif 'low' in param_config and 'high' in param_config:
                    # Continuous parameter
                    if param_config.get('type') == 'int':
                        # Integer parameter
                        sklearn_space[param_name] = np.arange(
                            param_config['low'],
                            param_config['high'] + 1,
                            1
                        )
                    else:
                        # Float parameter
                        sklearn_space[param_name] = np.linspace(
                            param_config['low'],
                            param_config['high'],
                            10  # Default number of values
                        )
                elif 'choices' in param_config:
                    # Categorical parameter
                    sklearn_space[param_name] = param_config['choices']
            
            return sklearn_space
        
        return param_space
    
    def _create_model(self, params: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance with the given parameters.
        
        Args:
            params (Dict[str, Any]): Model parameters.
            
        Returns:
            BaseModel: Model instance.
        """
        # Get default parameters
        default_params = self.config.get('MODEL', {}).get('params', {})
        
        # Update with tuning parameters
        model_params = {**default_params, **params}
        
        # Create model instance
        model = self.model_class(params=model_params)
        
        return model
    
    def _objective_optuna(self, trial: 'optuna.Trial', 
                         X: Union[np.ndarray, pd.DataFrame], 
                         y: Union[np.ndarray, pd.Series],
                         cv: int = 5) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial (optuna.Trial): Optuna trial.
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
            cv (int): Number of cross-validation folds.
            
        Returns:
            float: Negative mean cross-validation score.
        """
        param_space = self.tuning_config.get('param_space', {})
        params = {}
        
        # Create parameters for this trial
        for param_name, param_config in param_space.items():
            if 'low' in param_config and 'high' in param_config:
                if param_config.get('type') == 'int' or 'step' in param_config:
                    # Integer parameter
                    step = param_config.get('step', 1)
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=step
                    )
                elif param_config.get('log', False):
                    # Log-uniform parameter
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=True
                    )
                else:
                    # Uniform parameter
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            elif 'choices' in param_config:
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        # Create model with these parameters
        model = self._create_model(params)
        
        # Choose appropriate CV method
        if len(np.unique(y)) <= 10:  # Classification task (arbitrary threshold)
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:  # Regression task
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Define the evaluation metric
        metric = self.tuning_config.get('metric', 'accuracy')
        
        # Perform cross-validation
        scores = []
        
        for train_idx, val_idx in cv_obj.split(X, y):
            # Split data
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            
            if isinstance(y, pd.Series):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.train(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate score based on the metric
            from sklearn import metrics as sk_metrics
            
            if metric == 'accuracy':
                score = sk_metrics.accuracy_score(y_val, y_pred)
            elif metric == 'f1':
                score = sk_metrics.f1_score(y_val, y_pred, average='weighted')
            elif metric == 'roc_auc':
                try:
                    y_proba = model.predict_proba(X_val)
                    if y_proba.shape[1] == 2:  # Binary classification
                        score = sk_metrics.roc_auc_score(y_val, y_proba[:, 1])
                    else:  # Multi-class
                        score = sk_metrics.roc_auc_score(
                            y_val, y_proba, multi_class='ovr', average='weighted'
                        )
                except (ValueError, NotImplementedError):
                    score = 0.0  # Fallback if predict_proba not available
            elif metric == 'mse':
                score = -sk_metrics.mean_squared_error(y_val, y_pred)  # Negative because we maximize
            elif metric == 'rmse':
                score = -np.sqrt(sk_metrics.mean_squared_error(y_val, y_pred))  # Negative because we maximize
            elif metric == 'mae':
                score = -sk_metrics.mean_absolute_error(y_val, y_pred)  # Negative because we maximize
            elif metric == 'r2':
                score = sk_metrics.r2_score(y_val, y_pred)
            else:
                score = 0.0  # Fallback for unknown metrics
            
            scores.append(score)
        
        # Return the mean score
        mean_score = np.mean(scores)
        
        # Log the score for this trial
        logger.info(f"Trial {trial.number} finished with parameters: {params} and score: {mean_score:.4f}")
        
        return mean_score
    
    def _objective_hyperopt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Objective function for Hyperopt.
        
        Args:
            params (Dict[str, Any]): Model parameters.
            
        Returns:
            Dict[str, Any]: Dictionary with loss and status.
        """
        # Get training data from instance variables
        X, y, cv = self.X, self.y, self.cv
        
        # Convert parameters
        # Some parameters might need to be converted (e.g., from float to int)
        param_space = self.tuning_config.get('param_space', {})
        for param_name, param_value in params.items():
            if param_name in param_space and param_space[param_name].get('type') == 'int':
                params[param_name] = int(param_value)
        
        # Create model with these parameters
        model = self._create_model(params)
        
        # Choose appropriate CV method
        if len(np.unique(y)) <= 10:  # Classification task (arbitrary threshold)
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:  # Regression task
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Define the evaluation metric
        metric = self.tuning_config.get('metric', 'accuracy')
        
        # Perform cross-validation
        scores = []
        
        for train_idx, val_idx in cv_obj.split(X, y):
            # Split data
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            
            if isinstance(y, pd.Series):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.train(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate score based on the metric
            from sklearn import metrics as sk_metrics
            
            if metric == 'accuracy':
                score = sk_metrics.accuracy_score(y_val, y_pred)
            elif metric == 'f1':
                score = sk_metrics.f1_score(y_val, y_pred, average='weighted')
            elif metric == 'roc_auc':
                try:
                    y_proba = model.predict_proba(X_val)
                    if y_proba.shape[1] == 2:  # Binary classification
                        score = sk_metrics.roc_auc_score(y_val, y_proba[:, 1])
                    else:  # Multi-class
                        score = sk_metrics.roc_auc_score(
                            y_val, y_proba, multi_class='ovr', average='weighted'
                        )
                except (ValueError, NotImplementedError):
                    score = 0.0  # Fallback if predict_proba not available
            elif metric == 'mse':
                score = -sk_metrics.mean_squared_error(y_val, y_pred)  # Negative because we maximize
            elif metric == 'rmse':
                score = -np.sqrt(sk_metrics.mean_squared_error(y_val, y_pred))  # Negative because we maximize
            elif metric == 'mae':
                score = -sk_metrics.mean_absolute_error(y_val, y_pred)  # Negative because we maximize
            elif metric == 'r2':
                score = sk_metrics.r2_score(y_val, y_pred)
            else:
                score = 0.0  # Fallback for unknown metrics
            
            scores.append(score)
        
        # Return the mean score
        mean_score = np.mean(scores)
        
        # Log the score for this trial
        logger.info(f"Trial finished with parameters: {params} and score: {mean_score:.4f}")
        
        # Hyperopt minimizes, so return negative score
        return {
            'loss': -mean_score,
            'status': STATUS_OK,
            'params': params,
            'score': mean_score
        }
    
    def _tune_optuna(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series],
                    cv: int = 5) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Dict[str, Any]: Best parameters.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed")
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        
        # Get number of trials and timeout
        n_trials = self.tuning_config.get('n_trials', 100)
        timeout = self.tuning_config.get('timeout', None)
        
        # Create objective function
        def objective(trial):
            return self._objective_optuna(trial, X, y, cv)
        
        # Run optimization
        logger.info(f"Starting Optuna optimization with {n_trials} trials and timeout {timeout} seconds")
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_trial = study.best_trial
        
        logger.info(f"Optuna optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def _tune_hyperopt(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      cv: int = 5) -> Dict[str, Any]:
        """
        Tune hyperparameters using Hyperopt.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Dict[str, Any]: Best parameters.
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt not installed")
        
        # Store data in instance variables (for the objective function)
        self.X, self.y, self.cv = X, y, cv
        
        # Get parameter space
        param_space = self._get_param_space('hyperopt')
        
        # Get number of trials and timeout
        n_trials = self.tuning_config.get('n_trials', 100)
        
        # Create trials object to store results
        trials = Trials()
        
        # Run optimization
        logger.info(f"Starting Hyperopt optimization with {n_trials} trials")
        
        best = fmin(
            fn=self._objective_hyperopt,
            space=param_space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials
        )
        
        # Convert parameters
        param_space_raw = self.tuning_config.get('param_space', {})
        for param_name, param_value in best.items():
            if param_name in param_space_raw and 'choices' in param_space_raw[param_name]:
                # For categorical parameters, hyperopt returns the index
                best[param_name] = param_space_raw[param_name]['choices'][int(param_value)]
        
        # Get best parameters
        self.best_params = best
        
        # Get best score
        trial_losses = [trial['result']['loss'] for trial in trials.trials]
        best_idx = np.argmin(trial_losses)
        self.best_score = -trial_losses[best_idx]
        self.best_trial = trials.trials[best_idx]
        
        logger.info(f"Hyperopt optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Clean up instance variables
        delattr(self, 'X')
        delattr(self, 'y')
        delattr(self, 'cv')
        
        return self.best_params
    
    def _tune_sklearn(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series],
                     cv: int = 5) -> Dict[str, Any]:
        """
        Tune hyperparameters using scikit-learn's grid search or random search.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Dict[str, Any]: Best parameters.
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        # Get parameter space
        param_space = self._get_param_space('sklearn')
        
        # Get default parameters
        default_params = self.config.get('MODEL', {}).get('params', {})
        
        # Create a wrapper estimator
        class ModelWrapper:
            def __init__(self, model_class, default_params):
                self.model_class = model_class
                self.default_params = default_params
                self.model = None
            
            def set_params(self, **params):
                return self
            
            def get_params(self, deep=True):
                return {param: None for param in param_space.keys()}
            
            def fit(self, X, y):
                # Combine default and tuning parameters
                params = {**self.default_params}
                
                # Update with parameters from GridSearchCV/RandomizedSearchCV
                for param in param_space.keys():
                    if hasattr(self, param):
                        params[param] = getattr(self, param)
                
                # Create and train model
                self.model = self.model_class(params=params)
                self.model.train(X, y)
                return self
            
            def predict(self, X):
                return self.model.predict(X)
            
            def score(self, X, y):
                # Define the evaluation metric
                metric = self.tuning_config.get('metric', 'accuracy')
                
                # Make predictions
                y_pred = self.model.predict(X)
                
                # Calculate score based on the metric
                from sklearn import metrics as sk_metrics
                
                if metric == 'accuracy':
                    return sk_metrics.accuracy_score(y, y_pred)
                elif metric == 'f1':
                    return sk_metrics.f1_score(y, y_pred, average='weighted')
                elif metric == 'roc_auc':
                    try:
                        y_proba = self.model.predict_proba(X)
                        if y_proba.shape[1] == 2:  # Binary classification
                            return sk_metrics.roc_auc_score(y, y_proba[:, 1])
                        else:  # Multi-class
                            return sk_metrics.roc_auc_score(
                                y, y_proba, multi_class='ovr', average='weighted'
                            )
                    except (ValueError, NotImplementedError):
                        return 0.0  # Fallback if predict_proba not available
                elif metric == 'mse':
                    return -sk_metrics.mean_squared_error(y, y_pred)  # Negative because we maximize
                elif metric == 'rmse':
                    return -np.sqrt(sk_metrics.mean_squared_error(y, y_pred))  # Negative because we maximize
                elif metric == 'mae':
                    return -sk_metrics.mean_absolute_error(y, y_pred)  # Negative because we maximize
                elif metric == 'r2':
                    return sk_metrics.r2_score(y, y_pred)
                else:
                    return 0.0  # Fallback for unknown metrics
        
        # Create estimator
        estimator = ModelWrapper(self.model_class, default_params)
        
        # Choose appropriate CV method
        if len(np.unique(y)) <= 10:  # Classification task (arbitrary threshold)
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:  # Regression task
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Get number of iterations
        n_iter = self.tuning_config.get('n_trials', 10)
        
        # Create search object
        if self.method == 'grid_search':
            search = GridSearchCV(
                estimator,
                param_space,
                cv=cv_obj,
                scoring=self.tuning_config.get('metric', 'accuracy'),
                n_jobs=-1,
                verbose=1
            )
        else:  # random_search
            search = RandomizedSearchCV(
                estimator,
                param_space,
                n_iter=n_iter,
                cv=cv_obj,
                scoring=self.tuning_config.get('metric', 'accuracy'),
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        # Run search
        logger.info(f"Starting {self.method} with {n_iter} iterations")
        
        search.fit(X, y)
        
        # Get best parameters
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        
        logger.info(f"{self.method} completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def tune(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            cv: int = 5) -> Dict[str, Any]:
        """
        Tune hyperparameters.
        
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features.
            y (Union[np.ndarray, pd.Series]): Target values.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Dict[str, Any]: Best parameters.
        """
        logger.info(f"Starting hyperparameter tuning with method: {self.method}")
        
        start_time = time.time()
        
        # Call appropriate tuning method
        if self.method == 'optuna':
            best_params = self._tune_optuna(X, y, cv)
        elif self.method == 'hyperopt':
            best_params = self._tune_hyperopt(X, y, cv)
        elif self.method in ['grid_search', 'random_search']:
            best_params = self._tune_sklearn(X, y, cv)
        else:
            raise ValueError(f"Unsupported tuning method: {self.method}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Hyperparameter tuning completed in {execution_time:.2f} seconds")
        
        return best_params
    
    def create_best_model(self) -> BaseModel:
        """
        Create a model with the best hyperparameters.
        
        Returns:
            BaseModel: Model instance with best hyperparameters.
            
        Raises:
            ValueError: If tune() has not been called yet.
        """
        if not self.best_params:
            raise ValueError("No best parameters available. Call tune() first")
        
        return self._create_model(self.best_params)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get tuning results.
        
        Returns:
            Dict[str, Any]: Dictionary with tuning results.
        """
        return {
            'method': self.method,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial': self.best_trial
        } 