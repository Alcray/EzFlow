"""
XGBoost model implementation for the ezflowx framework.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from ezflowx.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """
    XGBoost model for tabular data.
    Uses a universal data loading approach from BaseModel._prepare_data().
    """
    
    def __init__(
        self,
        params: Dict[str, Any],
        problem_type: str = "classification",
        target_key: str = "target",
    ):
        """
        Initialize XGBoost model.
        
        Args:
            params (Dict[str, Any]): XGBoost parameters.
            problem_type (str): 'classification' or 'regression'.
            target_key (str): Column/key for the target in the data.
        """
        super().__init__(params)
        self.target_key = target_key
        self.problem_type = problem_type
        
        if problem_type == "classification":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "random_state": 42,
            }
            self.params = {**default_params, **params}
            self.model = xgb.XGBClassifier(**self.params)
        else:
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": 42,
            }
            self.params = {**default_params, **params}
            self.model = xgb.XGBRegressor(**self.params)

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame, str],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        eval_set: Optional[List[tuple]] = None
    ) -> None:
        """
        Train the XGBoost model. 
        - If X is a filename (JSONL/CSV), it will be loaded automatically.
        - Otherwise, X and y must be data arrays/frames.
        """
        # start MLflow run
        self.start_run(experiment_name=f"xgboost_{self.problem_type}")
        
        # Use the base model's universal data loader
        X_train, y_train = self._prepare_data(X, y, target_key=self.target_key)

        # If classification with multiple classes, switch objective
        if self.problem_type == "classification":
            unique_labels = np.unique(y_train)
            if len(unique_labels) > 2:
                self.params["objective"] = "multi:softprob"
                self.params["num_class"] = len(unique_labels)
                # Re-initialize the underlying model
                self.model = xgb.XGBClassifier(**self.params)

        # Ensure X_train, y_train are DF/Series
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)

        # Save feature names for future checks
        self.feature_names = X_train.columns.tolist()

        # Prepare fit params for early stopping, etc.
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["eval_metric"] = (
                "logloss" if self.problem_type == "classification" else "rmse"
            )
            fit_params["early_stopping_rounds"] = 10
            fit_params["verbose"] = True

        # Callback for logging metrics
        class MetricCallback(xgb.callback.TrainingCallback):
            def __init__(self, model_instance):
                self.model = model_instance
                
            def after_iteration(self, model, epoch, evals_log):
                metrics = {}
                # evals_log looks like: {"validation_0": {"logloss": [val, val, ...], ...}}
                for data_name, metric_dict in evals_log.items():
                    for metric_name, values in metric_dict.items():
                        metrics[f"{data_name}_{metric_name}"] = values[-1]
                self.model.log_metrics(metrics, step=epoch)
                return False
        
        fit_params["callbacks"] = [MetricCallback(self)]

        logger.info(f"Training XGBoost with params: {self.params}")
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        logger.info("XGBoost model training complete.")

        # End run for MLflow
        self.end_run()

    def predict(self, X: Union[np.ndarray, pd.DataFrame, str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Let the base model handle data
        X_test, _ = self._prepare_data(X, None, target_key=self.target_key)
        
        # Reorder columns if needed
        if self.feature_names:
            missing = set(self.feature_names) - set(X_test.columns)
            if missing:
                raise ValueError(f"Missing features in input: {missing}")
            X_test = X_test[self.feature_names]

        return self.model.predict(X_test)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not trained. Call train() first.")
        if self.problem_type != "classification":
            raise NotImplementedError("predict_proba is only supported for classification.")
        
        X_test, _ = self._prepare_data(X, None, target_key=self.target_key)
        if self.feature_names:
            missing = set(self.feature_names) - set(X_test.columns)
            if missing:
                raise ValueError(f"Missing features in input: {missing}")
            X_test = X_test[self.feature_names]

        return self.model.predict_proba(X_test)

    def save(self, path: str) -> None:
        if not self.is_fitted:
            logger.warning("Saving an unfitted model.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "target_key": self.target_key,
            "problem_type": self.problem_type,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.params = model_data["params"]
        self.feature_names = model_data["feature_names"]
        self.target_key = model_data["target_key"]
        self.problem_type = model_data["problem_type"]
        self.is_fitted = model_data["is_fitted"]
        
        logger.info(f"Model loaded from {path}")

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not trained.")
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("This model doesn't support feature importances.")
        
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        })
        return df.sort_values("Importance", ascending=False).reset_index(drop=True)
    
    @staticmethod
    def get_default_params(problem_type: str = "classification") -> Dict[str, Any]:
        if problem_type == "classification":
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "random_state": 42,
            }
        else:
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": 42,
            }
    
    @staticmethod
    def get_param_search_space(problem_type: str = "classification") -> Dict[str, Any]:
        # an example search space
        param_space = {
            "n_estimators": ("int", 50, 500),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "max_depth": ("int", 3, 10),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "gamma": ("float", 0, 5),
        }
        if problem_type == "classification":
            param_space["scale_pos_weight"] = ("float", 0.1, 10)
        return param_space
