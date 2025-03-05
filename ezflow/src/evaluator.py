"""
Evaluator module for the ezflow framework.

This module contains the Evaluator class for calculating metrics and generating visualizations.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Any, Tuple
from sklearn import metrics

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluator class for calculating metrics and generating visualizations.
    
    This class provides methods for evaluating model performance and visualizing results.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Evaluator with configuration.
        
        Args:
            config (Dict): Configuration dictionary.
        """
        self.config = config
        self.results = {}
        
        # Default metrics
        self.classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        self.regression_metrics = ['mse', 'rmse', 'mae', 'r2']
        
        # Get configured metrics or use defaults
        self.metrics = config.get('METRICS', [])
        
        logger.info("Evaluator initialized")
    
    def calculate_metrics(
        self, 
        y_true: Union[np.ndarray, pd.Series], 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metrics_list: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (Union[np.ndarray, pd.Series]): True target values.
            y_pred (np.ndarray): Predicted target values.
            y_proba (Optional[np.ndarray]): Predicted probabilities for classification.
            metrics_list (Optional[List[str]]): List of metrics to calculate.
            
        Returns:
            Dict[str, float]: Dictionary of metric name to value.
        """
        logger.info("Calculating evaluation metrics...")
        
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        # Use provided metrics list or default from config
        metrics_list = metrics_list or self.metrics
        
        # Auto-detect classification or regression if no metrics provided
        if not metrics_list:
            # Heuristic: if fewer than 10 unique values, assume classification
            unique_values = np.unique(y_true)
            is_classification = len(unique_values) <= 10
            
            metrics_list = self.classification_metrics if is_classification else self.regression_metrics
            logger.info(f"Auto-detected {'classification' if is_classification else 'regression'} task")
        
        results = {}
        
        # Calculate classification metrics
        if 'accuracy' in metrics_list:
            results['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        
        if 'precision' in metrics_list:
            try:
                results['precision'] = metrics.precision_score(y_true, y_pred, average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate precision score: {str(e)}")
        
        if 'recall' in metrics_list:
            try:
                results['recall'] = metrics.recall_score(y_true, y_pred, average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate recall score: {str(e)}")
        
        if 'f1' in metrics_list:
            try:
                results['f1'] = metrics.f1_score(y_true, y_pred, average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate F1 score: {str(e)}")
        
        if 'roc_auc' in metrics_list and y_proba is not None:
            try:
                # Binary classification
                if len(np.unique(y_true)) == 2 and y_proba.shape[1] == 2:
                    results['roc_auc'] = metrics.roc_auc_score(y_true, y_proba[:, 1])
                # Multi-class classification
                elif y_proba.shape[1] > 2:
                    results['roc_auc'] = metrics.roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC score: {str(e)}")
        
        # Calculate regression metrics
        if 'mse' in metrics_list:
            results['mse'] = metrics.mean_squared_error(y_true, y_pred)
        
        if 'rmse' in metrics_list:
            results['rmse'] = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        
        if 'mae' in metrics_list:
            results['mae'] = metrics.mean_absolute_error(y_true, y_pred)
        
        if 'r2' in metrics_list:
            results['r2'] = metrics.r2_score(y_true, y_pred)
        
        logger.info(f"Calculated metrics: {results}")
        
        # Store results
        self.results = results
        
        return results
    
    def confusion_matrix(
        self, 
        y_true: Union[np.ndarray, pd.Series], 
        y_pred: np.ndarray,
        normalize: bool = False,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate and optionally plot the confusion matrix.
        
        Args:
            y_true (Union[np.ndarray, pd.Series]): True target values.
            y_pred (np.ndarray): Predicted target values.
            normalize (bool): Whether to normalize the confusion matrix.
            save_path (Optional[str]): Path to save the confusion matrix plot.
            
        Returns:
            np.ndarray: Confusion matrix.
        """
        logger.info("Calculating confusion matrix...")
        
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        # Calculate confusion matrix
        cm = metrics.confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            logger.info("Normalized confusion matrix")
        
        # Plot confusion matrix if save_path is provided
        if save_path:
            plt.figure(figsize=(10, 8))
            
            # Get unique classes
            classes = np.unique(np.concatenate([y_true, y_pred]))
            
            # Plot with seaborn
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='.2f' if normalize else 'd', 
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes
            )
            
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title('Confusion Matrix')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save plot
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        return cm
    
    def roc_curve(
        self, 
        y_true: Union[np.ndarray, pd.Series], 
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate and optionally plot the ROC curve.
        
        Args:
            y_true (Union[np.ndarray, pd.Series]): True target values.
            y_proba (np.ndarray): Predicted probabilities.
            save_path (Optional[str]): Path to save the ROC curve plot.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (fpr, tpr, thresholds)
        """
        logger.info("Calculating ROC curve...")
        
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        # For binary classification
        if len(np.unique(y_true)) == 2:
            # Get probability of positive class
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba)
            roc_auc = metrics.auc(fpr, tpr)
            
            # Plot ROC curve if save_path is provided
            if save_path:
                plt.figure(figsize=(10, 8))
                
                plt.plot(
                    fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (area = {roc_auc:.2f})'
                )
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save plot
                plt.savefig(save_path)
                plt.close()
                
                logger.info(f"ROC curve plot saved to {save_path}")
            
            return fpr, tpr, thresholds
        else:
            logger.warning("ROC curve is only supported for binary classification")
            return None, None, None
    
    def feature_importance_plot(
        self, 
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importances.
        
        Args:
            importance_df (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns.
            top_n (int): Number of top features to plot.
            save_path (Optional[str]): Path to save the feature importance plot.
        """
        logger.info(f"Plotting top {top_n} feature importances...")
        
        # Validate input
        if 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
            logger.error("importance_df must have 'Feature' and 'Importance' columns")
            return
        
        # Get top N features
        top_features = importance_df.head(top_n).copy()
        
        # Sort by importance (ascending for horizontal bar plot)
        top_features = top_features.sort_values('Importance')
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        
        # Add values at the end of each bar
        for i, importance in enumerate(top_features['Importance']):
            plt.text(importance, i, f' {importance:.3f}')
        
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save plot
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_regression_results(
        self, 
        y_true: Union[np.ndarray, pd.Series], 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot regression results.
        
        Args:
            y_true (Union[np.ndarray, pd.Series]): True target values.
            y_pred (np.ndarray): Predicted target values.
            save_path (Optional[str]): Path to save the regression results plot.
        """
        logger.info("Plotting regression results...")
        
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        plt.figure(figsize=(10, 8))
        
        # Plot actual vs predicted values
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line (y = x)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        
        # Add metrics text
        if self.results:
            text = []
            for metric in ['mse', 'rmse', 'mae', 'r2']:
                if metric in self.results:
                    text.append(f"{metric.upper()}: {self.results[metric]:.3f}")
            
            if text:
                plt.annotate(
                    '\n'.join(text),
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8)
                )
        
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save plot
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Regression results plot saved to {save_path}")
        else:
            plt.show()
    
    def save_metrics(self, save_path: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            save_path (str): Path to save the metrics.
        """
        import json
        
        logger.info(f"Saving metrics to {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save metrics
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        logger.info(f"Metrics saved to {save_path}")
    
    def print_classification_report(
        self, 
        y_true: Union[np.ndarray, pd.Series], 
        y_pred: np.ndarray
    ) -> str:
        """
        Print a classification report.
        
        Args:
            y_true (Union[np.ndarray, pd.Series]): True target values.
            y_pred (np.ndarray): Predicted target values.
            
        Returns:
            str: Classification report.
        """
        logger.info("Generating classification report...")
        
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        report = metrics.classification_report(y_true, y_pred)
        
        print(report)
        
        return report 