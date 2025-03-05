import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import json

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")


def plot_metric_comparison(experiment_dirs: List[str], metric_name: str, 
                           title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[str] = None) -> None:
    """
    Plot a comparison of the same metric across multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directory paths
        metric_name: Name of the metric to compare
        title: Optional title for the plot
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        metrics_path = os.path.join(exp_dir, "metrics.json")
        
        if not os.path.exists(metrics_path):
            print(f"Warning: No metrics found in {exp_dir}")
            continue
        
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
        
        if metric_name not in all_metrics:
            print(f"Warning: Metric '{metric_name}' not found in {exp_dir}")
            continue
        
        metric_data = all_metrics[metric_name]
        values = [entry["value"] for entry in metric_data]
        
        if "step" in metric_data[0]:
            steps = [entry["step"] for entry in metric_data]
            plt.plot(steps, values, marker='o', label=exp_name)
        else:
            plt.plot(values, marker='o', label=exp_name)
    
    plt.xlabel("Step" if "step" in metric_data[0] else "Measurement")
    plt.ylabel(metric_name)
    plt.title(title or f"Comparison of {metric_name} across experiments")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: Optional[List[str]] = None,
                          normalize: bool = False, figsize: Tuple[int, int] = (10, 8),
                          cmap: str = 'Blues', save_path: Optional[str] = None) -> None:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of the classes
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size as (width, height)
        cmap: Colormap for the plot
        save_path: Optional path to save the figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap=cmap, square=True, linewidths=.5, ax=ax)
    
    # Set labels
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names, rotation=45)
    ax.yaxis.set_ticklabels(class_names, rotation=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, 
                   class_names: Optional[List[str]] = None,
                   figsize: Tuple[int, int] = (10, 8),
                   save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve for binary or multiclass classification.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        class_names: Names of the classes
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Handle binary classification
    if y_score.ndim == 1 or y_score.shape[1] == 1 or y_score.shape[1] == 2:
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]  # Get the probability of the positive class
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    
    # Handle multiclass classification
    else:
        n_classes = y_score.shape[1]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            # Convert to one-vs-rest binary classification
            y_true_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, 
                     label=f'{class_names[i]} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray,
                               class_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> None:
    """
    Plot precision-recall curve for binary or multiclass classification.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        class_names: Names of the classes
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Handle binary classification
    if y_score.ndim == 1 or y_score.shape[1] == 1 or y_score.shape[1] == 2:
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]  # Get the probability of the positive class
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, lw=2, 
                 label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    
    # Handle multiclass classification
    else:
        n_classes = y_score.shape[1]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        
        # Compute precision-recall curve for each class
        for i in range(n_classes):
            # Convert to one-vs-rest binary classification
            y_true_bin = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_bin, y_score[:, i])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, lw=2, 
                     label=f'{class_names[i]} (area = {pr_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_feature_importance(feature_importances: np.ndarray, 
                           feature_names: Optional[List[str]] = None,
                           top_n: Optional[int] = None,
                           figsize: Tuple[int, int] = (12, 10),
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importances.
    
    Args:
        feature_importances: Array of feature importance values
        feature_names: Names of the features
        top_n: Number of top features to display
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_importances))]
    
    # Create DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Select top N features if specified
    if top_n is not None and top_n < len(importance_df):
        importance_df = importance_df.iloc[:top_n]
    
    plt.figure(figsize=figsize)
    
    # Create horizontal bar plot
    sns.barplot(x='importance', y='feature', data=importance_df)
    
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_learning_curve(train_sizes: np.ndarray, train_scores: np.ndarray, 
                        test_scores: np.ndarray, ylim: Optional[Tuple[float, float]] = None,
                        figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None) -> None:
    """
    Plot learning curve showing model performance as training set size increases.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores: Array of training scores for each size
        test_scores: Array of validation/test scores for each size
        ylim: Y-axis limits as (min, max)
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    if ylim is not None:
        plt.ylim(*ylim)
    
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def create_experiment_dashboard(experiment_dir: str, output_path: Optional[str] = None) -> str:
    """
    Create an HTML dashboard for visualizing experiment results.
    
    Args:
        experiment_dir: Path to the experiment directory
        output_path: Optional path to save the dashboard HTML file
        
    Returns:
        Path to the created HTML file
    """
    # Load experiment metadata and metrics
    metadata_path = os.path.join(experiment_dir, "metadata.json")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    
    if not os.path.exists(metadata_path) or not os.path.exists(metrics_path):
        raise ValueError(f"Metadata or metrics not found in {experiment_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create HTML content
    experiment_name = metadata['name']
    exp_id = metadata['id']
    
    if output_path is None:
        output_path = os.path.join(experiment_dir, "dashboard.html")
    
    # Create HTML file
    with open(output_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Dashboard - {experiment_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2C3E50;
        }}
        h2 {{
            color: #3498DB;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .artifact-link {{
            display: inline-block;
            background-color: #3498DB;
            color: white;
            padding: 5px 10px;
            text-decoration: none;
            border-radius: 4px;
            margin: 5px 0;
        }}
        .artifact-link:hover {{
            background-color: #2980B9;
        }}
        .log-entry {{
            padding: 5px;
            border-bottom: 1px solid #eee;
        }}
        .log-entry:nth-child(even) {{
            background-color: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Experiment: {experiment_name}</h1>
        <p><strong>ID:</strong> {exp_id}</p>
        <p><strong>Description:</strong> {metadata.get('description', 'No description provided')}</p>
        <p><strong>Tags:</strong> {', '.join(metadata.get('tags', []))}</p>
        <p><strong>Start Time:</strong> {metadata.get('start_time', 'Unknown')}</p>
        <p><strong>End Time:</strong> {metadata.get('end_time', 'Running') if metadata.get('end_time') else 'Running'}</p>
    </div>

    <div class="section">
        <h2>Parameters</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
""")
        
        # Add parameters
        params = metadata.get('params', {})
        for param_name, param_value in params.items():
            f.write(f"""
            <tr>
                <td>{param_name}</td>
                <td>{param_value}</td>
            </tr>
""")
        
        f.write("""
        </table>
    </div>

    <div class="section">
        <h2>Metrics</h2>
""")
        
        # Add metrics
        for metric_name, metric_data in metrics.items():
            if not metric_data:
                continue
                
            last_value = metric_data[-1]['value']
            if isinstance(last_value, list):
                last_value = "Array data"
            
            # Find best value based on metric name convention
            best_value = None
            if metric_data and len(metric_data) > 0:
                values = [entry['value'] for entry in metric_data if isinstance(entry['value'], (int, float))]
                if values:
                    if metric_name.startswith(('acc', 'auc', 'f1', 'precision', 'recall')):
                        best_value = max(values)
                    else:
                        best_value = min(values)
            
            # Create metric card
            f.write(f"""
            <div class="metric-card">
                <h3>{metric_name}</h3>
                <p><strong>Last Value:</strong> {last_value}</p>
""")
            if best_value is not None:
                f.write(f"""
                <p><strong>Best Value:</strong> {best_value}</p>
""")
            
            # Check if we have a plot for this metric
            plot_path = f"{metric_name}_plot.png"
            if plot_path in metadata.get('artifacts', {}):
                rel_path = os.path.relpath(metadata['artifacts'][plot_path], os.path.dirname(output_path))
                f.write(f"""
                <img src="{rel_path}" alt="{metric_name} plot" style="max-width: 100%;">
""")
            
            f.write("""
            </div>
""")
        
        f.write("""
    </div>

    <div class="section">
        <h2>Artifacts</h2>
""")
        
        # Add artifacts
        artifacts = metadata.get('artifacts', {})
        for artifact_name, artifact_path in artifacts.items():
            # Skip plots as they are already shown in metrics section
            if artifact_name.endswith('_plot'):
                continue
                
            rel_path = os.path.relpath(artifact_path, os.path.dirname(output_path))
            f.write(f"""
        <p><a href="{rel_path}" class="artifact-link">{artifact_name}</a></p>
""")
        
        f.write("""
    </div>

    <div class="section">
        <h2>Logs</h2>
        <div style="max-height: 400px; overflow-y: auto;">
""")
        
        # Add logs
        logs = metadata.get('logs', [])
        for log_entry in logs:
            f.write(f"""
            <div class="log-entry">
                <span>{log_entry.get('time', '')}</span>: {log_entry.get('message', '')}
            </div>
""")
        
        f.write("""
        </div>
    </div>
</body>
</html>
""")
    
    return output_path 