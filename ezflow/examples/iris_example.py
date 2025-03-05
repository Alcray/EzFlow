from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

from ezflow.core.dataset.base import BaseDataset
from ezflow.core.pipeline.sklearn import SklearnPipelineWrapper
from ezflow.core.experiment.tracker import ExperimentTracker, ExperimentConfig

class IrisDataset(BaseDataset):
    """Simple dataset implementation for the Iris dataset."""
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: str = None
    ):
        super().__init__(
            name="iris",
            data_dir=data_dir,
            cache_dir=cache_dir
        )
        
    def load_data(self) -> None:
        """Load the Iris dataset."""
        # Load iris dataset
        iris = load_iris()
        
        # Convert to pandas DataFrame with a consistent index
        self.train_data = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=[*iris['feature_names'], 'target']
        ).reset_index(drop=True)
    
    def preprocess(self) -> None:
        """Preprocess the data (standardization is handled in pipeline)."""
        pass
    
    def get_features(self, data: pd.DataFrame) -> np.ndarray:
        """Get feature columns."""
        return data.iloc[:, :-1].values
    
    def get_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Get target column."""
        return data.iloc[:, -1].values

def run_iris_experiment():
    """Run a simple experiment with the Iris dataset."""
    
    # Create directories
    Path("experiments").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Initialize dataset
    dataset = IrisDataset(data_dir=".")
    dataset.load_data()
    dataset.split_data(val_size=0.2, random_state=42)
    dataset.preprocess()
    
    # Get features and labels
    X_train = dataset.get_features(dataset.train_data)
    y_train = dataset.get_labels(dataset.train_data)
    X_val = dataset.get_features(dataset.val_data)
    y_val = dataset.get_labels(dataset.val_data)
    
    # Create pipeline with preprocessing and model
    pipeline = SklearnPipelineWrapper([
        ("scaler", "sklearn.preprocessing.StandardScaler", {}),
        ("classifier", "sklearn.ensemble.RandomForestClassifier", {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        })
    ])
    
    # Create experiment config
    config = ExperimentConfig(
        name="iris_classification",
        model_params={
            "n_estimators": 100,
            "max_depth": 5
        },
        dataset_params={
            "val_size": 0.2
        },
        training_params={
            "random_state": 42
        }
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name="iris_example",
        artifacts_dir="experiments"
    )
    
    # Run experiment
    with tracker:
        # Start run
        tracker.start_run(config)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        # Log metrics
        tracker.log_metrics({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })
        
        # Save model
        pipeline.save("models/iris_rf.joblib")
        tracker.log_artifact("models/iris_rf.joblib")
        
        # Print results
        print("\nTraining Results:")
        print("-" * 50)
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("\nValidation Classification Report:")
        print("-" * 50)
        print(classification_report(y_val, val_preds, target_names=load_iris().target_names))

if __name__ == "__main__":
    run_iris_experiment() 