from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        
        # Store feature names and target names for later use
        self.feature_names = iris['feature_names']
        self.target_names = iris['target_names']
    
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
    """Run a more comprehensive experiment with the Iris dataset using cross-validation."""
    
    # Create directories
    Path("experiments").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Initialize dataset
    dataset = IrisDataset(data_dir=".")
    dataset.load_data()
    dataset.preprocess()
    
    # Get all data
    X = dataset.get_features(dataset.train_data)
    y = dataset.get_labels(dataset.train_data)
    
    # Create pipeline with preprocessing and model
    # Using a simpler model to prevent overfitting
    pipeline = SklearnPipelineWrapper([
        ("scaler", "sklearn.preprocessing.StandardScaler", {}),
        ("classifier", "sklearn.ensemble.RandomForestClassifier", {
            "n_estimators": 50,
            "max_depth": 3,
            "random_state": 42
        })
    ])
    
    # Create experiment config
    config = ExperimentConfig(
        name="iris_classification_cv",
        model_params={
            "n_estimators": 50,
            "max_depth": 3
        },
        dataset_params={
            "n_splits": 5
        },
        training_params={
            "random_state": 42
        }
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name="iris_example_cv",
        artifacts_dir="experiments"
    )
    
    # Run experiment with cross-validation
    with tracker:
        # Start run
        tracker.start_run(config)
        
        # Perform cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline.pipeline, X, y, cv=cv)
        
        # Train final model on full dataset
        pipeline.fit(X, y)
        
        # Make predictions
        y_pred = pipeline.predict(X)
        
        # Calculate and log metrics
        tracker.log_metrics({
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "final_accuracy": accuracy_score(y, y_pred)
        })
        
        # Generate and save confusion matrix plot
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=dataset.target_names)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax)
        plt.title("Confusion Matrix")
        plt.savefig("experiments/confusion_matrix.png")
        plt.close()
        
        # Log the plot as an artifact
        tracker.log_artifact("experiments/confusion_matrix.png")
        
        # Save model
        pipeline.save("models/iris_rf.joblib")
        tracker.log_artifact("models/iris_rf.joblib")
        
        # Print results
        print("\nCross-validation Results:")
        print("-" * 50)
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nFinal Model Results:")
        print("-" * 50)
        print(classification_report(y, y_pred, target_names=dataset.target_names))

if __name__ == "__main__":
    run_iris_experiment() 