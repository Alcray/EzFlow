from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from ezflow.core.dataset.molecular import MolecularDataset
from ezflow.core.pipeline.sklearn import SklearnPipelineWrapper
from ezflow.core.experiment.tracker import ExperimentTracker, ExperimentConfig

def run_molecular_experiment():
    """Run a simple molecular machine learning experiment using EZFlow."""
    
    # Initialize dataset
    dataset = MolecularDataset(
        name="example_molecular",
        data_dir="data",
        smiles_col="SMILES",
        label_col="Activity",
        fingerprint_radius=3,
        fingerprint_bits=2048
    )
    
    # Load and preprocess data
    dataset.load_data()
    dataset.split_data(val_size=0.2, random_state=42)
    dataset.preprocess()
    
    # Get features and labels
    X_train = dataset.get_features(dataset.train_data)
    y_train = dataset.get_labels(dataset.train_data)
    X_val = dataset.get_features(dataset.val_data)
    y_val = dataset.get_labels(dataset.val_data)
    
    # Create pipeline
    pipeline = SklearnPipelineWrapper([
        ("classifier", "sklearn.ensemble.RandomForestClassifier", {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        })
    ])
    
    # Create experiment configuration
    config = ExperimentConfig(
        name="molecular_rf",
        model_params={
            "n_estimators": 100,
            "max_depth": 10
        },
        dataset_params={
            "fingerprint_radius": 3,
            "fingerprint_bits": 2048
        },
        training_params={
            "val_size": 0.2,
            "random_state": 42
        }
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name="molecular_classification",
        artifacts_dir="experiments"
    )
    
    # Run experiment
    with tracker:
        # Start run
        tracker.start_run(config)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        train_preds = pipeline.predict_proba(X_train)[:, 1]
        val_preds = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        train_score = average_precision_score(y_train, train_preds)
        val_score = average_precision_score(y_val, val_preds)
        
        # Log metrics
        tracker.log_metrics({
            "train_avg_precision": train_score,
            "val_avg_precision": val_score
        })
        
        # Save model
        pipeline.save("models/molecular_rf.joblib")
        tracker.log_artifact("models/molecular_rf.joblib")
        
        print(f"Training Score: {train_score:.4f}")
        print(f"Validation Score: {val_score:.4f}")

if __name__ == "__main__":
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)
    
    # Run the experiment
    run_molecular_experiment() 