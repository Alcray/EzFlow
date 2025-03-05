"""
Pipeline module for the ezflow framework.

This module contains the Pipeline class for orchestrating the entire ML workflow.
"""

import os
import logging
import time
import json
from typing import Dict, Optional, Union, List, Any, Tuple
import pandas as pd
import numpy as np
import importlib
import joblib

from ezflow.src.data_loader import DataLoader
from ezflow.src.feature_engineering import FeatureEngineer
from ezflow.src.trainer import ModelTrainer
from ezflow.src.evaluator import Evaluator
from ezflow.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Pipeline class for orchestrating the entire ML workflow.
    
    This class combines data loading, feature engineering, model training, 
    and evaluation into a complete ML pipeline.
    """
    
    def __init__(self, config: Dict, model: Optional[BaseModel] = None):
        """
        Initialize the Pipeline with configuration and optionally a model.
        
        Args:
            config (Dict): Configuration dictionary.
            model (Optional[BaseModel]): Model instance to use. If None, will be instantiated based on config.
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.evaluator = Evaluator(config)
        
        # Set up model if provided
        self.model = model
        self.trainer = None
        
        if self.model is not None:
            self.trainer = ModelTrainer(self.model, config)
        
        logger.info("Pipeline initialized")
    
    def set_model(self, model: BaseModel) -> None:
        """
        Set the model to use in the pipeline.
        
        Args:
            model (BaseModel): Model instance.
        """
        self.model = model
        self.trainer = ModelTrainer(self.model, self.config)
        logger.info(f"Pipeline model set to {type(model).__name__}")
    
    def _instantiate_model_from_config(self) -> BaseModel:
        """
        Instantiate a model from configuration.
        
        Returns:
            BaseModel: Instantiated model.
            
        Raises:
            ValueError: If model configuration is invalid.
        """
        model_config = self.config.get('MODEL', {})
        model_type = model_config.get('type')
        model_params = model_config.get('params', {})
        
        if not model_type:
            raise ValueError("Model type not specified in config")
        
        try:
            # Try to import from ezflow.models first
            module_path = f"ezflow.models.{model_type.lower()}_model"
            module = importlib.import_module(module_path)
            
            # Get class name based on model type
            class_name = ''.join(word.capitalize() for word in model_type.split('_')) + 'Model'
            model_class = getattr(module, class_name)
            
            # Instantiate model
            model = model_class(params=model_params)
            logger.info(f"Instantiated {class_name} from config")
            
            return model
        
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to instantiate model from config: {str(e)}")
            raise ValueError(f"Invalid model configuration: {str(e)}")
    
    def run(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the full pipeline from data loading to evaluation.
        
        Args:
            save_results (bool): Whether to save pipeline results.
            
        Returns:
            Dict[str, Any]: Pipeline results.
        """
        logger.info("Starting pipeline execution...")
        start_time = time.time()
        
        # Dictionary to track results
        results = {
            'pipeline_config': self.config,
            'metrics': {},
            'feature_importance': None,
            'execution_time': None,
        }
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading data...")
            train_data = self.data_loader.load_data('train')
            
            # Optional: Load test data if available
            test_data = None
            try:
                test_data = self.data_loader.load_data('test')
            except (ValueError, FileNotFoundError):
                logger.info("Test data not available, will use train-test split instead")
            
            # Step 2: Clean data
            logger.info("Step 2: Cleaning data...")
            train_data = self.data_loader.clean_data(train_data)
            
            if test_data is not None:
                test_data = self.data_loader.clean_data(test_data)
            
            # Step 3: Feature engineering
            logger.info("Step 3: Performing feature engineering...")
            train_data = self.feature_engineer.create_features(train_data)
            
            if test_data is not None:
                test_data = self.feature_engineer.create_features(test_data)
            
            # Step 4: Split data
            logger.info("Step 4: Splitting data...")
            target_col = self.config.get('TARGET_COLUMN')
            
            if not target_col:
                raise ValueError("Target column not specified in config")
            
            if test_data is not None:
                # If test data is available, use it
                X_train, y_train = self.data_loader.split_features_target(train_data, target_col)
                X_test, y_test = self.data_loader.split_features_target(test_data, target_col)
            else:
                # Otherwise, perform train-test split
                test_size = self.config.get('TRAIN_PARAMS', {}).get('test_size', 0.2)
                random_state = self.config.get('TRAIN_PARAMS', {}).get('random_state', 42)
                
                X_train, X_test, y_train, y_test = self.data_loader.train_test_split(
                    train_data, target_col, test_size=test_size, random_state=random_state
                )
            
            # Step 5: Preprocess data
            logger.info("Step 5: Preprocessing data...")
            # Detect feature types
            numeric_features, categorical_features = self.feature_engineer.detect_feature_types(X_train)
            
            # Apply preprocessing
            X_train = self.feature_engineer.fit_transform(X_train, numeric_features, categorical_features)
            X_test = self.feature_engineer.transform(X_test)
            
            # Step 6: Initialize model if not provided
            if self.model is None:
                logger.info("Step 6: Initializing model from config...")
                self.model = self._instantiate_model_from_config()
                self.trainer = ModelTrainer(self.model, self.config)
            
            # Step 7: Train model
            logger.info("Step 7: Training model...")
            self.trainer.train(X_train, y_train)
            
            # Step 8: Evaluate model
            logger.info("Step 8: Evaluating model...")
            y_pred = self.trainer.predict(X_test)
            
            # Get probability predictions if available (for classification tasks)
            y_proba = None
            try:
                y_proba = self.trainer.predict_proba(X_test)
            except (ValueError, NotImplementedError):
                logger.info("Probability predictions not available")
            
            metrics = self.evaluator.calculate_metrics(y_test, y_pred, y_proba)
            results['metrics'] = metrics
            
            # Step 9: Get feature importance if available
            try:
                feature_importance = self.trainer.get_feature_importance()
                if feature_importance is not None:
                    results['feature_importance'] = feature_importance.to_dict(orient='records')
                    
                    if save_results:
                        feature_importance_path = self.config.get('FEATURE_IMPORTANCE_PATH', 'models/feature_importance.png')
                        self.evaluator.feature_importance_plot(
                            feature_importance, 
                            top_n=min(20, len(feature_importance)),
                            save_path=feature_importance_path
                        )
            except (ValueError, NotImplementedError):
                logger.info("Feature importance not available")
            
            # Step 10: Save model and results
            if save_results:
                logger.info("Step 10: Saving model and results...")
                
                # Save model
                model_path = self.config.get('SAVE_MODEL_PATH', 'models/model.pkl')
                self.trainer.save_model(model_path)
                
                # Save metrics
                results_path = self.config.get('RESULTS_PATH', 'models/results.json')
                self.evaluator.save_metrics(results_path)
                
                # Generate and save plots
                if y_proba is not None and len(np.unique(y_test)) == 2:
                    roc_path = self.config.get('ROC_CURVE_PATH', 'models/roc_curve.png')
                    self.evaluator.roc_curve(y_test, y_proba, save_path=roc_path)
                
                # For classification tasks
                if len(np.unique(y_test)) <= 10:
                    cm_path = self.config.get('CONFUSION_MATRIX_PATH', 'models/confusion_matrix.png')
                    self.evaluator.confusion_matrix(y_test, y_pred, normalize=True, save_path=cm_path)
                    report = self.evaluator.print_classification_report(y_test, y_pred)
                    results['classification_report'] = report
                
                # For regression tasks
                else:
                    regression_plot_path = self.config.get('REGRESSION_PLOT_PATH', 'models/regression_plot.png')
                    self.evaluator.plot_regression_results(y_test, y_pred, save_path=regression_plot_path)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            
            logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
            logger.info(f"Evaluation metrics: {metrics}")
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def cross_validate(self, n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the pipeline.
        
        Args:
            n_splits (int): Number of folds for cross-validation.
            
        Returns:
            Dict[str, Any]: Cross-validation results.
        """
        logger.info(f"Starting {n_splits}-fold cross-validation...")
        
        # Load and preprocess data
        train_data = self.data_loader.load_data('train')
        train_data = self.data_loader.clean_data(train_data)
        train_data = self.feature_engineer.create_features(train_data)
        
        # Get target and features
        target_col = self.config.get('TARGET_COLUMN')
        
        if not target_col:
            raise ValueError("Target column not specified in config")
        
        X, y = self.data_loader.split_features_target(train_data, target_col)
        
        # Initialize model if not provided
        if self.model is None:
            self.model = self._instantiate_model_from_config()
            self.trainer = ModelTrainer(self.model, self.config)
        
        # Perform cross-validation
        cv_results = self.trainer.cross_validate(
            X, y, n_splits=n_splits, 
            metrics=self.config.get('METRICS')
        )
        
        # Calculate and log mean scores
        mean_scores = {f"{metric}_mean": np.mean(scores) for metric, scores in cv_results.items()}
        std_scores = {f"{metric}_std": np.std(scores) for metric, scores in cv_results.items()}
        
        logger.info(f"Cross-validation results (mean): {mean_scores}")
        logger.info(f"Cross-validation results (std): {std_scores}")
        
        return {
            'cv_results': cv_results,
            'mean_scores': mean_scores,
            'std_scores': std_scores
        }
    
    def predict(self, data: Union[pd.DataFrame, str], model_path: Optional[str] = None) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data (Union[pd.DataFrame, str]): Input data or path to data file.
            model_path (Optional[str]): Path to the model file. If None, uses the pipeline's model.
            
        Returns:
            np.ndarray: Predictions.
        """
        logger.info("Making predictions on new data...")
        
        # Load data if path is provided
        if isinstance(data, str):
            # Assume it's a path to a data file
            df = pd.read_csv(data) if data.endswith('.csv') else pd.read_parquet(data)
        else:
            df = data.copy()
        
        # Clean and preprocess data
        df = self.data_loader.clean_data(df)
        df = self.feature_engineer.create_features(df)
        
        # Remove target column if present
        target_col = self.config.get('TARGET_COLUMN')
        if target_col in df.columns:
            df = df.drop(columns=[target_col])
        
        # Apply feature transformation
        df = self.feature_engineer.transform(df)
        
        # Load model if path is provided
        if model_path is not None:
            logger.info(f"Loading model from {model_path}")
            if self.trainer is None:
                self.model = self._instantiate_model_from_config()
                self.trainer = ModelTrainer(self.model, self.config)
            
            self.trainer.load_model(model_path)
        elif self.trainer is None:
            raise ValueError("Model not initialized and model_path not provided")
        
        # Make predictions
        predictions = self.trainer.predict(df)
        
        logger.info(f"Made predictions with shape {predictions.shape}")
        
        return predictions
    
    def save_pipeline(self, path: str) -> None:
        """
        Save the entire pipeline to disk.
        
        Args:
            path (str): Path to save the pipeline.
        """
        logger.info(f"Saving pipeline to {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save pipeline state
        joblib.dump(self, path)
        
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load_pipeline(cls, path: str) -> 'Pipeline':
        """
        Load a pipeline from disk.
        
        Args:
            path (str): Path to load the pipeline from.
            
        Returns:
            Pipeline: Loaded pipeline.
        """
        logger.info(f"Loading pipeline from {path}")
        
        # Load pipeline
        pipeline = joblib.load(path)
        
        if not isinstance(pipeline, cls):
            raise ValueError(f"Loaded object is not a {cls.__name__}")
        
        logger.info(f"Pipeline loaded from {path}")
        
        return pipeline


def run_pipeline_from_config(config_path: str, save_results: bool = True) -> Dict[str, Any]:
    """
    Run a pipeline from a configuration file.
    
    Args:
        config_path (str): Path to the configuration file (JSON or YAML).
        save_results (bool): Whether to save pipeline results.
        
    Returns:
        Dict[str, Any]: Pipeline results.
    """
    import json
    import yaml
    
    logger.info(f"Loading configuration from {config_path}")
    
    # Load configuration
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    # Initialize and run pipeline
    pipeline = Pipeline(config)
    results = pipeline.run(save_results=save_results)
    
    return results


def main():
    """
    CLI entry point for the pipeline module.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the ML pipeline")
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file", 
        required=True
    )
    
    parser.add_argument(
        "--mode", 
        choices=["train", "predict", "cross-validate"],
        default="train",
        help="Pipeline execution mode"
    )
    
    parser.add_argument(
        "--input", 
        help="Path to input data for prediction mode",
        required=False
    )
    
    parser.add_argument(
        "--output", 
        help="Path to save predictions",
        required=False
    )
    
    parser.add_argument(
        "--model", 
        help="Path to model for prediction mode",
        required=False
    )
    
    parser.add_argument(
        "--folds", 
        type=int,
        default=5,
        help="Number of folds for cross-validation"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "train":
            # Train mode
            results = run_pipeline_from_config(args.config)
            
            # Output a summary of results
            print("\nTraining Results:")
            print("-" * 50)
            for metric, value in results['metrics'].items():
                print(f"{metric}: {value:.4f}")
            print(f"Execution time: {results['execution_time']:.2f} seconds")
            
        elif args.mode == "predict":
            # Predict mode
            if not args.input:
                raise ValueError("Input data path required for prediction mode")
            
            if not args.model:
                raise ValueError("Model path required for prediction mode")
            
            # Load config
            import json
            import yaml
            
            if args.config.endswith('.json'):
                with open(args.config, 'r') as f:
                    config = json.load(f)
            elif args.config.endswith(('.yaml', '.yml')):
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {args.config}")
            
            # Initialize pipeline
            pipeline = Pipeline(config)
            
            # Make predictions
            predictions = pipeline.predict(args.input, model_path=args.model)
            
            # Save predictions if output path is provided
            if args.output:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                
                # Save predictions based on file extension
                if args.output.endswith('.csv'):
                    pd.DataFrame(predictions, columns=['prediction']).to_csv(args.output, index=False)
                elif args.output.endswith('.npy'):
                    np.save(args.output, predictions)
                else:
                    pd.DataFrame(predictions, columns=['prediction']).to_csv(args.output, index=False)
                
                print(f"Predictions saved to {args.output}")
            else:
                # Print preview of predictions
                print("\nPredictions Preview:")
                print("-" * 50)
                print(predictions[:10])
                print(f"Total predictions: {len(predictions)}")
            
        elif args.mode == "cross-validate":
            # Cross-validation mode
            import json
            import yaml
            
            if args.config.endswith('.json'):
                with open(args.config, 'r') as f:
                    config = json.load(f)
            elif args.config.endswith(('.yaml', '.yml')):
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {args.config}")
            
            # Initialize pipeline
            pipeline = Pipeline(config)
            
            # Perform cross-validation
            cv_results = pipeline.cross_validate(n_splits=args.folds)
            
            # Output a summary of results
            print("\nCross-Validation Results:")
            print("-" * 50)
            for metric, value in cv_results['mean_scores'].items():
                std = cv_results['std_scores'][metric.replace('_mean', '_std')]
                print(f"{metric}: {value:.4f} ± {std:.4f}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 