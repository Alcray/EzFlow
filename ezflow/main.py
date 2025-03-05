#!/usr/bin/env python
"""
CLI entry point for the ezflow framework.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Any

from ezflow.utils.config import Config, load_config_from_file
from ezflow.src.pipeline import Pipeline
from ezflow.models.xgb_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ezflow")

def init_project(project_name):
    """Initialize a new project with the ezflow structure."""
    if os.path.exists(project_name):
        logger.error(f"Directory {project_name} already exists.")
        return False
        
    logger.info(f"Creating new project: {project_name}")
    project_dirs = [
        "",
        "data",
        "data/raw",
        "data/interim",
        "data/processed",
        "models",
        "src",
        "notebooks",
        "utils",
        "deployment"
    ]
    
    for directory in project_dirs:
        dir_path = os.path.join(project_name, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    # Create initial files
    create_initial_files(project_name)
    logger.info(f"Project {project_name} initialized successfully!")
    return True

def create_initial_files(project_name):
    """Create initial template files for the project."""
    # Create README.md
    with open(os.path.join(project_name, "README.md"), "w") as f:
        f.write(f"# {project_name}\n\n")
        f.write("ML project created with ezflow framework.\n\n")
        f.write("## Getting Started\n\n")
        f.write("1. Place your data in the `data/raw` directory\n")
        f.write("2. Run exploratory data analysis using notebooks\n")
        f.write("3. Implement feature engineering in `src/feature_engineering.py`\n")
        f.write("4. Train models using `python -m src.pipeline --train`\n")
    
    # Create basic pipeline.py
    pipeline_content = '''import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pipeline")

def run_pipeline(config):
    """Run the full ML pipeline."""
    logger.info("Starting pipeline...")
    # TODO: Implement your pipeline logic here
    logger.info("Pipeline completed!")

def main():
    parser = argparse.ArgumentParser(description="Run the ML pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    parser.add_argument("--deploy", action="store_true", help="Deploy the model")
    
    args = parser.parse_args()
    
    # Simple config for now, can be replaced with config.py later
    config = {"mode": "development"}
    
    if args.train:
        run_pipeline(config)
    elif args.predict:
        logger.info("Prediction mode not implemented yet.")
    elif args.deploy:
        logger.info("Deployment mode not implemented yet.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
'''
    
    os.makedirs(os.path.join(project_name, "src"), exist_ok=True)
    with open(os.path.join(project_name, "src", "pipeline.py"), "w") as f:
        f.write(pipeline_content)
    
    # Create config.py
    config_content = '''class Config:
    """Configuration class for the project."""
    
    # Data paths
    DATA_PATHS = {
        "train": "data/raw/train.csv",
        "test": "data/raw/test.csv"
    }
    
    # Model parameters
    MODEL_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5
    }
    
    # Training parameters
    TRAIN_PARAMS = {
        "test_size": 0.2,
        "random_state": 42
    }
    
    # Evaluation metrics
    METRICS = ["accuracy", "f1", "precision", "recall"]
    
    # Paths for saving models and results
    SAVE_MODEL_PATH = "models/model.pkl"
    RESULTS_PATH = "models/results.json"
'''
    
    os.makedirs(os.path.join(project_name, "utils"), exist_ok=True)
    with open(os.path.join(project_name, "utils", "config.py"), "w") as f:
        f.write(config_content)

    # Create __init__.py files to make directories importable
    for dir_name in ["src", "utils", "models"]:
        with open(os.path.join(project_name, dir_name, "__init__.py"), "w") as f:
            f.write("# Make the directory a Python package\n")

def train_model(config_path: str) -> None:
    """Train a model using the specified configuration."""
    # Load configuration
    config = load_config_from_file(config_path)
    
    # Create model based on configuration
    model_type = config.MODEL.get('type', 'xgb').lower()
    if model_type == 'xgb':
        model = XGBoostModel(config.MODEL.get('params', {}))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create and run pipeline
    pipeline = Pipeline(config, model)
    results = pipeline.run(save_results=True)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results: {results}")

def predict(input_path: str, model_path: str, output_path: str) -> None:
    """Make predictions using a trained model."""
    # Load the pipeline
    pipeline = Pipeline.load_pipeline(model_path)
    
    # Make predictions
    predictions = pipeline.predict(input_path)
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Determine format based on file extension
    if output_path.endswith('.csv'):
        pd.DataFrame({'prediction': predictions}).to_csv(output_path, index=False)
    elif output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump({'predictions': predictions.tolist()}, f)
    else:
        np.save(output_path, predictions)
    
    logger.info(f"Predictions saved to {output_path}")

def deploy_model(model_path: str, deploy_type: str) -> None:
    """Deploy the model as an API or dashboard."""
    if deploy_type == 'api':
        try:
            from flask import Flask, request, jsonify
            import numpy as np
            
            app = Flask(__name__)
            
            # Load the pipeline
            pipeline = Pipeline.load_pipeline(model_path)
            
            @app.route('/predict', methods=['POST'])
            def predict():
                try:
                    # Get data from request
                    data = request.get_json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data['data'])
                    
                    # Make predictions
                    predictions = pipeline.predict(df)
                    
                    return jsonify({
                        'status': 'success',
                        'predictions': predictions.tolist()
                    })
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': str(e)
                    }), 400
            
            # Run the app
            app.run(host='0.0.0.0', port=8000)
            
        except ImportError:
            logger.error("Flask not installed. Install it with: pip install flask")
            return
            
    elif deploy_type == 'dashboard':
        try:
            import streamlit as st
            import pandas as pd
            
            # Load the pipeline
            pipeline = Pipeline.load_pipeline(model_path)
            
            def run_dashboard():
                st.title('ezflow Model Dashboard')
                
                # File upload
                uploaded_file = st.file_uploader("Upload data for predictions", type=['csv'])
                
                if uploaded_file is not None:
                    # Load and display data
                    data = pd.read_csv(uploaded_file)
                    st.write("Data Preview:", data.head())
                    
                    # Make predictions
                    if st.button('Make Predictions'):
                        predictions = pipeline.predict(data)
                        
                        # Display results
                        results = pd.DataFrame({
                            'Predictions': predictions
                        })
                        st.write("Predictions:", results)
                        
                        # Download predictions
                        st.download_button(
                            "Download Predictions",
                            results.to_csv(index=False),
                            "predictions.csv",
                            "text/csv"
                        )
            
            run_dashboard()
            
        except ImportError:
            logger.error("Streamlit not installed. Install it with: pip install streamlit")
            return
    else:
        logger.error(f"Unsupported deployment type: {deploy_type}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="ez - ML framework for hackathons")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init project command
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument("project_name", help="Name of the project to initialize")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", help="Path to config file", default="utils/config.py")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--input", help="Path to input data", required=True)
    predict_parser.add_argument("--model", help="Path to trained model", required=True)
    predict_parser.add_argument("--output", help="Path to save predictions", required=True)
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model")
    deploy_parser.add_argument("--model", help="Path to trained model", required=True)
    deploy_parser.add_argument("--type", choices=["api", "dashboard"], default="api", 
                              help="Type of deployment (api or dashboard)")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_project(args.project_name)
    elif args.command == "train":
        logger.info(f"Training with config from {args.config}")
        train_model(args.config)
    elif args.command == "predict":
        logger.info(f"Predicting with model {args.model} on {args.input}")
        predict(args.input, args.model, args.output)
    elif args.command == "deploy":
        logger.info(f"Deploying model {args.model} as {args.type}")
        deploy_model(args.model, args.type)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 