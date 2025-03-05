#!/usr/bin/env python
"""
CLI entry point for the ezflow framework.
"""

import argparse
import os
import logging
import json
from typing import Dict, Any
import shutil

from ezflow.models import get_model
from ezflow.data.processor import load_manifest, save_manifest, process_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ezflow")

def init_project(project_name: str):
    """
    Create basic project structure.
    
    Args:
        project_name (str): Name of the project directory to create
    """
    if os.path.exists(project_name):
        logger.error(f"Directory {project_name} already exists.")
        return False
    
    logger.info(f"Creating project structure in: {project_name}")
    
    # Create basic directory structure
    project_dirs = [
        "",
        "data",
        "data/raw",
        "data/processed",
        "models",
    ]
    
    for directory in project_dirs:
        dir_path = os.path.join(project_name, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    # Create example manifest.jsonl
    example_manifest = """{"feature1": 1.0, "feature2": "value", "target": 0}
{"feature1": 2.0, "feature2": "other", "target": 1}"""
    
    with open(os.path.join(project_name, "data/raw/example_manifest.jsonl"), "w") as f:
        f.write(example_manifest)
    
    # Create example preprocessing steps
    example_steps = [
        {"name": "drop_duplicates"},
        {"name": "filter_by_value", "filters": {"feature1": 1.0}},
        {"name": "split_manifest", "splits": {"train": 0.8, "val": 0.2}}
    ]
    
    with open(os.path.join(project_name, "data/raw/preprocess_steps.json"), "w") as f:
        json.dump(example_steps, f, indent=2)
    
    # Create .gitignore
    gitignore_content = """# Data
data/raw/*
data/processed/*

# Models
models/*.pkl
models/*.json
models/*.png

# Logs
*.log

# Python
__pycache__/
*.py[cod]
*$py.class

# Keep examples
!data/raw/example_manifest.jsonl
!data/raw/preprocess_steps.json
"""
    
    with open(os.path.join(project_name, ".gitignore"), "w") as f:
        f.write(gitignore_content)
    
    logger.info(f"""
Project initialized successfully!
    
To get started:
1. Create your manifest.jsonl file in data/raw/
   Example format is in data/raw/example_manifest.jsonl
2. Modify preprocessing steps in data/raw/preprocess_steps.json
3. Train your model:
   ez train --model xgboost --manifest data/raw/manifest.jsonl
""")
    return True

def train_model(model_type: str, manifest_path: str, preprocess_steps: str = None,
                model_params: Dict = None, target_key: str = "target", 
                output_path: str = None):
    """Train a model using manifest data."""
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found: {manifest_path}")
        return False
    
    # Process manifest if needed
    processed_manifest_path = manifest_path
    try:
        if preprocess_steps:
            # Check if it's a YAML config file for processing
            if preprocess_steps.endswith('.yaml') or preprocess_steps.endswith('.yml'):
                # This will process according to the config
                process_data(preprocess_steps)
                
                # We would need to know where the processed manifest is saved
                # For now, we'll continue using the original manifest
                logger.warning("Using original manifest, processed manifest path unknown.")
            else:
                # Assume it's a JSON file with processing steps
                logger.warning("JSON-based preprocessing is not fully implemented.")
    except Exception as e:
        logger.error(f"Failed to process manifest: {str(e)}")
        return False
    
    # Create and train model
    try:
        model = get_model(model_type, params=model_params or {}, target_key=target_key)
        model.train(processed_manifest_path)
        logger.info("Training completed successfully!")
        
        # Save model if path provided
        if output_path:
            model.save(output_path)
            logger.info(f"Model saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

def predict(model_path: str, manifest_path: str, output_path: str):
    """Make predictions using a trained model."""
    # Load model
    try:
        # Default to xgboost model type
        model_type = "xgboost"
        model = get_model(model_type)
        model.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False
    
    # Make predictions
    try:
        # Load manifest
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest file not found: {manifest_path}")
            return False
            
        predictions = model.predict(manifest_path)
        
        # Create directory for output if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's a directory part
            os.makedirs(output_dir, exist_ok=True)
        
        # Save as manifest
        result_manifest = [{"prediction": float(p)} for p in predictions]
        
        # Write directly to file
        with open(output_path, 'w') as f:
            for item in result_manifest:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Predictions saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return False

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
    train_parser.add_argument("--model", required=True, help="Model type (e.g., xgboost)")
    train_parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    train_parser.add_argument("--preprocess", help="Path to preprocessing steps JSON")
    train_parser.add_argument("--target-key", default="target", help="Key for target variable")
    train_parser.add_argument("--output", help="Path to save model")
    train_parser.add_argument("--params", type=json.loads, help="Model parameters as JSON string")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    predict_parser.add_argument("--output", required=True, help="Path to save predictions")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_project(args.project_name)
    elif args.command == "train":
        output_path = args.output or f"models/{args.model}_model.pkl"
        train_model(args.model, args.manifest, args.preprocess, 
                   args.params, args.target_key, output_path)
    elif args.command == "predict":
        predict(args.model, args.manifest, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 