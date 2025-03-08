"""
Data preprocessing module for ezflowx.

This module provides a configurable data processing pipeline using YAML configs.
"""

import os
import json
import logging
from typing import List, Dict, Any, Union, Optional, Callable
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class Processor:
    """Base class for all processors."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the manifest."""
        raise NotImplementedError

class CreateManifestFromCSV(Processor):
    """Create manifest from CSV file."""
    
    def __call__(self, manifest: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        input_file = self.kwargs['input_file']
        key_mapping = self.kwargs.get('key_mapping', {})
        
        logger.info(f"Creating manifest from CSV: {input_file}")
        df = pd.read_csv(input_file)
        
        if key_mapping:
            df = df.rename(columns=key_mapping)
        
        return df.to_dict(orient='records')

class DropDuplicates(Processor):
    """Remove duplicate entries."""
    
    def __call__(self, manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        keys = self.kwargs.get('keys')
        df = pd.DataFrame(manifest)
        n_before = len(df)
        df = df.drop_duplicates(subset=keys)
        n_after = len(df)
        
        if n_before > n_after:
            logger.info(f"Removed {n_before - n_after} duplicate entries")
        
        return df.to_dict(orient='records')

class FilterByValue(Processor):
    """Filter entries by field values."""
    
    def __call__(self, manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filters = self.kwargs['filters']
        filtered = manifest
        
        for key, value in filters.items():
            if isinstance(value, list):
                filtered = [entry for entry in filtered if entry.get(key) in value]
            else:
                filtered = [entry for entry in filtered if entry.get(key) == value]
        
        logger.info(f"Filtered from {len(manifest)} to {len(filtered)} entries")
        return filtered

class AddComputedFields(Processor):
    """Add computed fields based on operations."""
    
    def _compute_value(self, entry: Dict[str, Any], computation: Dict[str, Any]) -> Any:
        operation = computation['operation']
        input_keys = computation['input_keys']
        values = [float(entry[k]) for k in input_keys]
        
        if operation == 'sum':
            return sum(values)
        elif operation == 'mean':
            return sum(values) / len(values)
        elif operation == 'ratio':
            if len(values) != 2:
                raise ValueError("Ratio operation requires exactly 2 input keys")
            return values[0] / values[1] if values[1] != 0 else None
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def __call__(self, manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        computations = self.kwargs['computations']
        
        for entry in manifest:
            for comp in computations:
                try:
                    entry[comp['key']] = self._compute_value(entry, comp)
                except Exception as e:
                    logger.warning(f"Failed to compute {comp['key']}: {str(e)}")
                    entry[comp['key']] = None
        
        return manifest

class SplitManifest(Processor):
    """Split manifest into multiple parts."""
    
    def __call__(self, manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        splits = self.kwargs['splits']
        shuffle = self.kwargs.get('shuffle', True)
        seed = self.kwargs.get('seed', 42)
        output_files = self.kwargs.get('output_files', {})
        
        # Validate splits sum to 1
        if not abs(sum(splits.values()) - 1.0) < 1e-6:
            raise ValueError("Split fractions must sum to 1")
        
        # Convert to DataFrame for easier splitting
        df = pd.DataFrame(manifest)
        
        if shuffle:
            df = df.sample(frac=1, random_state=seed)
        
        # Calculate split indices
        n = len(df)
        indices = [0]
        current_idx = 0
        
        for fraction in splits.values():
            current_idx += int(n * fraction)
            indices.append(current_idx)
        
        # Create and save splits
        result = {}
        for (name, _), start_idx, end_idx in zip(splits.items(), indices[:-1], indices[1:]):
            split_manifest = df.iloc[start_idx:end_idx].to_dict(orient='records')
            result[name] = split_manifest
            
            # Save if output path provided
            if name in output_files:
                save_manifest(split_manifest, output_files[name])
            
            logger.info(f"Split '{name}': {len(split_manifest)} entries")
        
        # Return the largest split by default
        largest_split = max(result.items(), key=lambda x: len(x[1]))[1]
        return largest_split

def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Load manifest from a JSONL file.
    
    Args:
        manifest_path (str): Path to manifest.jsonl file
        
    Returns:
        List[Dict[str, Any]]: List of manifest entries
    """
    data = []
    with open(manifest_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_manifest(manifest: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save manifest to a JSONL file.
    
    Args:
        manifest (List[Dict[str, Any]]): List of manifest entries
        output_path (str): Path to save manifest.jsonl
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in manifest:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Manifest saved to {output_path}")

class DataProcessor:
    """Main data processing class."""
    
    def __init__(self, config_path: str):
        """Initialize with YAML config file."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Resolve workspace directory
        if self.config['workspace_dir'] == '???':
            self.config['workspace_dir'] = os.getcwd()
        
        # Create workspace directory
        os.makedirs(self.config['workspace_dir'], exist_ok=True)
        
        # Initialize processors
        self.processors = {}
        for proc in [
            CreateManifestFromCSV,
            DropDuplicates,
            FilterByValue,
            AddComputedFields,
            SplitManifest
        ]:
            self.processors[proc.__name__] = proc
    
    def process(self) -> List[Dict[str, Any]]:
        """Run the processing pipeline."""
        manifest = None
        
        # Process each step
        for step in self.config['processors']:
            processor_name = step.pop('_target_')
            
            if processor_name not in self.processors:
                raise ValueError(f"Unknown processor: {processor_name}")
            
            # Resolve variables in arguments
            args = {}
            for k, v in step.items():
                if isinstance(v, str) and v.startswith('${'):
                    key = v[2:-1]  # Remove ${ and }
                    if key not in self.config:
                        raise ValueError(f"Config variable not found: {key}")
                    args[k] = self.config[key]
                else:
                    args[k] = v
            
            # Create and run processor
            processor = self.processors[processor_name](**args)
            manifest = processor(manifest)
            logger.info(f"Applied {processor_name}")
        
        return manifest

def process_data(config_path: str) -> List[Dict[str, Any]]:
    """Process data using config file."""
    processor = DataProcessor(config_path)
    return processor.process() 