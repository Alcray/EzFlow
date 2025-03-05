#!/usr/bin/env python
"""
Script to prepare iris dataset for ezflow
"""

import os
import json
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Create a complete dataframe
df = X.copy()
df['target'] = y

# Print dataset info
print(f"Dataset shape: {df.shape}")
print(f"Features: {X.columns.tolist()}")
print(f"Target classes: {iris.target_names.tolist()}")

# Create data directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create training manifest
train_df = X_train.copy()
train_df['target'] = y_train
train_manifest = train_df.to_dict(orient='records')

# Create test manifest
test_df = X_test.copy()
test_df['target'] = y_test
test_manifest = test_df.to_dict(orient='records')

# Save manifests
with open('data/processed/train_manifest.jsonl', 'w') as f:
    for item in train_manifest:
        f.write(json.dumps(item) + '\n')
        
with open('data/processed/test_manifest.jsonl', 'w') as f:
    for item in test_manifest:
        f.write(json.dumps(item) + '\n')

print(f"Created train manifest with {len(train_manifest)} samples")
print(f"Created test manifest with {len(test_manifest)} samples")
print("Files saved to data/processed/train_manifest.jsonl and data/processed/test_manifest.jsonl") 