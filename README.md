# 🚀 ezflow: Machine Learning Framework for Hackathons

A lightweight, flexible machine learning framework designed for rapid prototyping and hackathons. ezflow uses a manifest-based approach for data handling and provides a modular architecture for easy customization.

## 🌟 Features

- 🚀 **Quick Setup**: Initialize a complete ML project structure with a single command
- 📊 **Data Handling**: Efficient data loading and preprocessing pipelines
- 🔧 **Feature Engineering**: Built-in feature creation and transformation tools
- 🤖 **Model Training**: Support for various ML models with a unified interface
- 📈 **Hyperparameter Tuning**: Integration with popular optimization frameworks (Optuna, Hyperopt)
- 📋 **Evaluation**: Comprehensive model evaluation and visualization tools
- 🌐 **Deployment**: Easy model deployment as REST API or interactive dashboard

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Alcray/ezflow.git
cd ezflow

# Install in development mode
pip install -e .

# Optional: Install extra dependencies
pip install -e .[all]  # Install all optional dependencies
pip install -e .[api]  # Only API deployment dependencies
pip install -e .[dashboard]  # Only dashboard dependencies
```

## 🏃‍♂️ Quick Start

### 1. Initialize Project
```bash
# Create new project
ez init my_project
cd my_project
```

### 2. Prepare Your Data
Create a manifest file (`data/raw/manifest.jsonl`) with your data:
```jsonl
{"feature1": 1.0, "feature2": "value", "target": 0}
{"feature1": 2.0, "feature2": "other", "target": 1}
```

Or create a preprocessing config (`data/raw/preprocess.yaml`) to convert your CSV:
```yaml
documentation: |
  Convert CSV to manifest format and prepare for training.

workspace_dir: data
input_file: data/raw/input.csv
target_column: target

processors:
  # Convert CSV to manifest
  - _target_: CreateManifestFromCSV
    input_file: ${input_file}
    key_mapping:
      target: ${target_column}

  # Add computed features
  - _target_: AddComputedFields
    computations:
      - key: "feature_ratio"
        operation: "ratio"
        input_keys: ["feature1", "feature2"]

  # Split into train/val
  - _target_: SplitManifest
    splits:
      train: 0.8
      val: 0.2
    output_files:
      train: ${workspace_dir}/train_manifest.jsonl
      val: ${workspace_dir}/val_manifest.jsonl
```

### 3. Train Model
```bash
# Basic training
ez train \
  --model xgboost \
  --manifest data/raw/manifest.jsonl \
  --output models/model.pkl

# With custom parameters
ez train \
  --model xgboost \
  --manifest data/raw/manifest.jsonl \
  --params '{"n_estimators": 200, "learning_rate": 0.05}' \
  --output models/model.pkl
```

### 4. Make Predictions
```bash
ez predict \
  --model models/model.pkl \
  --manifest data/test_manifest.jsonl \
  --output predictions.jsonl
```

## Example End-to-End Pipeline

Here's a complete example using the Iris dataset:

```bash
# Create project
ez init iris_project
cd iris_project

# Download Iris dataset
python -c "
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('data/raw/iris.csv', index=False)
"

# Create preprocessing config
cat > data/raw/preprocess.yaml << EOL
workspace_dir: data
input_file: data/raw/iris.csv
target_column: target

processors:
  # Create manifest
  - _target_: CreateManifestFromCSV
    input_file: \${input_file}
    key_mapping:
      target: \${target_column}

  # Add feature interactions
  - _target_: AddComputedFields
    computations:
      - key: "petal_ratio"
        operation: "ratio"
        input_keys: ["petal length (cm)", "petal width (cm)"]
      - key: "sepal_ratio"
        operation: "ratio"
        input_keys: ["sepal length (cm)", "sepal width (cm)"]

  # Split data
  - _target_: SplitManifest
    splits:
      train: 0.8
      val: 0.2
    shuffle: true
    seed: 42
    output_files:
      train: \${workspace_dir}/train_manifest.jsonl
      val: \${workspace_dir}/val_manifest.jsonl
EOL

# Process data
python -c "
from ezflow.data.processor import process_data
process_data('data/raw/preprocess.yaml')
"

# Train model
ez train \
  --model xgboost \
  --manifest data/train_manifest.jsonl \
  --params '{
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1
  }' \
  --output models/iris_model.pkl

# Make predictions
ez predict \
  --model models/iris_model.pkl \
  --manifest data/val_manifest.jsonl \
  --output predictions.jsonl

# View results
python -c "
import json
with open('predictions.jsonl') as f:
    preds = [json.loads(line)['prediction'] for line in f]
print(f'Predictions: {preds[:5]}')
"
```

## 📁 Project Structure

```
my_project/
├── data/
│   ├── raw/          # Original data & preprocessing configs
│   └── processed/    # Processed manifests
├── models/           # Trained models
└── .gitignore       # Git ignore file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 