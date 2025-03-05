# Command-Line Interface

EZFlow provides a command-line interface (CLI) that makes it easy to create, manage, and run machine learning projects.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Available Commands](#available-commands)
  - [Create Project](#create-project)
  - [Run Experiment](#run-experiment)
  - [List Experiments](#list-experiments)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Extending the CLI](#extending-the-cli)

## Overview

The EZFlow CLI provides a set of commands to help you:

- Create new projects with appropriate directory structure
- Run experiments from configuration files
- List and compare experiment results
- Manage models and datasets

The CLI follows consistent patterns and provides helpful error messages to make the workflow smooth and intuitive.

## Installation

The CLI is automatically installed when you install the EZFlow package:

```bash
# Install from PyPI
pip install ezflow-ml

# Or install in development mode from source
pip install -e .
```

After installation, the `ezflow` command will be available in your terminal.

## Available Commands

### Create Project

The `create` command sets up a new EZFlow project with the recommended directory structure.

#### Usage

```bash
ezflow create <project_name> [--template <template_name>]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `project_name` | Name of the project to create |

#### Options

| Option | Description |
|--------|-------------|
| `--template` | Template to use for project creation (default: "basic") |

#### Directory Structure

The created project will have the following structure:

```
project_name/
├── data/
│   ├── raw/            # Raw, immutable data
│   └── processed/      # Processed data ready for training
├── models/             # Saved models
├── experiments/        # Experiment results and artifacts
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── __init__.py
│   ├── datasets/       # Custom dataset implementations
│   ├── pipelines/      # Custom pipeline implementations
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── .gitignore          # Git ignore file with ML-specific patterns
└── README.md           # Project README
```

#### Example

```bash
# Create a new project named "drug_discovery"
ezflow create drug_discovery

# Create a project with a specific template
ezflow create image_classification --template vision
```

### Run Experiment

The `run` command executes an experiment based on a configuration file.

#### Usage

```bash
ezflow run <config_file> [--output <output_dir>] [--name <experiment_name>]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `config_file` | Path to experiment configuration file (JSON or YAML) |

#### Options

| Option | Description |
|--------|-------------|
| `--output` | Directory to store experiment results (default: "./experiments") |
| `--name` | Name for the experiment run (default: auto-generated) |
| `--no-tracking` | Disable experiment tracking |

#### Example

```bash
# Run an experiment with a specific configuration
ezflow run config/random_forest.json

# Run with custom output directory and name
ezflow run config/neural_network.yaml --output ./results --name nn_run_1
```

### List Experiments

The `list` command displays information about previous experiment runs.

#### Usage

```bash
ezflow list [--n <number>] [--filter <filter_string>]
```

#### Options

| Option | Description |
|--------|-------------|
| `--n` | Number of experiments to show (default: 10) |
| `--filter` | Filter experiments by name, status, or parameters |
| `--sort` | Sort by metric or parameter (e.g., "accuracy:desc") |
| `--output` | Output format: "table", "json", or "csv" (default: "table") |

#### Example

```bash
# List 5 most recent experiments
ezflow list --n 5

# Filter experiments by name
ezflow list --filter "name=random_forest"

# Sort by validation accuracy (descending)
ezflow list --sort "metrics.val_accuracy:desc"

# Export results as CSV
ezflow list --output csv > experiments.csv
```

## Examples

### Complete Workflow Example

```bash
# Create a new project
ezflow create my_ml_project
cd my_ml_project

# Create data directories
mkdir -p data/raw data/processed

# Prepare your dataset
# ... (download/copy data files to data/raw)

# Create experiment configuration
cat > config/experiment.json << EOL
{
  "name": "first_experiment",
  "model_params": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "dataset_params": {
    "val_size": 0.2
  },
  "training_params": {
    "random_state": 42
  }
}
EOL

# Run the experiment
ezflow run config/experiment.json --name initial_run

# List experiment results
ezflow list --n 1
```

### Creating and Running Multiple Experiments

```bash
# Create multiple configuration files
for n_estimators in 50 100 200; do
  for max_depth in 5 10 15; do
    cat > config/rf_${n_estimators}_${max_depth}.json << EOL
{
  "name": "rf_${n_estimators}_${max_depth}",
  "model_params": {
    "n_estimators": ${n_estimators},
    "max_depth": ${max_depth}
  },
  "dataset_params": {
    "val_size": 0.2
  },
  "training_params": {
    "random_state": 42
  }
}
EOL
  done
done

# Run all experiments
for config in config/rf_*.json; do
  ezflow run $config
done

# Compare results
ezflow list --sort "metrics.val_accuracy:desc"
```

## Best Practices

For optimal results when using the CLI:

1. **Version Control**: Store configuration files in version control to track experiment setups
2. **Consistent Naming**: Use consistent naming conventions for experiment configurations
3. **Configuration Files**: Prefer configuration files over command-line arguments for reproducibility
4. **Documentation**: Document the purpose and expected outcomes of each configuration
5. **Automation**: Use shell scripts or Makefiles to automate common workflows
6. **Environment Variables**: Use environment variables for sensitive information (API keys, paths)

## Extending the CLI

You can extend the EZFlow CLI with custom commands by creating plugins.

### Creating a Custom Command

```python
# In your_extension.py
from ezflow.cli import register_command

@register_command("analyze", "Analyze experiment results")
def analyze_command(args):
    """Analyze experiment results and generate reports."""
    # Implementation of the command
    pass

# Register command arguments
analyze_command.add_argument("experiment", help="Experiment ID to analyze")
analyze_command.add_argument("--output", help="Output directory for reports")
```

### Installing the Extension

```python
# In setup.py
setup(
    name="ezflow-extension",
    # ...
    entry_points={
        "ezflow.cli_plugins": [
            "analyze=your_package.your_extension:analyze_command",
        ]
    }
)
```

### Using the Extension

After installing your extension, the command will be available:

```bash
ezflow analyze experiment_id --output ./reports
``` 