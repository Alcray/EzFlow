# EZFlow Architecture

This document provides a comprehensive overview of the EZFlow architecture, design principles, and how components interact with each other.

## Overview

EZFlow is designed with modularity and flexibility in mind, allowing researchers and data scientists to focus on their specific tasks rather than boilerplate code. The architecture follows a layered approach with clear separation of concerns.

```
┌─────────────────────────────────────────────────────────────┐
│                     EZFlow Framework                         │
├─────────────┬─────────────┬───────────────┬─────────────────┤
│   Dataset   │  Pipeline   │  Experiment   │      CLI        │
│    Layer    │    Layer    │    Layer      │     Layer       │
├─────────────┼─────────────┼───────────────┼─────────────────┤
│ BaseDataset │  Pipeline   │ ExperimentCfg │   CLI Commands  │
│ IrisDataset │  SklearnPipe│ ExpTracker    │   Project Setup │
│ MolecDataset│  TorchPipe  │ Artifacts     │   Utilities     │
└─────────────┴─────────────┴───────────────┴─────────────────┘
```

## Design Principles

EZFlow's architecture is guided by the following design principles:

1. **Modularity**: Each component should be replaceable without affecting the rest of the system.
2. **Extensibility**: New components can be added with minimal changes to existing code.
3. **Consistency**: Common interfaces across different implementations for ease of use.
4. **Transparency**: Clear visibility into what happens at each step of the pipeline.
5. **Reproducibility**: Experiments should be fully reproducible from the stored configuration.

## Core Components

### Dataset Layer

The dataset layer handles data loading, preprocessing, and splitting:

```
┌───────────────────────────────────────────────────────┐
│                    BaseDataset                         │
├───────────────────────────────────────────────────────┤
│ - name: str                                           │
│ - data_dir: str/Path                                  │
│ - cache_dir: str/Path                                 │
│ - metadata: Dict                                      │
├───────────────────────────────────────────────────────┤
│ + load_data() -> None                                 │
│ + split_data(val_size, random_state) -> None          │
│ + preprocess() -> None                                │
│ + get_features(data) -> np.ndarray                    │
│ + get_labels(data) -> np.ndarray                      │
│ + save_splits() -> None                               │
│ + load_splits() -> None                               │
└───────────────────────────────────────────────────────┘
                           ▲
                           │
                           │ inherits
                           │
           ┌───────────────┴──────────────┐
           │                              │
┌──────────┴─────────────┐   ┌────────────┴─────────────┐
│      IrisDataset       │   │    MolecularDataset      │
├────────────────────────┤   ├──────────────────────────┤
│ + load_data()          │   │ - smiles_col: str        │
│ + split_data()         │   │ - label_col: str         │
│ + preprocess()         │   │ - fingerprint_radius: int│
│ + get_features()       │   │ - fingerprint_bits: int  │
│ + get_labels()         │   ├──────────────────────────┤
└────────────────────────┘   │ + generate_fingerprint() │
                             │ + load_data()            │
                             │ + preprocess()           │
                             │ + get_features()         │
                             │ + get_labels()           │
                             └──────────────────────────┘
```

The dataset layer follows the Template Method pattern, where `BaseDataset` defines the interface and workflow, while specific dataset implementations provide the concrete implementations.

### Pipeline Layer

The pipeline layer handles model training, evaluation, and prediction:

```
┌───────────────────────────────────────────────────────┐
│                     Pipeline                           │
├───────────────────────────────────────────────────────┤
│ - pipeline: Any                                       │
│ - pipeline_config: Dict                               │
├───────────────────────────────────────────────────────┤
│ + fit(X, y) -> None                                   │
│ + predict(X) -> np.ndarray                            │
│ + score(X, y) -> float                                │
│ + save(path) -> None                                  │
│ + load(path) -> None                                  │
└───────────────────────────────────────────────────────┘
                           ▲
                           │
                           │ inherits
                           │
           ┌───────────────┴──────────────┐
           │                              │
┌──────────┴─────────────┐   ┌────────────┴─────────────┐
│ SklearnPipelineWrapper │   │   TorchPipelineWrapper   │
├────────────────────────┤   ├──────────────────────────┤
│ - steps: List          │   │ - model: nn.Module       │
│ - sklearn_pipeline     │   │ - optimizer: Optimizer   │
├────────────────────────┤   │ - criterion: Loss        │
│ + fit()                │   ├──────────────────────────┤
│ + predict()            │   │ + fit()                  │
│ + score()              │   │ + predict()              │
│ + save()               │   │ + score()                │
│ + load()               │   │ + save()                 │
└────────────────────────┘   │ + load()                 │
                             └──────────────────────────┘
```

The pipeline layer uses the Strategy pattern, allowing different ML frameworks to be used with the same interface.

### Experiment Layer

The experiment layer handles experiment tracking, configuration, and artifact management:

```
┌───────────────────────────────────────────────────────┐
│                 ExperimentConfig                       │
├───────────────────────────────────────────────────────┤
│ - name: str                                           │
│ - model_params: Dict                                  │
│ - dataset_params: Dict                                │
│ - training_params: Dict                               │
│ - metadata: Dict                                      │
├───────────────────────────────────────────────────────┤
│ + to_dict() -> Dict                                   │
│ + from_dict(config_dict) -> ExperimentConfig          │
│ + save(path) -> None                                  │
│ + load(path) -> ExperimentConfig                      │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│                 ExperimentTracker                      │
├───────────────────────────────────────────────────────┤
│ - experiment_name: str                                │
│ - artifacts_dir: str/Path                             │
│ - backend: str                                        │
├───────────────────────────────────────────────────────┤
│ + start_run(config) -> None                           │
│ + end_run() -> None                                   │
│ + log_metric(name, value) -> None                     │
│ + log_metrics(metrics_dict) -> None                   │
│ + log_parameter(name, value) -> None                  │
│ + log_parameters(params_dict) -> None                 │
│ + log_artifact(artifact_path) -> None                 │
│ + get_artifact(artifact_path) -> str/Path             │
└───────────────────────────────────────────────────────┘
```

The experiment layer follows the Facade pattern, providing a simplified interface to the underlying tracking systems (like MLflow).

### CLI Layer

The CLI layer provides command-line utilities for project management:

```
┌───────────────────────────────────────────────────────┐
│                     CLI                                │
├───────────────────────────────────────────────────────┤
│ + create_project(name) -> None                        │
│ + run_experiment(config_path) -> None                 │
│ + list_experiments() -> None                          │
└───────────────────────────────────────────────────────┘
```

The CLI layer follows the Command pattern, with each CLI command implementing a specific action.

## Data Flow

A typical data flow in EZFlow follows these steps:

1. **Data Loading**: The dataset loads data from files or external sources
2. **Data Splitting**: The dataset splits data into training and validation sets
3. **Feature Extraction**: Features and labels are extracted from the dataset
4. **Model Training**: The pipeline trains the model using the extracted features and labels
5. **Model Evaluation**: The pipeline evaluates the model on validation data
6. **Metrics Tracking**: The experiment tracker logs metrics, parameters, and artifacts
7. **Model Saving**: The trained model is saved as an artifact

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│  Dataset  │────>│  Pipeline │────>│ Experiment│────>│ Artifacts │
│  Loading  │     │  Training │     │ Tracking  │     │ Saving    │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
```

## Component Interaction

Components in EZFlow interact through well-defined interfaces:

1. **Dataset + Pipeline**: Pipelines use datasets to obtain features and labels
2. **Pipeline + Experiment**: Experiments track pipeline performance and artifacts
3. **CLI + Components**: CLI commands orchestrate the use of other components

## Extension Points

EZFlow is designed to be extended in several ways:

1. **Custom Datasets**: Create new dataset classes by inheriting from `BaseDataset`
2. **Custom Pipelines**: Create new pipeline wrappers by inheriting from `Pipeline`
3. **Custom Tracking Backends**: Add new experiment tracking backends by extending the tracker implementation
4. **Custom CLI Commands**: Add new CLI commands by extending the CLI functionality

## Best Practices

When working with or extending EZFlow, follow these best practices:

1. **Dataset Implementation**:
   - Implement all required methods from `BaseDataset`
   - Ensure data preprocessing is idempotent
   - Use appropriate data types and structures

2. **Pipeline Implementation**:
   - Follow the pipeline interface strictly
   - Handle exceptions and provide meaningful error messages
   - Implement proper model serialization/deserialization

3. **Experiment Tracking**:
   - Track all relevant hyperparameters
   - Use consistent naming conventions for metrics
   - Save all necessary artifacts for reproducibility

4. **General**:
   - Add comprehensive unit tests for new components
   - Document all public methods and classes
   - Follow the coding standards 