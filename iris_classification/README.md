# Iris Classification with ezflow

This project demonstrates how to use the ezflow framework to classify the Iris dataset using XGBoost.

## Project Structure

```
iris_classification/
├── data/
│   ├── processed/
│   │   ├── train_manifest.jsonl  # Training data in manifest format
│   │   └── test_manifest.jsonl   # Test data in manifest format
│   └── raw/                      # Raw data directory
├── models/
│   └── iris_model.pkl            # Trained XGBoost model
├── prepare_iris_data.py          # Script to prepare the Iris dataset
├── evaluate_model.py             # Script to evaluate model performance
├── predictions.jsonl             # Model predictions on test data
└── confusion_matrix.png          # Confusion matrix visualization
```

## Workflow

1. **Data Preparation**: The Iris dataset is loaded from scikit-learn and converted to manifest format.
   ```bash
   python prepare_iris_data.py
   ```

2. **Model Training**: An XGBoost model is trained on the training data.
   ```bash
   ez train --model xgboost --manifest data/processed/train_manifest.jsonl --output models/iris_model.pkl --params '{"n_estimators": 100, "max_depth": 3}'
   ```

3. **Making Predictions**: The trained model is used to make predictions on the test data.
   ```bash
   ez predict --model models/iris_model.pkl --manifest data/processed/test_manifest.jsonl --output predictions.jsonl
   ```

4. **Model Evaluation**: The model's performance is evaluated using accuracy, classification report, and confusion matrix.
   ```bash
   python evaluate_model.py
   ```

## Results

The XGBoost model achieved 100% accuracy on the test set, correctly classifying all Iris flowers into their respective species (setosa, versicolor, and virginica).

## Features Used

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Model Parameters

- n_estimators: 100
- max_depth: 3
- learning_rate: 0.1 (default)
- objective: multi:softprob (automatically set for multiclass classification) 