#!/usr/bin/env python
"""
Script to evaluate the model's performance on the Iris dataset
"""

import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the true labels from the test manifest
true_labels = []
with open('data/processed/test_manifest.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        true_labels.append(data['target'])

# Load the predictions
predictions = []
with open('predictions.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        predictions.append(data['prediction'])

# Convert to numpy arrays
true_labels = np.array(true_labels)
predictions = np.array(predictions)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
target_names = ['setosa', 'versicolor', 'virginica']
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=target_names))

# Create confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")

# Print some example predictions
print("\nSample Predictions (True vs Predicted):")
for i in range(min(10, len(true_labels))):
    true_class = target_names[int(true_labels[i])]
    pred_class = target_names[int(predictions[i])]
    print(f"Sample {i+1}: True: {true_class}, Predicted: {pred_class}") 