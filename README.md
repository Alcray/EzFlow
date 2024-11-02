# bio_ml_handler

A data handler for bioinformatics machine learning tasks, including data loading, processing, and model handling.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/bio_ml_handler.git
   cd bio_ml_handler
   ```

2. Install the package with pip:

   ```bash
   pip install .
   ```

Or, install directly from GitHub:

   ```bash
   pip install git+https://github.com/Alcray/BioML.git
   ```

## Usage

```python
from bio_ml_handler import BioMLDataHandler
# Initialize handler with paths to data folders
handler = BioMLDataHandler(data_path='data', split_data_path='split_data')

# Prepare data in fingerprint format (for model training and evaluation)
handler.prepare_train_data(representation='fingerprint')
handler.prepare_validation_data(representation='fingerprint')
handler.prepare_test_data(representation='fingerprint')

# Train and evaluate the model
handler.train_model()
print("Model Average Precision Score:", handler.evaluate_model())

# Export train_split data to JSONL format with SMILES representation
handler.export_to_jsonl(handler.train_split, 'train_split.jsonl')

# Generate submission file
handler.generate_submission('submission.csv')
```
---
