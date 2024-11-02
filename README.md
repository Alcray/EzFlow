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
   pip install git+https://github.com/yourusername/bio_ml_handler.git
   ```

## Usage

```python
from bio_ml_handler import BioMLDataHandler

# Initialize handler without data path
handler = BioMLDataHandler()

# Download and load data
bio_ml_path = handler.download_data(competition_name='bio-ml')

# Prepare data
handler.prepare_train_data()
handler.train_model()
print("Model Average Precision Score:", handler.evaluate_model())

# Generate submission file
handler.generate_submission('submission.csv')
```
---
