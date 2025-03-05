# 🚀 ezflow: Machine Learning Framework for Hackathons

ezflow is a lightweight, flexible machine learning framework designed specifically for hackathons and rapid prototyping. It provides a streamlined workflow for data processing, model training, and deployment while maintaining code quality and reproducibility.

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
git clone https://github.com/yourusername/ezflow.git
cd ezflow

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

## 🏃‍♂️ Quick Start

1. Initialize a new project:
```bash
python -m ezflow init my_project
cd my_project
```

2. Place your data in `data/raw/`

3. Configure your project in `utils/config.py`

4. Train a model:
```bash
python -m ezflow train --config utils/config.py
```

5. Make predictions:
```bash
python -m ezflow predict --input data/test.csv --model models/model.pkl --output predictions.csv
```

6. Deploy your model:
```bash
# As REST API
python -m ezflow deploy --model models/model.pkl --type api

# As interactive dashboard
python -m ezflow deploy --model models/model.pkl --type dashboard
```

## 📁 Project Structure

```
my_project/
├── data/
│   ├── raw/          # Original data
│   ├── interim/      # Intermediate processed data
│   └── processed/    # Final processed data
├── models/           # Trained models and predictions
├── notebooks/        # Jupyter notebooks
├── src/             # Source code
├── utils/           # Utility functions
├── deployment/      # Deployment configurations
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 