from setuptools import setup, find_packages
import os

# Read the contents of README.md file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ezflow",
    version="0.1.0",
    description="A lightweight, extensible machine learning framework for hackathons",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alex Hayrapetyan",
    author_email="hayrapetyan.alexan@gmail.com",
    url="https://github.com/Alcray/ezflow",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ez=ezflow.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Core data processing
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        
        # Machine Learning models
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        
        # Visualization
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        
        # Hyperparameter tuning
        "optuna>=2.10.0",
        "hyperopt>=0.2.7",
        
        # Experiment tracking
        "mlflow>=1.14.0",
        
        # Deployment
        "flask>=2.0.0",
        "streamlit>=1.10.0",
        "uvicorn>=0.13.0",
        
        # Utilities
        "joblib>=1.1.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.50.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b0",
            "flake8>=3.9.0",
            "isort>=5.8.0",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        'api': ["flask>=2.0.0"],
        'dashboard': ["streamlit>=1.10.0"],
        'all': [
            "optuna>=2.10.0",
            "hyperopt>=0.2.7",
            "flask>=2.0.0",
            "streamlit>=1.10.0",
        ],
    },
)

# Create directory structure if not exists
def create_dir_structure():
    """Create the directory structure for a new ezflow project."""
    dirs = [
        "ezflow",
        "ezflow/data",
        "ezflow/data/raw",
        "ezflow/data/interim",
        "ezflow/data/processed",
        "ezflow/models",
        "ezflow/src",
        "ezflow/utils",
        "ezflow/notebooks",
        "ezflow/deployment",
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# Uncomment to create the directory structure
# create_dir_structure()
