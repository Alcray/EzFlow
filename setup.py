from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ezflow-ml",
    version="0.1.0",
    author="Alexan Hayrapetyan",
    author_email="hayrapetyan.alexan@gmail.com",
    description="A lightweight and easy-to-use ML experiment pipeline framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Alcray/ezflow",
    packages=find_packages(include=["ezflow", "ezflow.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "torch>=1.8.0",
        "tensorflow>=2.4.0",
        "rdkit>=2022.3.1",
        "mlflow>=1.20.0",
        "tensorboard>=2.4.0",
        "optuna>=2.8.0",
        "dash>=2.0.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.8.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ezflow=ezflow.cli:main",
        ],
    },
)
