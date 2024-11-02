from setuptools import setup, find_packages

setup(
    name="bio_ml_handler",
    version="0.1.0",
    description="A data handler for bio-ML projects, including data loading, processing, and model handling.",
    author="Alexan Hayrapetyan",
    author_email="hayrapetyan.alexan@gmail.com",
    url="https://github.com/Alcray/BioML",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "rdkit-pypi",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
