from setuptools import setup, find_packages

# Read requirements.txt for dependencies
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ezflow",
    version="0.1.0",
    description="A streamlined framework for running and tracking ML experiments for hackathons.",
    author="Alexan Hayrapetyan",
    author_email="hayrapetyan.alexan@gmail.com",
    url="https://github.com/Alcray/BioML",
    packages=find_packages(),
    install_requires=required,  # Use the list from requirements.txt
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
