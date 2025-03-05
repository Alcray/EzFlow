# Contributing to EZFlow

Thank you for considering contributing to EZFlow! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Commit Guidelines](#commit-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers understand your report and reproduce the issue.

**Before Submitting A Bug Report:**
- Check the documentation for a solution to your issue
- Check if the bug has already been reported in the Issues section
- Verify that the issue can be reproduced reliably

**How Do I Submit A Good Bug Report?**

Bugs are tracked as GitHub issues. Create an issue and provide the following information:

- **Title**: Use a clear and descriptive title
- **Description**: Provide a detailed description of the issue
- **Steps to Reproduce**: List all steps needed to reproduce the behavior
- **Expected Behavior**: Describe what you expected to happen
- **Actual Behavior**: Describe what actually happened
- **Environment**: Include Python version, OS, package versions, etc.
- **Additional Context**: Add any other context about the problem

### Suggesting Features

This section guides you through submitting a feature suggestion:

- **Title**: Use a clear and descriptive title
- **Description**: Provide a step-by-step description of the suggested enhancement
- **Use Case**: Describe the use case that your enhancement would address
- **Alternatives**: Describe alternatives you've considered
- **Additional Context**: Add any other context or examples related to the feature request

### Pull Requests

- Fill in the required pull request template
- Follow the coding standards (see below)
- Include appropriate tests (see below)
- Update documentation if necessary
- Pull requests should be made to the `develop` branch, not `main`

## Development Setup

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/ezflow.git
   cd ezflow
   ```
3. Create a conda environment using the provided environment.yml:
   ```bash
   conda env create -f environment.yml
   conda activate ezflow
   ```
4. Install the package in development mode:
   ```bash
   pip install -e .
   ```
5. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

We use the following tools to maintain code quality:

- **Black**: For consistent code formatting
- **Flake8**: For style guide enforcement
- **isort**: For consistent import ordering
- **mypy**: For static type checking

Our CI pipeline will check these automatically, but you can run them locally:

```bash
# Format code
black ezflow
isort ezflow

# Check style
flake8 ezflow

# Check types
mypy ezflow
```

### General Guidelines

- Follow PEP 8 style guidelines
- Include type hints for function arguments and return values
- Write docstrings in the NumPy/SciPy style
- Keep functions focused on a single responsibility
- Limit line length to 88 characters (Black's default)

## Testing

We use pytest for testing. All new features should include tests.

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=ezflow

# Run specific tests
pytest ezflow/tests/test_specific_module.py
```

## Documentation

- Documentation is written in Markdown
- API documentation is generated from docstrings
- Examples should be clear, concise, and runnable
- Update the relevant documentation files when adding features

## Commit Guidelines

We follow conventional commits:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

Example:
```
feat(dataset): add support for image datasets
```

Thank you for contributing to EZFlow! 