# Contributing to Hyperlogica

Thank you for your interest in contributing to Hyperlogica! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker to report bugs
- Describe the bug and include steps to reproduce
- Include any relevant error messages or logs
- Specify your operating system and Python version

### Suggesting Enhancements

- Use the GitHub issue tracker to suggest enhancements
- Clearly describe the enhancement and its expected benefits
- Provide examples of how the enhancement would be used

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## Development Setup

1. Clone the repository:
```
git clone https://github.com/hyperlogica-ai/hyperlogica-core.git
cd hyperlogica-core
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. Install dependencies:
```
pip install -e ".[dev]"
```

4. Setup pre-commit hooks:
```
pre-commit install
```

## Coding Standards

- Follow PEP 8 guidelines
- Include docstrings for all functions, classes, and modules
- Add type hints to function signatures
- Write unit tests for all new features
- Maintain 80% or higher test coverage

## Testing

Run tests with pytest:
```
pytest
```

For test coverage:
pytest --cov=hyperlogica tests/

## Documentation

- Update documentation when adding or changing features
- Follow Google style docstring format
- Add examples for new functionality

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add feature" not "Added feature")
- Reference issue numbers when applicable

## Versioning

We use [Semantic Versioning](https://semver.org/) for releases:
- MAJOR version for incompatible API changes
- MINOR version for backward-compatible new functionality
- PATCH version for backward-compatible bug fixes

## License

By contributing to Hyperlogica, you agree that your contributions will be licensed under the project's Apache 2.0 license.