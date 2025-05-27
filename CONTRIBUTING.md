# Contributing to MANXO

Thank you for your interest in contributing to MANXO! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### 1. Reporting Issues

- Check if the issue already exists in [GitHub Issues](https://github.com/mintiasaikoh/manxo/issues)
- Provide a clear description of the problem
- Include steps to reproduce the issue
- Add relevant logs or error messages

### 2. Suggesting Features

- Open a new issue with the `enhancement` label
- Describe the feature and its use case
- Explain why it would be valuable to MANXO

### 3. Code Contributions

#### Getting Started

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes following our coding standards
4. Write tests for your changes
5. Commit with clear messages
6. Push to your fork and submit a Pull Request

#### Coding Standards

Please refer to the coding conventions in [CLAUDE.md](CLAUDE.md):

- Use PascalCase for class names
- Use snake_case for functions and variables
- Add type hints to all functions
- Write docstrings for all classes and functions
- Follow PEP 8 style guide

#### Testing

```bash
# Run all tests
pytest scripts/tests/

# Run specific test file
pytest scripts/tests/test_neural_kb.py

# Run with coverage
pytest --cov=scripts scripts/tests/
```

### 4. Documentation

- Update relevant documentation when adding features
- Keep README.md and CLAUDE.md in sync
- Add docstrings to new code
- Include examples where appropriate

## 📋 Pull Request Process

1. **Update Documentation**: Update README.md and CLAUDE.md if needed
2. **Add Tests**: Ensure your code has appropriate test coverage
3. **Run Tests**: All tests must pass
4. **Code Review**: Wait for review from maintainers
5. **Address Feedback**: Make requested changes promptly

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `priority:critical`: Critical priority
- `priority:high`: High priority
- `priority:medium`: Medium priority
- `priority:low`: Low priority

## 📊 Priority Areas

Current development priorities (see [Issues](https://github.com/mintiasaikoh/manxo/issues)):

1. **Neural Knowledge Base** - Core AI infrastructure
2. **GNN Model Training** - Graph neural network implementation
3. **NLP Integration** - Natural language processing
4. **Testing & Documentation** - Improve coverage

## 🎯 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Assume good intentions

## 📞 Contact

- GitHub Issues: [Project Issues](https://github.com/mintiasaikoh/manxo/issues)
- Discussions: [GitHub Discussions](https://github.com/mintiasaikoh/manxo/discussions)

Thank you for contributing to MANXO! 🎵