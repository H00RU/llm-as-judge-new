# Contributing to LLM-as-Judge

First off, thank you for considering contributing to LLM-as-Judge! It's people like you that make this project better.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps to reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed and what behavior you expected
* Include logs and error messages
* Specify your environment (Python version, PyTorch version, GPU model, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* A clear and descriptive title
* A detailed description of the proposed functionality
* Explain why this enhancement would be useful
* List any similar features in other projects

### Pull Requests

* Fill in the required template
* Follow the Python style guide (PEP 8)
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

## Development Process

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llm-as-judge.git
cd llm-as-judge

# Add upstream remote
git remote add upstream https://github.com/beita6969/llm-as-judge.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### Code Style

* Follow PEP 8 guidelines
* Use type hints for function signatures
* Write docstrings for all public functions and classes
* Keep functions focused and under 50 lines when possible
* Use meaningful variable names

**Example:**

```python
def compute_reward(
    prediction: str,
    ground_truth: str,
    problem_type: str
) -> float:
    """Compute reward score for a prediction.

    Args:
        prediction: Model's predicted answer
        ground_truth: Expected correct answer
        problem_type: Type of problem (math/code/qa)

    Returns:
        Reward score between -10.0 and 10.0
    """
    # Implementation
    pass
```

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_llm_judge.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Commit Messages

Follow the conventional commits specification:

* `feat`: A new feature
* `fix`: A bug fix
* `docs`: Documentation changes
* `style`: Code style changes (formatting, etc.)
* `refactor`: Code refactoring
* `test`: Adding or updating tests
* `chore`: Maintenance tasks

**Examples:**
```
feat: add retry mechanism for LLM Judge
fix: resolve UnboundLocalError in workflow generation
docs: update installation instructions
```

### Branch Naming

* `feature/description` - For new features
* `fix/description` - For bug fixes
* `docs/description` - For documentation
* `refactor/description` - For refactoring

## Project Structure

Key directories and their purposes:

```
src/          - Core source code
tests/        - Test files
config/       - Configuration files
data/         - Dataset directory (not in repo)
docs/         - Additional documentation
scripts/      - Utility scripts
```

## Adding New Features

### Adding a New Operator

1. Define the operator in `config/operator_descriptions/`
2. Update `prompt_optimizer.py` with API documentation
3. Add Layer 2 enhancement rules in `operator_prompt_enhancer.py`
4. Write tests in `tests/test_operators.py`
5. Update documentation

### Adding a New Reward Component

1. Implement the metric in `reward_computer.py`
2. Add configuration in `config/training.yaml`
3. Write tests
4. Update documentation

### Adding a New Dataset Type

1. Create dataset loader in `data_manager.py`
2. Add type-specific evaluation in `unified_evaluator.py`
3. Update prompt optimization for the new type
4. Write tests
5. Update documentation

## Documentation

* Update README.md for user-facing changes
* Add docstrings to all new functions and classes
* Update relevant markdown files in `docs/`
* Add examples for new features

## Testing Guidelines

### Unit Tests

* Test each function in isolation
* Mock external dependencies (LLM calls, file I/O)
* Aim for >80% code coverage

### Integration Tests

* Test complete workflows
* Use small sample datasets
* Verify end-to-end functionality

### Performance Tests

* Profile critical paths
* Ensure no memory leaks
* Test with various batch sizes

## Review Process

1. Create a pull request
2. Automated checks will run (linting, tests)
3. A maintainer will review your code
4. Address any feedback
5. Once approved, your PR will be merged

## Community

* Be respectful and inclusive
* Help others learn and grow
* Share knowledge and best practices
* Celebrate successes together

## Questions?

Feel free to ask questions by:
* Opening an issue
* Starting a discussion
* Contacting the maintainers

Thank you for contributing! 🎉
