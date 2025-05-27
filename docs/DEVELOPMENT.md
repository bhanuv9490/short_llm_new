# Development Guide

This guide provides information for developers working on the Short LLM project.

## Project Structure

- `src/`: Main source code
  - `data/`: Data loading and processing
  - `models/`: Model implementations
  - `utils/`: Utility functions
  - `main.py`: Main application entry point

## Adding a New Model

1. Create a new Python file in `models/saved_models/` (e.g., `my_model.py`)
2. Implement your model class with the required methods
3. Update `LLMInterface` in `src/main.py` to include your new model

Example model implementation:

```python
class MyCustomModel:
    def __init__(self):
        # Initialize your model here
        pass
        
    def predict(self, text):
        # Implement your prediction logic
        return f"Processed: {text}"
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_models.py

# Run with coverage report
pytest --cov=src tests/
```

### Writing Tests

- Place test files in the `tests/` directory
- Follow the naming convention `test_*.py`
- Use pytest fixtures for common test setup

Example test:

```python
def test_llm_interface():
    interface = LLMInterface()
    result = interface.process_text_with_llm1("test")
    assert isinstance(result, str)
```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Keep lines under 88 characters

## Git Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add my new feature"
   ```

3. Push your branch and create a pull request

## Versioning

This project uses [Semantic Versioning](https://semver.org/).

- MAJOR version for incompatible API changes
- MINOR version for added functionality in a backward-compatible manner
- PATCH version for backward-compatible bug fixes
