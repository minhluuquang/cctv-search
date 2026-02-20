# AGENTS.md - Guidelines for AI Coding Agents

## Build, Test, and Lint Commands

### Package Management (uv)
```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_example.py

# Run a single test function
uv run pytest tests/test_example.py::test_function_name

# Run with verbose output
uv run pytest -v

# Run with coverage (if coverage tool installed)
uv run pytest --cov=src/cctv_search --cov-report=term-missing
```

### Linting and Formatting
```bash
# Check code with ruff
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check specific file
uv run ruff check src/cctv_search/module.py
```

### Running the Application
```bash
# Run the CLI
uv run cctv-search

# Run a Python script
uv run python src/cctv_search/__init__.py
```

## Code Style Guidelines

### Imports
- Use absolute imports: `from cctv_search.module import function`
- Group imports in this order: stdlib, third-party, local
- Use explicit imports over wildcard imports
- Sort imports automatically with ruff (configured in pyproject.toml)

### Formatting
- Line length: 88 characters (black-compatible)
- Use double quotes for strings
- Use 4 spaces for indentation
- Trailing commas for multi-line structures
- Run `uv run ruff format .` before committing

### Type Hints
- Use type hints for function parameters and return values
- Use `from __future__ import annotations` for forward references
- Use `typing.Optional` or `| None` syntax for nullable types
- Use `typing.Union` or `|` syntax for union types

### Naming Conventions
- Classes: PascalCase (e.g., `VideoProcessor`)
- Functions/variables: snake_case (e.g., `search_videos`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_RETRY_COUNT`)
- Private methods/variables: _leading_underscore
- Test files: test_*.py
- Test functions: test_*

### Docstrings
- Use triple double quotes for all docstrings
- Include description, args, returns, and raises sections
- Follow Google style or NumPy style consistently
- All public modules, classes, and functions must have docstrings

```python
def example_function(param: str) -> int:
    """Short description.

    Longer description if needed.

    Args:
        param: Description of parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: When input is invalid.
    """
    pass
```

### Error Handling
- Use specific exceptions over generic `Exception`
- Catch exceptions at appropriate levels, don't suppress silently
- Use context managers (`with` statements) for resource cleanup
- Log exceptions with context before re-raising
- Prefer `raise ValueError("message")` over `raise ValueError`

### Code Structure
- Follow src layout: code lives in `src/cctv_search/`
- Keep functions focused and under 50 lines when possible
- Avoid deeply nested structures (max 3-4 levels)
- Use early returns to reduce nesting
- Separate I/O from business logic

### Testing
- Write tests for all new functionality
- Use descriptive test names: `test_should_do_something_when_condition`
- Follow Arrange-Act-Assert pattern
- Use pytest fixtures for common setup
- Mock external dependencies appropriately
- Aim for high coverage on business logic

### Git Workflow
- Create feature branches from main
- Write clear commit messages
- Run tests and linting before committing
- Keep commits focused on single changes

### Dependencies
- Pin versions in pyproject.toml for reproducibility
- Keep dependencies minimal
- Document why each dependency is needed
- Update uv.lock after dependency changes
