# AGENTS.md - Guidelines for AI Coding Agents

## Monorepo Overview

This is a pnpm + Turbo monorepo with:
- **apps/web**: Next.js 16 + React 19 frontend (TypeScript)
- **apps/api**: Python FastAPI backend (uv + Ruff)
- **packages/shared-types**: Shared TypeScript types

## Build, Test, and Lint Commands

### Root Level (Turbo)
```bash
# Install dependencies
pnpm install

# Run all builds
pnpm build

# Run all dev servers (uses TUI - interactive terminal UI)
pnpm dev

# Run all linting
pnpm lint

# Run all formatting
pnpm format

# Run all tests
pnpm test

# Type check all packages
pnpm typecheck

# Clean all build artifacts
pnpm clean
```

**Turbo TUI Mode**: The monorepo uses Turborepo's experimental Terminal UI. When running `pnpm dev`:
- Arrow keys (↑/↓) switch between processes
- See logs for selected process on the right
- Shows all running tasks with status indicators
- Press `?` for help with keyboard shortcuts
- Press `q` to quit

### Python API (apps/api)

#### Package Management (uv)
```bash
cd apps/api

# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

#### Running Tests
```bash
cd apps/api

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_nvr.py

# Run a single test function
uv run pytest tests/test_nvr.py::test_nvr_client_init_with_params -v

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=src/cctv_search --cov-report=term-missing
```

#### Linting and Formatting (Ruff)
```bash
cd apps/api

# Check code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check specific file
uv run ruff check src/cctv_search/nvr/hikvision.py
```

#### Running the API
```bash
# From root
pnpm api:dev

# Or directly
cd apps/api && uv run uvicorn cctv_search.api:app --reload --host 0.0.0.0 --port 8000
```

### Next.js Web (apps/web)

```bash
cd apps/web

# Development server
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start

# Lint with ESLint
pnpm lint

# Type check
pnpm typecheck

# Clean build artifacts
pnpm clean
```

### Shared Types (packages/shared-types)

```bash
cd packages/shared-types

# Build TypeScript
pnpm build

# Watch mode
pnpm dev

# Lint
pnpm lint
```

## Code Style Guidelines

### TypeScript / Next.js

#### Imports
- Use `@/*` path alias for local imports (configured in tsconfig.json)
- Group imports: React/Next, third-party, local, styles
- Use named exports over default exports
- Sort imports automatically with ESLint

#### Formatting
- Use TypeScript strict mode
- 2 spaces for indentation
- Semicolons required
- Single quotes for strings
- Trailing commas for multi-line structures

#### Naming Conventions
- Components: PascalCase (e.g., `VideoPlayer.tsx`)
- Hooks: camelCase with `use` prefix (e.g., `useVideoSearch`)
- Types/Interfaces: PascalCase (e.g., `SearchResult`)
- Constants: UPPER_SNAKE_CASE
- Files: kebab-case or PascalCase for components

#### React Patterns
- Use Server Components by default
- Mark Client Components explicitly with `'use client'`
- Prefer async/await for data fetching in Server Components
- Use React Query for client-side data fetching

### Python

#### Imports
- Use absolute imports: `from cctv_search.module import function`
- Group imports: stdlib, third-party, local
- Use explicit imports over wildcard imports
- Sort imports with Ruff (configured in pyproject.toml)

#### Formatting
- Line length: 88 characters (configured in pyproject.toml)
- Use double quotes for strings
- 4 spaces for indentation
- Trailing commas for multi-line structures

#### Type Hints
- Use type hints for all function parameters and return values
- Use `from __future__ import annotations` for forward references
- Use `|` syntax for unions (e.g., `str | None`)

#### Naming Conventions
- Classes: PascalCase (e.g., `VideoProcessor`)
- Functions/variables: snake_case (e.g., `search_videos`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_RETRY_COUNT`)
- Private methods: _leading_underscore
- Test files: test_*.py
- Test functions: test_*

### Docstrings (Python)
- Use triple double quotes
- Include description, args, returns, and raises sections
- Follow Google style consistently

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

#### TypeScript
- Use specific error types
- Handle errors at appropriate levels
- Use try/catch with proper error boundaries in React

#### Python
- Use specific exceptions over generic `Exception`
- Catch exceptions at appropriate levels, don't suppress silently
- Use context managers (`with` statements) for resource cleanup
- Log exceptions with context before re-raising

### Code Structure

#### General
- Keep functions focused and under 50 lines when possible
- Avoid deeply nested structures (max 3-4 levels)
- Use early returns to reduce nesting
- Separate I/O from business logic

#### Python
- Follow src layout: code lives in `apps/api/src/cctv_search/`
- Tests live in `apps/api/tests/`

#### TypeScript
- Next.js app directory: `apps/web/app/`
- Components: `apps/web/components/`
- Shared types: `packages/shared-types/src/`

### Testing

#### Python
- Write tests for all new functionality
- Use descriptive test names: `test_should_do_something_when_condition`
- Follow Arrange-Act-Assert pattern
- Use pytest fixtures for common setup
- Mock external dependencies appropriately

#### TypeScript
- Write tests for utilities and complex logic
- Use Vitest or Jest (if configured)
- Test hooks with React Testing Library

### Git Workflow
- Create feature branches from main
- Write clear commit messages
- Run tests and linting before committing
- Keep commits focused on single changes

### Dependencies

#### TypeScript
- Use workspace protocol for local packages: `"@cctv-search/shared-types": "workspace:*"`
- Pin major versions in package.json
- Update pnpm-lock.yaml after dependency changes

#### Python
- Pin versions in pyproject.toml for reproducibility
- Keep dependencies minimal
- Update uv.lock after dependency changes
