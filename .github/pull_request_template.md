## Description

<!-- Provide a brief description of your changes -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [ ] Test addition or update

## Related Issues

<!-- Link to related issues using #issue_number -->

Closes #

## Changes Made

<!-- Describe the specific changes in bullet points -->

-
-
-

## Testing

<!-- Describe how you tested your changes -->

- [ ] All existing tests pass (`uv run pytest`)
- [ ] Added new tests for new functionality
- [ ] Tests achieve >80% coverage for new code
- [ ] JAX validation tests pass (if applicable): `uv run --group jax-validation pytest -m jax_validation`

## Code Quality

<!-- Ensure your code meets quality standards -->

- [ ] Code is formatted with ruff (`uv run ruff format src/ tests/`)
- [ ] Code passes linting (`uv run ruff check src/ tests/`)
- [ ] Added Google-style docstrings to all public functions
- [ ] Added type hints to all public APIs
- [ ] Documentation builds successfully (`uv run --group docs mkdocs build`)

## Documentation

<!-- Describe documentation updates -->

- [ ] Updated docstrings for modified functions
- [ ] Updated user guides (if user-facing changes)
- [ ] Updated API reference (auto-generated from docstrings)
- [ ] Added/updated examples in docstrings
- [ ] Updated CHANGELOG.md

## Privacy & Security

<!-- For DP-related changes, confirm correctness -->

- [ ] Changes maintain DP guarantees
- [ ] Validated against JAX-Privacy (if applicable)
- [ ] No introduction of privacy leaks
- [ ] Security implications considered

## Checklist

<!-- Final checks before submitting -->

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code where necessary
- [ ] My changes generate no new warnings
- [ ] I have read and followed the [Contributing Guide](../CONTRIBUTING.md)
- [ ] I have read and followed the [TDD Workflow](../docs/development/tdd-workflow.md)

## Additional Context

<!-- Add any other context, screenshots, or information about the PR -->
