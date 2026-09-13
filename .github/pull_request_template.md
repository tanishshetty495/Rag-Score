## What does this PR do?

<!-- One or two sentences. -->

## Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Lint passes locally (`ruff check rag_score/ tests/ examples/`)
- [ ] Added tests for new functionality (no real API calls - see `tests/conftest.py` for the `FakeJudge`/callable-adapter pattern)
- [ ] Updated `README.md` if this changes user-facing behavior
- [ ] New optional dependencies (if any) are imported lazily and added to the right extra in `pyproject.toml`

## Related issue

<!-- Closes #123, if applicable -->
