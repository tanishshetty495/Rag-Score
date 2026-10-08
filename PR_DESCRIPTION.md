## Summary
Prepares Rag-Score for the v0.6-v1.0 roadmap by locking down existing v0.5.0 behavior. No runtime behavior changes.

## Backward compatibility
- All 215 original tests pass unmodified
- No public names, CLI commands, flags or config keys removed or renamed

## How to test
pytest; ruff check .; mypy rag_score; python -m build; twine check dist/*
