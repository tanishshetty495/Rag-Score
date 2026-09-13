# Contributing to rag-score

Thanks for considering a contribution. This project is intentionally small and dependency-light, so the bar for a good PR is: does it fit that philosophy?

## Setup

```bash
git clone https://github.com/tanishcode-12/Rag-Score.git
cd Rag-Score
pip install -e ".[dev]"
```

This installs every optional extra needed to run the full test suite (jinja2, pyyaml, pandas, openai, anthropic, langchain-core, llama-index-core) alongside pytest and ruff.

## Running tests

```bash
pytest tests/ -v
```

All 107 tests run with **zero API keys and zero network access** - LLM judges are tested against a `FakeJudge` fixture, and framework adapters are tested against real LangChain/LlamaIndex base classes rather than mocks of them. If your change needs a new test, follow that pattern: no real API calls in the test suite, ever.

## Linting

```bash
ruff check rag_score/ tests/ examples/
```

CI runs this on every PR. Fix anything it flags before requesting review; `ruff check --fix` handles most issues automatically.

## Design principles to keep in mind

These are the things a PR review will check for, so it's worth reading before you start:

- **Zero lock-in**: don't add code that only works with one specific framework outside of `adapters/`. New retriever/generator integrations belong in `adapters/`, behind their own optional dependency.
- **Offline-first stays offline-first**: `metrics/retrieval/` must never require an API key or network access. New metrics that need an LLM judge belong in `metrics/generation/`.
- **Optional dependencies stay optional**: anything beyond `pydantic` and `click` (the core install) must be imported lazily, inside the function/class that needs it, with a clear `pip install rag-score[extra]` error message on failure - not at module import time.
- **Every metric returns 0.0 on insufficient input, never raises**: a single malformed test case shouldn't crash a whole evaluation run. Reserve exceptions for genuine bugs (see `metrics/base.py`'s docstring).

## Adding a new metric

1. Pick the right home: `metrics/retrieval/` (offline, pure math) or `metrics/generation/` (needs an `LLMJudge`).
2. Subclass `Metric`, set `name` and `requires_api_key`, implement `async def score(...)`.
3. If it produces a human-readable justification (like the LLM-judge metrics do), also override `score_with_reasoning(...)`.
4. Add it to the CLI's metric-name resolution in `cli.py` (`_build_metric` / `_K_PREFIXES` / `_JUDGE_METRICS`) so it's usable from a config file, not just the Python API.
5. Write tests using the fixtures in `tests/conftest.py` - `sample_test_cases`, `sample_eval_result`, `fake_judge`, etc.

## Adding a new framework adapter

Follow the shape of `adapters/langchain_adapter.py`: import the framework lazily inside `__init__`, raise a clear `ImportError` with the install command if it's missing, `isinstance`-check the wrapped object so mistakes fail fast with a useful message, and add a runnable example to `examples/`.

## Pull requests

- Keep PRs focused - one feature or fix per PR is easier to review and easier to revert if something's wrong.
- Update `README.md` if you're adding a user-facing feature (a new metric, adapter, or CLI option).
- Add tests. A PR that adds functionality with no test coverage will be asked to add some before merge.

## Reporting bugs / requesting features

Use the issue templates - they ask for exactly what's needed to reproduce a bug or evaluate a feature request without a back-and-forth.

## Code of conduct

Be respectful, be patient with newcomers, and keep discussion focused on the work. That's it.
