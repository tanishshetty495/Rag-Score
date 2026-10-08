# RAG-Score Architecture

## Overview

RAG-Score is a framework-agnostic RAG (Retrieval-Augmented Generation) evaluation library designed to be zero lock-in and zero forced API cost. It provides a comprehensive suite of metrics for evaluating both traditional retrieve-then-generate pipelines and agentic trajectory-based systems.

## Modules

### Core (`rag_score.core`)
The core module contains the fundamental data models and evaluation pipeline.

**Key Components:**
- `TestCase`: Input representing a question in the evaluation dataset
- `EvalResult`: Raw output from running a retriever/generator against a TestCase
- `MetricScore`: One metric's verdict on one EvalResult
- `Dataset loading`: Functions to load test sets from JSON/YAML files
- `Evaluation runner`: Concurrent pipeline that runs retriever+generator then scores results

### Adapters (`rag_score.adapters`)
Adapter interfaces for connecting to various retrieval and generation frameworks.

**Key Components:**
- `RetrieverAdapter` / `GeneratorAdapter`: Base classes for retrieval and generation
- `CallableRetrieverAdapter` / `CallableGeneratorAdapter`: Wrappers for plain functions
- Framework-specific adapters:
  - `LangChainRetrieverAdapter` / `LangChainGeneratorAdapter`
  - `LlamaIndexRetrieverAdapter` / `LlamaIndexGeneratorAdapter`

### Metrics (`rag_score.metrics`)
Collection of metrics for evaluating RAG performance.

**Retrieval Metrics (offline-first, zero API keys required):**
- `PrecisionAtK`: Fraction of top-k retrieved chunks that are relevant
- `RecallAtK`: Fraction of relevant documents that were retrieved in top-k
- `MRR` (Mean Reciprocal Rank): Reciprocal rank of the first relevant document
- `NDCGAtK` (Normalized Discounted Cumulative Gain): Ranking quality measure

**Generation Metrics (require LLM judge):**
- `Faithfulness`: Whether the answer is grounded in the retrieved context
- `AnswerRelevance`: How well the answer addresses the question
- `ContextPrecision`: Whether the retrieved context is relevant to the question
- `ContextCarryOver`: How well multi-turn conversations resolve references

**Local ML Metrics (zero API calls after initial model download):**
- `LocalFaithfulness`: Semantic similarity via local embedding model
- `LocalAnswerRelevance`: Semantic similarity via local embedding model

### Agentic (`rag_score.agentic`)
Components for evaluating multi-step agent trajectories.

**Key Components:**
- `TrajectoryTestCase`: Input representing a question for agentic evaluation
- `TrajectoryEvalResult`: Full recording of an agent's trajectory (tool calls + final answer)
- `TrajectoryMetric`: Base class for trajectory metrics
- Trajectory metrics:
  - `ToolSelectionRecall`: Fraction of expected tools that were actually called
  - `ToolSelectionPrecision`: Fraction of called tools that were expected
  - `ToolCallOrderCorrectness`: Whether tools were called in the expected order

### Judges (`rag_score.judges`)
LLM judge implementations for metrics that require LLM evaluation.

**Key Components:**
- `LLMJudge`: Abstract base class for LLM judges
- `JudgeVerdict`: Score (0-1) plus reasoning from a judge
- Provider implementations:
  - `OpenAIJudge`: Uses OpenAI API
  - `AnthropicJudge`: Uses Anthropic API
  - `LocalJudge`: For Ollama or any OpenAI-compatible local server

### Export (`rag_score.export`)
Data export functionality for integration with BI tools and notebooks.

**Key Components:**
- `export_to_sqlite`: Writes evaluation results to SQLite star schema
- `results_to_dataframe`: Converts results to Pandas DataFrame
- `scores_to_dataframe`: Converts scores to Pandas DataFrame
- `scores_to_wide_dataframe`: Pivots scores for easy analysis
- `full_report_dataframe`: Joins results, scores, and test cases

### Report (`rag_score.report`)
Report generation functionality.

**Key Components:**
- `generate_html_report`: Creates self-contained HTML report with inline SVG charts

### Synthesize (`rag_score.synthesize`)
Synthetic test set generation from raw documents.

**Key Components:**
- `synthesize_test_set`: Main function to generate test sets from documents
- Query types:
  - `standard`: Questions answerable from a single passage
  - `adversarial`: Questions with typos/vague phrasing/ambiguous pronouns
  - `multi_hop`: Questions requiring information from two passages
  - `unanswerable`: Questions that cannot be answered from the context

### Cache (`rag_score.cache`)
Judge response caching to avoid re-paying for identical LLM judge calls.

**Key Components:**
- `JudgeCache`: Protocol for judge caching
- `InMemoryCacheBackend`: Simple in-memory cache
- `FileCacheBackend`: SQLite-based file cache with WAL mode
- `CacheConfig`: Configuration for cache backend
- `get_cache_backend`: Factory function to get appropriate backend

### Telemetry (`rag_score.telemetry`)
Token counting and cost estimation for evaluation runs.

**Key Components:**
- `TelemetryConfig`: Enables token/cost tracking
- `count_tokens`: Estimates tokens using tiktoken with word-based fallback
- `estimate_cost`: Estimates cost in USD based on token counts

### CLI (`rag_score.cli`)
Command-line interface for running evaluations.

**Commands:**
- `rageval run`: Run an evaluation from a config file
- `rageval synthesize`: Generate a synthetic test set from documents
- `rageval compare`: Compare two evaluation result files
- `rageval gate`: Evaluate quality gates against a baseline
- `rageval run-trajectory`: Run an agentic trajectory evaluation

## Data Flow

### Traditional RAG Pipeline
```
TestCase (input)
    ↓
[Retriever → Generator] 
    ↓
EvalResult (retrieved_context + generated_answer)
    ↓
[Metric 1, Metric 2, ..., Metric N] (scoring)
    ↓
MetricScore (individual metric scores)
    ↓
RunReport (collection of EvalResult and MetricScore)
```

### Agentic Trajectory Pipeline
```
TrajectoryTestCase (input)
    ↓
[Agent] 
    ↓
TrajectoryEvalResult (tool_calls + final_answer)
    ↓
[TrajectoryMetric 1, TrajectoryMetric 2, ..., TrajectoryMetric N] (scoring)
    ↓
MetricScore (individual metric scores)
    ↓
RunReport (collection of TrajectoryEvalResult and MetricScore)
```

### Star Schema (SQLite Export)
The SQLite export follows a star schema optimized for BI tools:

**Dimension Tables:**
- `dim_runs`: One row per evaluation run (config, timestamp)
- `dim_test_cases`: One row per question in a dataset (deduped by test_case_id)

**Fact Tables:**
- `fact_evaluations`: One row per (run, test_case) pairing - the actual I/O
- `fact_metric_scores`: One row per (evaluation, metric) - the scored numbers

## Public API

### Python Import API
Everything that can be imported from the `rag_score` package:

```python
# Core
from rag_score import (
    TestCase,
    EvalResult,
    MetricScore,
    load_dataset,
    RunConfig,
    run_evaluation,
    DimRun,
    DimTestCase,
    FactEvaluation,
    FactMetricScore,
)

# Metrics
from rag_score.metrics import (
    PrecisionAtK,
    RecallAtK,
    MRR,
    NDCG,
    Faithfulness,
    AnswerRelevance,
    ContextPrecision,
    ContextCarryOver,
    LocalFaithfulness,
    LocalAnswerRelevance,
)

# Agentic
from rag_score.agentic import (
    TrajectoryTestCase,
    TrajectoryEvalResult,
    ToolSelectionRecall,
    ToolSelectionPrecision,
    ToolCallOrderCorrectness,
)

# Judges
from rag_score.judges import LLMJudge, JudgeVerdict, OpenAIJudge, AnthropicJudge, LocalJudge

# Export
from rag_score.export import (
    export_to_sqlite,
    results_to_dataframe,
    scores_to_dataframe,
    scores_to_wide_dataframe,
    full_report_dataframe,
)

# Report
from rag_score.report import generate_html_report

# Synthesize
from rag_score.synthesize import synthesize_test_set

# Cache
from rag_score.cache import (
    JudgeCache,
    InMemoryCacheBackend,
    FileCacheBackend,
    CacheConfig,
    get_cache_backend,
)

# Telemetry
from rag_score.telemetry import TelemetryConfig, count_tokens, estimate_cost
```

### CLI API
All available CLI commands and their flags:

#### `rageval run`
```
Usage: rageval run CONFIG_PATH

Run an evaluation from a YAML or JSON config file.

Config format:
    dataset: test_set.json
    retriever: my_module:my_retriever
    generator: my_module:my_generator
    metrics: [precision_at_5, recall_at_5, mrr, ndcg_at_5]
    top_k: 5
    max_concurrency: 8
    output: results.json                   # optional
    sqlite_output: results.db              # optional
    html_output: report.html               # optional
    telemetry:                             # optional
      model_name: gpt-4o-mini
      pricing: {...}                       # optional
```

#### `rageval synthesize`
```
Usage: rageval synthesize DOCS_DIR [OPTIONS]

Generate a synthetic test set from documents.

Options:
    --output FILE           Where to write test_set.json (default: test_set.json)
    --judge JSON            Judge config as JSON (required)
    --chunk-size INT        Words per chunk (default: 500)
    --chunk-overlap INT     Overlap between chunks (default: 50)
    --questions-per-chunk   Questions per chunk (default: 1)
    --max-concurrency INT   Concurrent judge calls (default: 5)
    --query-types LIST      Query types: standard,adversarial,multi_hop,unanswerable
```

#### `rageval compare`
```
Usage: rageval compare RESULTS_A.json RESULTS_B.json [OPTIONS]

Compare two evaluation result files.

Options:
    --output FILE           Write markdown table to file
    --threshold FLOAT       Threshold for improvement/regression (default: 0.02)
    --with-significance     Include statistical significance testing
```

#### `rageval gate`
```
Usage: rageval gate RESULTS.json BASELINE.json [OPTIONS]

Evaluate quality gates against a baseline.

Options:
    --min-score METRIC=VALUE    Minimum score required (repeatable)
    --max-regression METRIC=VALUE Maximum allowed regression (repeatable)
    --max-regression-significant METRIC=VALUE 
                                Maximum allowed significant regression (repeatable)
    --significance-level FLOAT  Significance level (default: 0.05)
    --gate-config FILE          Path to JSON/YAML gate config
    --output FILE               Write markdown table to file
```

#### `rageval run-trajectory`
```
Usage: rageval run-trajectory CONFIG_PATH

Run an agentic trajectory evaluation from a config file.

Config format:
    dataset: trajectory_test_set.json
    agent: my_module:my_agent
    metrics: [tool_selection_recall, tool_selection_precision, tool_call_order_correctness]
    max_concurrency: 8
    output: results.json               # optional
```

## Extensibility

### Adding Custom Metrics
1. Inherit from `Metric` (for traditional metrics) or `TrajectoryMetric` (for agentic)
2. Implement the `score()` method
3. Optionally override `score_with_reasoning()` if your metric provides explanations
4. Set appropriate `name` and `requires_api_key` attributes

### Adding Custom Adapters
1. Inherit from `RetrieverAdapter` or `GeneratorAdapter`
2. Implement the `retrieve()` or `generate()` method
3. For callable adapters, use `CallableRetrieverAdapter` or `CallableGeneratorAdapter`

### Adding Custom Judges
1. Inherit from `LLMJudge`
2. Implement the `complete()` method to send prompts to your model
3. The base class handles retry logic, caching, and JSON parsing

## Design Principles

1. **Zero Lock-In**: Framework-agnostic adapters allow use with any retrieval/generation system
2. **Offline-First**: Core retrieval metrics work with zero API keys
3. **Optional LLM Features**: LLM-judge features require explicit opt-in and can be tested with fake judges
4. **No Input Mutation**: Never modifies user input files
5. **Backward Compatibility**: Breaking changes require maintaining old paths with deprecation warnings
6. **Statistical Rigor**: Includes statistical significance testing for regression detection
7. **Determinism**: Seed options available where randomness exists
8. **Extensible**: Clean interfaces for adding custom metrics, adapters, and judges