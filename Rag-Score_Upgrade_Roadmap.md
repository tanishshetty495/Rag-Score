# Rag-Score — Upgrade & Development Roadmap

> **Rag-Score** is a framework-agnostic evaluation toolkit for RAG systems and, over time, AI agents.  
> Current package baseline: **v0.5.0**.

The goal is to evolve Rag-Score without breaking its existing RAG evaluation functionality.

---

# 1. Product Direction

## Current

```text
RAG System
    ↓
Rag-Score
    ↓
Evaluation Metrics
    ↓
Results
```

## Target

```text
                    AI System
                       │
             ┌─────────┴─────────┐
             │                   │
            RAG                Agent
             │                   │
             └─────────┬─────────┘
                       ↓
                    Rag-Score
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
    Evaluation     Trace Analysis   Experiments
        │              │              │
        └──────────────┼──────────────┘
                       ↓
                Failure Analysis
                       ↓
                 Optimization
                       ↓
                  Re-evaluation
                       ↓
              Regression Protection
```

## Product thesis

> **Rag-Score should answer not only "How good is my RAG system?" but also "Why did it fail, did the agent use its tools correctly, and did a new version actually improve?"**

---

# 2. Protect the Existing Core

Before adding agent functionality, make the existing RAG functionality stable.

## Tasks

- [ ] Freeze and document the current public API
- [ ] Add regression tests for existing metrics
- [ ] Add tests for edge cases
- [ ] Verify Python version compatibility
- [ ] Verify dependency compatibility
- [ ] Improve error messages
- [ ] Add type hints where missing
- [ ] Add API documentation
- [ ] Add examples for the current API
- [ ] Ensure deterministic evaluation where possible
- [ ] Add CI for tests
- [ ] Add package build checks
- [ ] Add versioning/changelog discipline

### Important rule

Do not break existing Rag-Score users when introducing agent evaluation.

Agent evaluation should be an extension of the project, not a rewrite of the original library.

---

# 3. Define a Clean Evaluation Data Model

Before building more metrics, establish a common internal schema.

A RAG evaluation record could contain:

```json
{
  "id": "example_001",
  "question": "What is X?",
  "contexts": [
    "retrieved document..."
  ],
  "answer": "X is...",
  "reference": "X is...",
  "metadata": {}
}
```

An agent evaluation record should extend this concept:

```json
{
  "id": "agent_001",
  "task": "Find the sales decline and create a report",
  "input": {},
  "trajectory": [],
  "tool_calls": [],
  "final_answer": {},
  "expected": {},
  "metadata": {}
}
```

Create internal models such as:

```text
EvaluationCase
EvaluationResult
MetricResult
Trace
ToolCall
ToolResult
Experiment
Failure
```

This becomes the foundation for everything else.

---

# 4. Separate Metrics From Evaluation Engine

Use a modular architecture:

```text
rag_score/
│
├── core/
│   ├── case.py
│   ├── result.py
│   ├── evaluator.py
│   └── registry.py
│
├── metrics/
│   ├── retrieval/
│   ├── generation/
│   ├── groundedness/
│   └── agent/
│
├── traces/
├── experiments/
├── failures/
└── optimization/
```

A metric should have a consistent interface.

Conceptually:

```python
class Metric:
    name: str

    def evaluate(self, case):
        ...
```

This allows users to add custom metrics without modifying the core.

---

# 5. RAG Evaluation Layer

Keep improving the existing RAG capabilities.

## Retrieval metrics

Potential metrics:

- [ ] Recall@K
- [ ] Precision@K
- [ ] Hit Rate
- [ ] MRR
- [ ] NDCG
- [ ] Context relevance
- [ ] Context coverage

## Generation metrics

- [ ] Faithfulness
- [ ] Groundedness
- [ ] Answer relevance
- [ ] Correctness
- [ ] Citation correctness
- [ ] Citation completeness

## Robustness

- [ ] No-answer cases
- [ ] Ambiguous questions
- [ ] Adversarial context
- [ ] Irrelevant retrieval
- [ ] Missing context
- [ ] Contradictory context

Do not implement every metric immediately. Prioritize metrics with clear definitions and reliable tests.

---

# 6. Dataset & Benchmark System

Build a standard benchmark format.

Example:

```json
{
  "name": "customer_support_rag",
  "version": "1.0",
  "cases": [
    {
      "id": "001",
      "question": "...",
      "reference_answer": "...",
      "relevant_context": ["..."]
    }
  ]
}
```

CLI:

```bash
rag-score benchmark dataset.json
```

Output:

```text
Rag-Score Benchmark
────────────────────────────

Cases:          500
Passed:         431
Failed:          69

Faithfulness:   0.89
Answer Rel.:    0.91
Recall@5:       0.84

Latency:        1.82s
```

Add:

- [ ] Dataset versioning
- [ ] Dataset validation
- [ ] Train/evaluation split support
- [ ] Benchmark reproducibility
- [ ] Seed support where applicable

---

# 7. Experiment Tracking

Users should be able to compare RAG versions.

Example:

```text
rag-score run --config baseline.yaml
rag-score run --config improved.yaml
```

Comparison:

```text
Metric                 Baseline     Improved

Faithfulness             0.81          0.89
Answer relevance         0.86          0.91
Recall@5                 0.77          0.84
Latency                  1.6s          1.8s
Cost                     $0.012        $0.015
```

Tasks:

- [ ] Experiment IDs
- [ ] Configuration capture
- [ ] Dataset version capture
- [ ] Model information
- [ ] Metric results
- [ ] Latency
- [ ] Token usage
- [ ] Cost when available
- [ ] Run comparison
- [ ] Export results

---

# 8. Agent Evaluation Layer

This is the major new feature.

Rag-Score should be able to evaluate an AI agent's complete trajectory.

```text
Task
 ↓
Agent
 ↓
Thought/action trace where available
 ↓
Tool call
 ↓
Tool result
 ↓
Tool call
 ↓
Final answer
```

Do not require access to private chain-of-thought.

Evaluate observable behavior:

- tool calls
- tool arguments
- tool results
- final answer
- task outcome
- latency
- errors
- retries

---

# 9. Agent Metrics

Create an agent metrics module.

```text
metrics/agent/
├── task_success.py
├── tool_selection.py
├── tool_arguments.py
├── tool_efficiency.py
├── trajectory.py
├── groundedness.py
├── safety.py
└── efficiency.py
```

Potential metrics:

### Task success

Did the agent accomplish the requested task?

### Tool selection

Did it choose an appropriate tool?

### Tool argument correctness

Were the arguments valid and appropriate?

### Tool efficiency

Did it use unnecessary tools or repeated calls?

### Recovery

Could it recover from tool failures?

### Safety

Did it attempt actions outside its permissions?

### Groundedness

Was the final answer supported by available evidence?

### Efficiency

Measure:

- number of tool calls
- latency
- token usage
- cost
- retries

---

# 10. Trace Format

Define a framework-neutral trace format.

Example:

```json
{
  "trace_id": "trace_001",
  "agent": "research-agent",
  "task": "Find information about X",
  "events": [
    {
      "type": "tool_call",
      "tool": "web.search",
      "arguments": {
        "query": "X"
      }
    },
    {
      "type": "tool_result",
      "tool": "web.search",
      "status": "success",
      "result": {}
    }
  ],
  "final_answer": "..."
}
```

This is extremely important because it lets Rag-Score evaluate agents regardless of framework.

---

# 11. Framework Adapters

Support common agent implementations without making Rag-Score dependent on one framework.

Potential adapters:

```text
adapters/
├── openai/
├── anthropic/
├── langchain/
├── llamaindex/
└── generic/
```

But the priority should be:

1. Generic trace format
2. Simple Python adapter
3. Additional frameworks afterward

---

# 12. Failure Analysis

Once evaluation exists, add automatic failure analysis.

Pipeline:

```text
Evaluation
    ↓
Failed cases
    ↓
Failure extraction
    ↓
Failure categorization
    ↓
Clustering
    ↓
Failure report
```

Example:

```text
127 failures

Retrieval problems       43
Tool selection           31
Unsupported claims       22
Wrong arguments          17
Other                    14
```

Each failure should preserve evidence.

Example:

```text
Failure: Wrong tool selection

Task:
"Create a GitHub issue"

Agent selected:
github.search_repository

Expected:
github.create_issue

Evidence:
tool_call_17.json
```

---

# 13. Failure Taxonomy

Create a standardized taxonomy.

```text
RETRIEVAL_FAILURE
GENERATION_FAILURE
GROUNDING_FAILURE
TOOL_SELECTION_FAILURE
TOOL_ARGUMENT_FAILURE
TOOL_EXECUTION_FAILURE
TASK_FAILURE
PERMISSION_FAILURE
SAFETY_FAILURE
TIMEOUT
LOOP
EXCESSIVE_TOOL_USE
```

This allows meaningful aggregate reports.

---

# 14. Optimization Layer

Only after evaluation and failure analysis are reliable should automated optimization be added.

Architecture:

```text
Baseline Agent
      ↓
Evaluation
      ↓
Failure Analysis
      ↓
Improvement Candidate
      ↓
Evaluation
      ↓
Regression Check
      ↓
Accept / Reject
```

The optimizer should never be allowed to declare its own improvement.

**Rag-Score must independently evaluate the candidate.**

---

# 15. Prompt Optimization

Possible first optimization target:

```text
System Prompt v1
        ↓
Evaluation
        ↓
Failure clusters
        ↓
Prompt candidate
        ↓
Evaluation
        ↓
Compare
```

Only accept if the candidate passes predefined criteria.

Example policy:

```yaml
optimization:
  primary_metric: task_success

  minimum_improvement: 0.02

  regression_limits:
    groundedness: -0.01
    safety: 0.00
```

This prevents an optimizer from improving one metric while seriously damaging another.

---

# 16. Regression Protection

This should become a major feature.

Example:

```text
Baseline:
Task Success     0.84
Groundedness     0.91
Safety           0.97

Candidate:
Task Success     0.89
Groundedness     0.83
Safety           0.95
```

Even though task success increased, the candidate could be rejected because of regression rules.

Concept:

```text
Improvement ≠ One Metric Going Up
```

---

# 17. Statistical Evaluation

For serious benchmark claims, avoid relying on one small sample.

Add:

- [ ] Confidence intervals where appropriate
- [ ] Bootstrap comparisons
- [ ] Paired evaluation
- [ ] Minimum sample requirements
- [ ] Effect size
- [ ] Variance reporting

Output:

```text
Baseline vs Candidate

Task success:
+5.2 percentage points

95% bootstrap CI:
[2.1, 8.4]

Cases:
500
```

The exact statistical method should depend on the metric and evaluation design.

---

# 18. Cost & Performance Evaluation

Add a common performance model.

Track:

```text
Latency
Token usage
API cost
Tool calls
Retries
Memory usage where available
```

Then allow tradeoff analysis:

```text
Version A
Accuracy: 88%
Latency: 1.2s
Cost: $0.008

Version B
Accuracy: 91%
Latency: 3.4s
Cost: $0.031
```

Do not automatically call one version "better." Let the user define optimization constraints.

---

# 19. Evaluation Dashboard

Eventually provide a simple dashboard.

Pages:

```text
Overview
Experiments
Metrics
Failures
Traces
Comparisons
Optimization
Datasets
```

Useful visualizations:

- metric history
- experiment comparison
- failure distribution
- latency
- cost
- tool usage
- regression alerts

The dashboard should remain optional so the core Python package stays lightweight.

---

# 20. CLI Design

Target CLI:

```bash
rag-score evaluate
rag-score benchmark
rag-score run
rag-score compare
rag-score trace
rag-score failures
rag-score optimize
rag-score report
```

Examples:

```bash
rag-score evaluate --dataset eval.json

rag-score compare run_001 run_002

rag-score failures run_002

rag-score optimize \
    --dataset benchmark.json \
    --agent agent.py
```

---

# 21. Configuration

Use a readable configuration format.

Example:

```yaml
project:
  name: customer-support-agent

evaluation:
  dataset: ./datasets/test.json

metrics:
  - task_success
  - groundedness
  - tool_selection
  - tool_efficiency

performance:
  latency: true
  tokens: true
  cost: true

regression:
  enabled: true
```

---

# 22. Testing Strategy

Testing should become a major part of the project.

## Unit tests

- [ ] Metrics
- [ ] Dataset validation
- [ ] Trace parsing
- [ ] Result aggregation
- [ ] Failure classification

## Integration tests

- [ ] Full RAG evaluation
- [ ] Agent trace evaluation
- [ ] Experiment comparison
- [ ] Regression detection

## Golden tests

Maintain known evaluation cases with expected outputs.

## Adversarial tests

Test:

- missing context
- irrelevant context
- malformed traces
- invalid tool calls
- repeated tool calls
- tool failures
- contradictory evidence
- empty answers

---

# 23. Documentation

Documentation should eventually contain:

```text
docs/
├── getting-started.md
├── concepts/
│   ├── metrics.md
│   ├── traces.md
│   ├── experiments.md
│   └── regression.md
│
├── rag/
│   ├── retrieval.md
│   └── generation.md
│
├── agents/
│   ├── agent-evaluation.md
│   ├── tool-use.md
│   └── failure-analysis.md
│
├── optimization/
│   └── prompt-optimization.md
│
└── integrations/
```

---

# 24. Benchmarks

Create public benchmark suites.

Start small.

```text
benchmarks/
├── basic_rag/
├── retrieval/
├── groundedness/
├── tool_use/
└── agent_tasks/
```

Every benchmark should document:

- dataset
- task definition
- metrics
- sample count
- model/version
- evaluation method
- limitations

Do not publish a single number without enough context to reproduce it.

---

# 25. Project Quality

For a serious open-source project:

- [ ] `pyproject.toml`
- [ ] Type checking
- [ ] Linting
- [ ] Formatting
- [ ] Unit tests
- [ ] Integration tests
- [ ] CI
- [ ] Coverage reporting
- [ ] Changelog
- [ ] Semantic versioning
- [ ] API documentation
- [ ] Examples
- [ ] Contribution guide
- [ ] Security policy
- [ ] License
- [ ] Code of conduct

---

# 26. Recommended Version Roadmap

## v0.5.x — Stabilization

Current focus:

- Existing RAG evaluation
- Bug fixes
- Tests
- Documentation
- API stability

---

## v0.6.0 — Evaluation Core

Add:

- Unified evaluation data model
- Metric registry
- Benchmark format
- Experiment runs
- Result comparison
- Better CLI

---

## v0.7.0 — Agent Evaluation

Add:

- Framework-neutral trace schema
- Agent task evaluation
- Tool-use metrics
- Trajectory metrics
- Agent adapters

---

## v0.8.0 — Failure Intelligence

Add:

- Failure taxonomy
- Failure extraction
- Failure clustering
- Failure reports
- Trace analysis

---

## v0.9.0 — Optimization

Add:

- Prompt optimization
- Candidate generation
- Automated re-evaluation
- Regression protection
- Multi-metric constraints

---

## v1.0.0 — Stable AI Evaluation Platform

Target:

```text
RAG Evaluation
+
Agent Evaluation
+
Experiments
+
Failure Analysis
+
Optimization
+
Regression Protection
```

with a stable public API.

---

# 27. Integration With AgentConnect

Long term:

```text
                  AI Agent
                     │
                     ▼
               AgentConnect
                     │
              Tool execution
                     │
                     ▼
                Trace data
                     │
                     ▼
                 Rag-Score
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Evaluation     Failures      Metrics
       │             │             │
       └─────────────┼─────────────┘
                     ▼
                 Optimizer
                     │
                     ▼
               New Agent Version
                     │
                     └──────→ Re-evaluate
```

AgentConnect answers:

> **What is the agent allowed to do and what did it do?**

Rag-Score answers:

> **Did the agent do it correctly?**

---

# 28. Priority Order

Do NOT implement everything simultaneously.

Recommended order:

### Priority 1 — Stabilize

- [ ] Existing tests
- [ ] API documentation
- [ ] CI
- [ ] Error handling
- [ ] Package quality

### Priority 2 — Evaluation Model

- [ ] EvaluationCase
- [ ] EvaluationResult
- [ ] Metric interface
- [ ] Dataset format
- [ ] Experiment format

### Priority 3 — Benchmarks

- [ ] Benchmark runner
- [ ] Dataset validation
- [ ] Result comparison

### Priority 4 — Agent Evaluation

- [ ] Trace schema
- [ ] Tool-call evaluation
- [ ] Task-success evaluation
- [ ] Agent metrics

### Priority 5 — Failure Analysis

- [ ] Failure taxonomy
- [ ] Failure extraction
- [ ] Failure clustering

### Priority 6 — Regression

- [ ] Baselines
- [ ] Thresholds
- [ ] Multi-metric regression checks

### Priority 7 — Optimization

- [ ] Candidate generation
- [ ] Prompt optimization
- [ ] Re-evaluation
- [ ] Acceptance/rejection logic

### Priority 8 — Dashboard

- [ ] Experiments
- [ ] Metrics
- [ ] Failures
- [ ] Traces

---

# 29. What NOT To Build Yet

Avoid scope explosion.

Do not initially build:

- A full agent framework
- A new vector database
- A new LLM
- A custom orchestration engine
- Dozens of integrations
- A huge UI
- Automatic production deployment
- Autonomous unrestricted self-modification

Rag-Score should remain primarily an **evaluation and reliability platform**.

---

# 30. Final Target Architecture

```text
rag-score/
│
├── rag_score/
│   │
│   ├── core/
│   │   ├── case.py
│   │   ├── result.py
│   │   ├── evaluator.py
│   │   └── registry.py
│   │
│   ├── metrics/
│   │   ├── retrieval/
│   │   ├── generation/
│   │   ├── grounding/
│   │   └── agent/
│   │
│   ├── datasets/
│   │
│   ├── benchmarks/
│   │
│   ├── traces/
│   │
│   ├── experiments/
│   │
│   ├── failures/
│   │
│   ├── optimization/
│   │
│   ├── adapters/
│   │
│   └── cli/
│
├── tests/
├── examples/
├── benchmarks/
├── docs/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

---

# The End Goal

Rag-Score should become:

> **A framework-agnostic evaluation and reliability platform for RAG systems and AI agents.**

The key differentiator is not simply having many metrics.

The important pipeline is:

```text
             RUN AI SYSTEM
                  ↓
              EVALUATE
                  ↓
            UNDERSTAND FAILURE
                  ↓
          GENERATE IMPROVEMENT
                  ↓
              RE-EVALUATE
                  ↓
         CHECK FOR REGRESSIONS
                  ↓
              ACCEPT / REJECT
```

That turns Rag-Score from a metric library into an **AI evaluation engineering platform**.

---

# Immediate Next Steps

Before writing the new agent features:

1. Inspect the current `ragmark`/Rag-Score v0.5.0 codebase.
2. Identify the existing public API.
3. Run the current test suite.
4. Document the current architecture.
5. Create the new internal evaluation data model.
6. Add regression tests around the existing behavior.
7. Add the framework-neutral trace format.
8. Implement the first agent metric: `task_success`.
9. Implement `tool_selection`.
10. Add benchmark comparison.
11. Only then begin failure analysis and optimization.

**Rule:** preserve the working v0.5.0 functionality while expanding the architecture around it.
