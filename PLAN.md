# PR 3 Implementation Plan: Experiments, Performance Tracking and Statistics

## Overview
This plan implements PR 3 of 7: "Experiments, performance tracking and statistics" (completes v0.6.0)
Based on PR3_PROMPT.md instructions.

## STEP 1 - Experiments (rag_score/experiments/)

### New Files to Create:
1. `rag_score/experiments/__init__.py` - Package initializer
2. `rag_score/experiments/models.py` - Experiment/Run models:
   - RunModel: run_id, timestamp, config snapshot, dataset name + version + hash, model info
   - ExperimentResult: metric results, per-case scores, latency, token usage, cost
3. `rag_score/experiments/storage.py` - JSON-based storage system:
   - Save runs to configurable directory (default .rag_score/runs)
   - Prevent overwriting existing runs
   - Load runs by ID or list all runs
4. `rag_score/experiments/manager.py` - Experiment manager:
   - Create, save, load, and list experiments
   - Integration with existing evaluation runner

### Existing Code to Touch:
- `rag_score/core/runner.py` - Modify to accept experiment tracking callback
- `rag_score/core/types.py` - May need to extend for experiment metadata

## STEP 2 - Performance Model

### New Files to Create:
1. `rag_score/performance.py` - Performance tracking utilities:
   - Per-case latency tracking
   - Token counting integration (build on existing telemetry)
   - Tool/API call counting via adapters
   - Retry tracking
   - Cost calculation from user-supplied price table
   - Aggregation functions (mean, p50, p95)

### Existing Code to Touch:
- `rag_score/core/runner.py` - Add performance metric collection
- `rag_score/telemetry.py` - Extend for token counting if needed
- Adapter classes to track API calls

## STEP 3 - Statistics (rag_score/stats.py)

### Enhancements to Existing File:
1. Add paired comparison functions:
   - Paired t-test for normally distributed per-case differences
   - Paired bootstrap for non-normal distributions
2. Add Welch's t-test for independent samples (case sets differ)
3. Add McNemar's test or bootstrap for binary metrics
4. Add bootstrap confidence intervals with fixed default seed
5. Add effect size calculations:
   - Cohen's d for continuous metrics
   - Mean delta with confidence interval
6. Add variance reporting functions
7. Add minimum-sample warnings
8. Add Holm correction for multiple comparisons across metrics
9. Create StatisticalResult object:
   - method used
   - p-value
   - confidence interval
   - effect size
   - sample size (n)
   - status: significant / not_significant / insufficient_data
10. Add safe handling for degenerate cases:
    - n<2
    - zero variance
    - identical runs
    - NaN or missing scores
    - mismatched case ids

### Existing Code to Touch:
- Keep existing `welch_t_test`, `bootstrap_confidence_interval`, `test_significance`, `extract_metric_scores` functions
- Enhance `test_significance` to delegate to new specialized functions

## STEP 4 - CLI

### New Commands to Add:
1. `rageval run --config <file>` - Existing run command enhanced with experiment tracking
2. `rageval compare <run_a> <run_b>` - Compare two experiments
3. `rageval runs` - List all stored experiments
4. `rageval export <run> --format json|csv|md` - Export experiment results

### Existing Code to Touch:
- `rag_score/cli.py` - Add new commands and enhance existing run command
- Update command help texts and examples

## STEP 5 - Tests and Docs

### New Test Files:
1. `tests/test_experiments.py` - Experiment model and storage tests
2. `tests/test_performance.py` - Performance model tests
3. `tests/test_stats_enhanced.py` - Enhanced statistics tests
4. `tests/test_cli_experiments.py` - CLI command tests

### Documentation:
1. `docs/concepts/experiments.md` - Experiments concept documentation
2. Update existing stats documentation
3. Update CHANGELOG.md for v0.6.0
4. Update README if needed

### Version Bump:
- Update version in `pyproject.toml` or equivalent to 0.6.0

## Dependencies:
- Keep stdlib-only requirement for core statistics (numpy optional, scipy optional)
- No breaking changes to public API or CLI
- Core remains offline-capable
- No writing to user input files

## Implementation Order:
1. STEP 1: Experiments module
2. STEP 2: Performance model
3. STEP 3: Enhanced statistics
4. STEP 4: CLI integration
5. STEP 5: Tests and documentation

## Integration Points:
- Experiments will use existing EvaluationCase and EvaluationResult models
- Performance tracking will build on existing telemetry and adapter systems
- Statistics will enhance existing stats.py while maintaining backward compatibility
- CLI will extend existing commands with new subcommands