# CI/CD Integration

This guide explains how to integrate ragmark evaluation into your CI/CD pipeline using the provided GitHub Actions workflow template.

## Using the Template

1. Copy the workflow template to your repository's `.github/workflows` directory, removing `-template` from the filename:

   ```bash
   cp .github/workflows/ragmark-eval-template.yml .github/workflows/ragmark-eval.yml
   ```

2. Customize the workflow to point to your evaluation configuration and dataset:

   - In both the `evaluate-pr` and `evaluate-main` jobs, update the `rageval run` command to point to your config file.
   - For example, if your config is located at `configs/evaluation.yaml`, change:
     ```yaml
     rageval run ./path/to/config.yaml --output ./pr-results.json
     ```
     to
     ```yaml
     rageval run ./configs/evaluation.yaml --output ./pr-results.json
     ```

   - Ensure your config file correctly references your dataset and any required API keys (see below).

3. If your evaluation uses LLM-based judges (e.g., faithfulness, answer relevance), you will need to provide API keys as secrets in your repository settings:

   - Go to your repository's Settings → Secrets and variables → Actions → New repository secret.
   - Add secrets for the API keys your judges require (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
   - Reference these secrets in your config file (if your config expects them) or set them as environment variables in the workflow steps.

   Example config snippet for an Anthropic judge:
   ```yaml
   judge:
     provider: anthropic
     model: claude-haiku-4-5
     api_key: ${{ secrets.ANTHROPIC_API_KEY }}
   ```

   Note: The workflow template does not automatically pass secrets to the config file; you must either:
   - Embed the secret reference directly in your config file (if using a GitHub Actions workflow that can resolve secrets at runtime), or
   - Pass the API key as an environment variable to the `rageval run` step and reference it in your config via environment variable interpolation (if your config supports it).

   Alternatively, you can avoid storing API keys in the config by setting them as environment variables in the workflow and having your config read from environment variables (see the [configuration](configuration.md) documentation for details).

## What to Expect

After setting up the workflow, each pull request will trigger the evaluation:

- The workflow checks out the PR branch and runs `rageval run` against your configuration, saving the results to `pr-results.json`.
- It separately checks out the `main` branch and runs the same evaluation, saving results to `main-results.json`.
- It then compares the two result files using `rageval compare` and generates a Markdown table showing the difference in each metric.
- This table is posted as a comment on the pull request using the `sticky-pull-request-comment` action, which updates the same comment on subsequent pushes (rather than creating duplicate comments).

## Example PR Comment

The comment will appear as follows:

```markdown
| Metric | main | This PR | Delta |
|---|---|---|---|
| precision_at_5 | 0.850 | 0.820 | -0.030 ⚠️ |
| faithfulness | 0.910 | 0.930 | +0.020 ✅ |
```

- ✅ indicates an improvement (delta greater than the threshold, default 0.02).
- ⚠️ indicates a regression (delta less than -threshold).
- No emoji indicates a negligible change (within ±threshold).
- N/A appears if a metric is present in only one of the result files.

## Quality Gates

In addition to the informational comparison, you can add a quality gate step that fails the pull request if metrics regress beyond defined thresholds.

To add a quality gate step, uncomment the relevant steps in the workflow template (see the template for instructions) and provide a gate configuration.

Example gate config (`gate_config.yaml`):

```yaml
min_score:
  faithfulness: 0.8
max_regression:
  precision_at_5: 0.05
```

The `rageval gate` command compares the PR results against the main branch results and exits with a non-zero code if any rule fails, thereby failing the workflow and the pull request.

## Customizing the Threshold

You can adjust the threshold for what counts as an improvement or regression by using the `--threshold` flag in the `rageval compare` step:

```yaml
- name: Compare results
  id: compare
  run: |
    rageval compare ${{ needs.evaluate-pr.outputs.results-path }} ${{ needs.evaluate-main.outputs.results-path }} --threshold 0.01 --output ./comparison.md
```

## Further Reading
- [Configuration Guide](configuration.md): Learn how to structure your evaluation config.
- [Metrics Reference](metrics.md): Details on available metrics and their requirements.
- [Quickstart](quickstart.md): A quick introduction to running ragmark locally.
