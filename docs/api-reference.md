# API reference

Auto-generated from docstrings in the source.

## Core

::: rag_score.core.types.TestCase
::: rag_score.core.types.RetrievedChunk
::: rag_score.core.types.EvalResult
::: rag_score.core.types.MetricScore
::: rag_score.core.dataset.load_dataset
::: rag_score.core.runner.run_evaluation
::: rag_score.core.runner.RunConfig

## Adapters

::: rag_score.adapters.base.RetrieverAdapter
::: rag_score.adapters.base.GeneratorAdapter
::: rag_score.adapters.langchain_adapter.LangChainRetrieverAdapter
::: rag_score.adapters.langchain_adapter.LangChainGeneratorAdapter
::: rag_score.adapters.llamaindex_adapter.LlamaIndexRetrieverAdapter
::: rag_score.adapters.llamaindex_adapter.LlamaIndexGeneratorAdapter

## Metrics

::: rag_score.metrics.base.Metric
::: rag_score.metrics.retrieval.precision_at_k.PrecisionAtK
::: rag_score.metrics.retrieval.recall_at_k.RecallAtK
::: rag_score.metrics.retrieval.mrr.MRR
::: rag_score.metrics.retrieval.ndcg.NDCG
::: rag_score.metrics.generation.faithfulness.Faithfulness
::: rag_score.metrics.generation.answer_relevance.AnswerRelevance
::: rag_score.metrics.generation.context_precision.ContextPrecision
::: rag_score.metrics.generation.local_faithfulness.LocalSemanticFaithfulness
::: rag_score.metrics.generation.local_answer_relevance.LocalSemanticAnswerRelevance

## Judges

::: rag_score.judges.base.LLMJudge
::: rag_score.judges.base.JudgeVerdict
::: rag_score.judges.openai_judge.OpenAIJudge
::: rag_score.judges.anthropic_judge.AnthropicJudge
::: rag_score.judges.local_judge.LocalJudge

## Export

::: rag_score.export.sqlite_export.export_to_sqlite
::: rag_score.export.dataframe_export.results_to_dataframe
::: rag_score.export.dataframe_export.scores_to_dataframe
::: rag_score.export.dataframe_export.scores_to_wide_dataframe
::: rag_score.export.dataframe_export.full_report_dataframe

## Synthesis

::: rag_score.synthesize.synthesize_test_set
::: rag_score.synthesize.chunk_text
::: rag_score.synthesize.load_documents_from_dir
::: rag_score.synthesize.SynthesisReport

## Agentic trajectory evaluation

::: rag_score.agentic.types.ToolCall
::: rag_score.agentic.types.TrajectoryTestCase
::: rag_score.agentic.types.TrajectoryEvalResult
::: rag_score.agentic.types.load_trajectory_dataset
::: rag_score.agentic.adapters.AgentAdapter
::: rag_score.agentic.runner.run_trajectory_evaluation
::: rag_score.agentic.runner.TrajectoryRunConfig
::: rag_score.agentic.metrics_base.TrajectoryMetric
::: rag_score.agentic.metrics.tool_selection_recall.ToolSelectionRecall
::: rag_score.agentic.metrics.tool_selection_precision.ToolSelectionPrecision
::: rag_score.agentic.metrics.tool_call_order.ToolCallOrderCorrectness

## Report

::: rag_score.report.html_report.generate_html_report
