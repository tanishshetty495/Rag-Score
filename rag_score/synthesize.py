"""
Synthetic test-set generation.

The biggest friction point in adopting an eval tool is writing the
test set by hand. This module removes that: chunk your existing docs,
have an LLM judge generate a question + ground-truth answer per chunk,
and get a real test_set.json out - the exact TestCase shape the rest
of the library already expects, so synthesized output plugs straight
into load_dataset() and every metric with zero glue code.

Deliberately reuses LLMJudge rather than inventing a separate
"generator" concept - any judge (OpenAI, Anthropic, local Ollama) that
already works for scoring also works for synthesis, since both are
just "send a prompt, parse structured JSON back".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from rag_score.core.types import TestCase
from rag_score.judges.base import LLMJudge, extract_json_object

_SYSTEM_PROMPT = """You are creating evaluation data for a RAG (Retrieval-Augmented \
Generation) system. Given a passage of text, write ONE question that can be \
answered using ONLY the information in this passage, along with the correct answer.

Rules:
- The question must be answerable from the passage alone, with no outside knowledge.
- The question should be specific, not a vague summary request.
- The answer should be concise and directly supported by the passage.

Respond with ONLY a JSON object, no other text:
{"question": "<the question>", "answer": "<the answer>"}"""

_USER_PROMPT_TEMPLATE = """Passage:
{passage}

Write one question and answer pair based only on this passage."""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into word-count-based chunks with overlap between
    consecutive chunks. Deliberately simple (no sentence-boundary
    detection, no tokenizer dependency) so this has zero extra
    dependencies beyond the stdlib - good enough for generating
    evaluation questions, where exact chunk boundaries matter far less
    than they would for retrieval quality itself.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def load_documents_from_dir(
    path: str | Path, extensions: tuple[str, ...] = (".txt", ".md")
) -> dict[str, str]:
    """Read every file with a matching extension in a directory
    (non-recursive) into a {doc_id: text} mapping, where doc_id is the
    filename stem. Kept deliberately simple - PDF/docx extraction is a
    different concern (see the pdf-reading patterns elsewhere in the
    ecosystem) and out of scope for a zero-dependency core."""
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")

    documents: dict[str, str] = {}
    for file_path in sorted(path.iterdir()):
        if file_path.suffix.lower() in extensions and file_path.is_file():
            documents[file_path.stem] = file_path.read_text(encoding="utf-8")

    return documents


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class SynthesisReport:
    """Result of a synthesis run: the successfully generated test
    cases plus a record of any chunks that failed (bad judge JSON,
    empty chunk, etc.) so a partial failure doesn't silently vanish -
    the user sees exactly what didn't make it into the test set."""

    test_cases: list[TestCase] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def _generate_one(
    chunk_doc_id: str, chunk_text_value: str, dataset_name: str, judge: LLMJudge
) -> TestCase:
    user_prompt = _USER_PROMPT_TEMPLATE.format(passage=chunk_text_value)
    raw = await judge.complete(_SYSTEM_PROMPT, user_prompt)
    data = extract_json_object(raw)

    if "question" not in data or "answer" not in data:
        raise ValueError(
            f"Synthesis response JSON is missing 'question' or 'answer': {data!r}"
        )

    return TestCase(
        question=str(data["question"]),
        ground_truth_answer=str(data["answer"]),
        expected_doc_ids=[chunk_doc_id],
        dataset_name=dataset_name,
    )


async def synthesize_test_set(
    documents: dict[str, str],
    judge: LLMJudge,
    dataset_name: str = "synthesized",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    questions_per_chunk: int = 1,
    max_concurrency: int = 5,
) -> SynthesisReport:
    """Generate a synthetic test set from raw documents.

    Each document is chunked, and each chunk gets its own doc_id in
    the form "<doc_id>::chunk_<n>" - this is what expected_doc_ids
    will contain, so your retriever needs to return matching chunk IDs
    for precision/recall/MRR/nDCG to work against a synthesized set.
    If your retriever returns whole-document IDs instead, only use the
    LLM-judge metrics (faithfulness, answer_relevance) against
    synthesized data, since those don't need expected_doc_ids at all.

    A single chunk's generation failing (bad judge JSON, API error)
    doesn't abort the whole run - it's recorded in the returned
    report's errors list and synthesis continues with the rest.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded_generate(chunk_doc_id: str, text: str) -> tuple[str, TestCase | None, str | None]:
        async with semaphore:
            try:
                tc = await _generate_one(chunk_doc_id, text, dataset_name, judge)
                return chunk_doc_id, tc, None
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see class docstring
                return chunk_doc_id, None, f"{type(exc).__name__}: {exc}"

    tasks = []
    for doc_id, full_text in documents.items():
        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunk_doc_id = f"{doc_id}::chunk_{i}"
            for _ in range(questions_per_chunk):
                tasks.append(_bounded_generate(chunk_doc_id, chunk))

    results = await asyncio.gather(*tasks)

    report = SynthesisReport()
    for chunk_doc_id, test_case, error in results:
        if test_case is not None:
            report.test_cases.append(test_case)
        else:
            report.errors.append(f"{chunk_doc_id}: {error}")

    return report
