"""Tests for synthesize.py - chunking, doc loading, and question/answer
generation. Uses FakeJudge throughout, consistent with the rest of the
suite - zero network access, zero API keys needed."""

from __future__ import annotations

import json

import pytest

from rag_score.judges.base import LLMJudge
from rag_score.synthesize import (
    SynthesisReport,
    chunk_text,
    load_documents_from_dir,
    synthesize_test_set,
)


class TestChunkText:
    def test_splits_into_multiple_chunks(self):
        text = " ".join(f"word{i}" for i in range(1000))
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_overlap_is_correct(self):
        text = " ".join(f"word{i}" for i in range(1000))
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        chunk0_words = chunks[0].split()
        chunk1_words = chunks[1].split()
        assert chunk0_words[-20:] == chunk1_words[:20]

    def test_short_text_returns_one_chunk(self):
        chunks = chunk_text("just a few words here", chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "just a few words here"

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_rejects_non_positive_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("some text", chunk_size=0)

    def test_rejects_negative_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("some text", chunk_size=10, overlap=-1)

    def test_rejects_overlap_equal_to_chunk_size(self):
        with pytest.raises(ValueError, match="overlap must be smaller"):
            chunk_text("some text", chunk_size=10, overlap=10)

    def test_rejects_overlap_greater_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap must be smaller"):
            chunk_text("some text", chunk_size=10, overlap=20)


class TestLoadDocumentsFromDir:
    def test_loads_txt_and_md_files(self, tmp_path):
        (tmp_path / "doc1.txt").write_text("content one")
        (tmp_path / "doc2.md").write_text("content two")
        (tmp_path / "ignore.json").write_text("{}")

        docs = load_documents_from_dir(tmp_path)
        assert set(docs.keys()) == {"doc1", "doc2"}
        assert docs["doc1"] == "content one"
        assert docs["doc2"] == "content two"

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        assert load_documents_from_dir(tmp_path) == {}

    def test_non_directory_raises(self, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            load_documents_from_dir(f)

    def test_custom_extensions(self, tmp_path):
        (tmp_path / "doc.rst").write_text("rst content")
        (tmp_path / "doc.txt").write_text("txt content")

        docs = load_documents_from_dir(tmp_path, extensions=(".rst",))
        assert set(docs.keys()) == {"doc"}
        assert docs["doc"] == "rst content"


class _FakeSynthesisJudge(LLMJudge):
    def __init__(self, fail_on_substring: str | None = None):
        self.call_count = 0
        self._fail_on_substring = fail_on_substring

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        if self._fail_on_substring and self._fail_on_substring in user_prompt:
            return "I refuse to produce JSON."
        return json.dumps({"question": f"Q{self.call_count}?", "answer": f"A{self.call_count}"})


class TestSynthesizeTestSet:
    async def test_generates_one_test_case_per_chunk(self):
        judge = _FakeSynthesisJudge()
        documents = {"doc1": " ".join(f"word{i}" for i in range(50))}
        report = await synthesize_test_set(documents, judge, chunk_size=30, chunk_overlap=5)

        assert isinstance(report, SynthesisReport)
        assert len(report.test_cases) >= 1
        assert len(report.errors) == 0

    async def test_expected_doc_ids_reference_chunk_ids(self):
        judge = _FakeSynthesisJudge()
        documents = {"mydoc": " ".join(f"word{i}" for i in range(50))}
        report = await synthesize_test_set(documents, judge, chunk_size=30, chunk_overlap=5)

        for tc in report.test_cases:
            assert tc.expected_doc_ids[0].startswith("mydoc::chunk_")

    async def test_dataset_name_propagates(self):
        judge = _FakeSynthesisJudge()
        documents = {"doc1": "short doc content here"}
        report = await synthesize_test_set(documents, judge, dataset_name="my-custom-set")
        assert all(tc.dataset_name == "my-custom-set" for tc in report.test_cases)

    async def test_failed_chunk_recorded_in_errors_not_raised(self):
        judge = _FakeSynthesisJudge(fail_on_substring="BADMARKER")
        documents = {
            "good_doc": "this is a fine document with normal content",
            "bad_doc": "this document has a BADMARKER in it somewhere",
        }
        report = await synthesize_test_set(documents, judge, chunk_size=100, chunk_overlap=10)

        assert len(report.test_cases) >= 1  # good_doc succeeded
        assert len(report.errors) >= 1  # bad_doc failed
        assert "bad_doc" in report.errors[0]

    async def test_questions_per_chunk_multiplies_output(self):
        judge = _FakeSynthesisJudge()
        documents = {"doc1": "a single short chunk of text"}
        report = await synthesize_test_set(
            documents, judge, chunk_size=100, questions_per_chunk=3
        )
        assert len(report.test_cases) == 3

    async def test_empty_documents_produces_empty_report(self):
        judge = _FakeSynthesisJudge()
        report = await synthesize_test_set({}, judge)
        assert report.test_cases == []
        assert report.errors == []

    async def test_output_is_valid_test_case_shape(self):
        """The whole point of this module - output must be directly
        usable by load_dataset()/run_evaluation() with zero glue."""
        judge = _FakeSynthesisJudge()
        documents = {"doc1": "some reasonably long piece of source content"}
        report = await synthesize_test_set(documents, judge)

        tc = report.test_cases[0]
        assert tc.question
        assert tc.ground_truth_answer
        assert tc.expected_doc_ids
        assert tc.test_case_id  # auto-generated
