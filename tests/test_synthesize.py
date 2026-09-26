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

    async def test_backward_compatibility_query_types_none(self):
        """query_types=None should produce identical behavior to before."""
        documents = {"doc1": " ".join(f"word{i}" for i in range(50))}

        # Test with query_types=None (the default)
        judge_none = _FakeSynthesisJudge()
        report_none = await synthesize_test_set(documents, judge_none, chunk_size=30, chunk_overlap=5)

        # Test with explicit query_types=["standard"]
        judge_standard = _FakeSynthesisJudge()
        report_standard = await synthesize_test_set(
            documents, judge_standard, chunk_size=30, chunk_overlap=5, query_types=["standard"]
        )

        # Should produce identical results
        assert len(report_none.test_cases) == len(report_standard.test_cases)
        assert len(report_none.errors) == len(report_standard.errors)
        for tc_none, tc_standard in zip(report_none.test_cases, report_standard.test_cases):
            assert tc_none.question == tc_standard.question
            assert tc_none.ground_truth_answer == tc_standard.ground_truth_answer
            assert tc_none.expected_doc_ids == tc_standard.expected_doc_ids
            assert tc_none.metadata.get("query_type") == tc_standard.metadata.get("query_type") == "standard"

    async def test_standard_query_type_explicit(self):
        """Explicit 'standard' query type works."""
        judge = _FakeSynthesisJudge()
        documents = {"doc1": " ".join(f"word{i}" for i in range(50))}
        report = await synthesize_test_set(
            documents, judge, chunk_size=30, chunk_overlap=5, query_types=["standard"]
        )

        assert len(report.test_cases) >= 1
        assert all(tc.metadata.get("query_type") == "standard" for tc in report.test_cases)
        for tc in report.test_cases:
            assert len(tc.expected_doc_ids) == 1
            assert tc.expected_doc_ids[0].startswith("doc1::chunk_")

    async def test_adversarial_query_type(self):
        """Adversarial query type generates questions."""
        judge = _FakeSynthesisJudge()
        documents = {"doc1": "This is a test document with some content."}
        report = await synthesize_test_set(
            documents, judge, chunk_size=30, chunk_overlap=5, query_types=["adversarial"]
        )

        assert len(report.test_cases) >= 1
        assert all(tc.metadata.get("query_type") == "adversarial" for tc in report.test_cases)
        # Adversarial questions should still have expected_doc_ids pointing to the chunk
        for tc in report.test_cases:
            assert len(tc.expected_doc_ids) == 1
            assert tc.expected_doc_ids[0].startswith("doc1::chunk_")

    async def test_unanswerable_query_type(self):
        """Unanswerable query type generates questions with empty expected_doc_ids."""
        judge = _FakeSynthesisJudge()
        documents = {"doc1": "This is a test document with some content."}
        report = await synthesize_test_set(
            documents, judge, chunk_size=30, chunk_overlap=5, query_types=["unanswerable"]
        )

        assert len(report.test_cases) >= 1
        assert all(tc.metadata.get("query_type") == "unanswerable" for tc in report.test_cases)
        # Unanswerable questions should have empty expected_doc_ids
        for tc in report.test_cases:
            assert tc.expected_doc_ids == []
            assert tc.ground_truth_answer == "This question cannot be answered from the given context."

    async def test_multi_hop_query_type(self):
        """Multi-hop query type generates questions requiring two chunks."""
        judge = _FakeSynthesisJudge()
        # Create a document with enough content for multiple chunks
        documents = {"doc1": " ".join(f"word{i}" for i in range(200))}  # ~200 words
        report = await synthesize_test_set(
            documents, judge, chunk_size=50, chunk_overlap=5, query_types=["multi_hop"]
        )

        # Should have generated multi-hop questions from adjacent chunk pairs
        # Number of pairs = number of chunks - 1
        chunks = chunk_text(" ".join(f"word{i}" for i in range(200)), chunk_size=50, overlap=5)
        expected_pairs = max(0, len(chunks) - 1)

        assert len(report.test_cases) >= expected_pairs
        assert all(tc.metadata.get("query_type") == "multi_hop" for tc in report.test_cases)
        # Multi-hop questions should have expected_doc_ids with two chunk IDs
        for tc in report.test_cases:
            assert len(tc.expected_doc_ids) == 2
            # Both should be from doc1 and follow chunk naming pattern
            for doc_id in tc.expected_doc_ids:
                assert doc_id.startswith("doc1::chunk_")
            # The two IDs should be different
            assert tc.expected_doc_ids[0] != tc.expected_doc_ids[1]

    async def test_invalid_query_type_raises_error(self):
        """Invalid query type should raise ValueError with clear message."""
        judge = _FakeSynthesisJudge()
        documents = {"doc1": "test content"}

        with pytest.raises(ValueError, match="Unknown query type"):
            await synthesize_test_set(
                documents, judge, query_types=["invalid_type"]
            )

        with pytest.raises(ValueError, match="Unknown query type"):
            await synthesize_test_set(
                documents, judge, query_types=["standard", "invalid_type"]
            )

        with pytest.raises(ValueError, match="Valid options are"):
            await synthesize_test_set(
                documents, judge, query_types=["invalid_type"]
            )

    async def test_multiple_query_types(self):
        """Multiple query types should generate the right mix."""
        judge = _FakeSynthesisJudge()
        documents = {"doc1": " ".join(f"word{i}" for i in range(100))}

        # Test with 2 questions_per_chunk and multiple query types
        report = await synthesize_test_set(
            documents,
            judge,
            chunk_size=30,
            chunk_overlap=5,
            questions_per_chunk=2,
            query_types=["standard", "adversarial"]
        )

        # Should have 2 query types × 2 questions_per_chunk × number of chunks
        chunks = chunk_text(" ".join(f"word{i}" for i in range(100)), chunk_size=30, overlap=5)
        expected_count = len(chunks) * 2 * 2  # chunks × questions_per_chunk × query_types

        assert len(report.test_cases) == expected_count

        # Check distribution
        standard_count = sum(1 for tc in report.test_cases if tc.metadata.get("query_type") == "standard")
        adversarial_count = sum(1 for tc in report.test_cases if tc.metadata.get("query_type") == "adversarial")

        assert standard_count == expected_count // 2
        assert adversarial_count == expected_count // 2

    async def test_query_types_metadata_tagging(self):
        """Each test case should be tagged with its query type in metadata."""
        judge = _FakeSynthesisJudge()
        documents = {"doc1": "test content for testing"}

        for query_type in ["standard", "adversarial", "unanswerable", "multi_hop"]:
            # For multi_hop we need enough content for at least one pair
            if query_type == "multi_hop":
                documents = {"doc1": "word " * 100}  # Enough for multiple chunks

            report = await synthesize_test_set(
                documents, judge, chunk_size=30, chunk_overlap=5, query_types=[query_type]
            )

            assert len(report.test_cases) >= 1
            for tc in report.test_cases:
                assert tc.metadata.get("query_type") == query_type
