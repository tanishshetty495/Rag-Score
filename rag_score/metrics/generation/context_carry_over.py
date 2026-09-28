"""
ContextCarryOver - does the generated answer correctly resolve references
(pronouns, ellipsis, "the second one", "that plan") to earlier
turns in the conversation?

This metric only makes sense on datasets whose test cases carry
conversation_history in metadata. On single-turn cases it scores 0.0 by
convention, so users should run it on a dedicated multi-turn dataset
rather than mixing it into a single-turn run.
"""

from __future__ import annotations

from rag_score.core.types import EvalResult, TestCase
from rag_score.judges.base import JudgeVerdict, LLMJudge
from rag_score.metrics.base import Metric

_SYSTEM_PROMPT = """You are a strict, careful evaluator of RAG (Retrieval-Augmented \
Generation) system outputs in multi-turn conversations. Your job is to judge \
CONTEXT CARRY OVER: whether the given answer correctly resolves references \
(pronouns, ellipsis, "the second one", "that plan") to earlier turns \
in the conversation.

Score from 0.0 to 1.0:
- 1.0: the answer fully and correctly resolves all references to earlier turns
- 0.5: the answer partially resolves references or leaves some ambiguity
- 0.0: the answer ignores or misresolves the context (e.g. answers as if \
the conversation started with this turn)

Respond with ONLY a JSON object, no other text:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence explaining the score>"}"""

_USER_PROMPT_TEMPLATE = """Conversation history:
{history}

Current question:
{question}

Answer to evaluate:
{answer}

Judge whether the answer correctly resolves references to the conversation
history above."""


def _format_history(history: list[dict]) -> str:
    """Format a list of {role: str, content: str} turns into a plain string."""
    lines = []
    for turn in history:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        # Capitalize the role for readability (e.g. "user" -> "User")
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


class ContextCarryOver(Metric):
    name = "context_carry_over"
    requires_api_key = True

    def __init__(self, judge: LLMJudge) -> None:
        self.judge = judge

    async def score(self, test_case: TestCase, result: EvalResult) -> float:
        verdict = await self._verdict(test_case, result)
        return verdict.score

    async def score_with_reasoning(
        self, test_case: TestCase, result: EvalResult
    ) -> tuple[float, str | None]:
        verdict = await self._verdict(test_case, result)
        return verdict.score, verdict.reasoning

    async def _verdict(self, test_case: TestCase, result: EvalResult) -> JudgeVerdict:
        # Extract conversation history from metadata
        history_raw = test_case.metadata.get("conversation_history")
        if not isinstance(history_raw, list) or not history_raw:
            return JudgeVerdict(
                score=0.0,
                reasoning="No usable conversation history in metadata.",
            )

        # Validate each history entry
        formatted_history_lines = []
        for i, turn in enumerate(history_raw):
            if not isinstance(turn, dict):
                return JudgeVerdict(
                    score=0.0,
                    reasoning=f"History entry {i} is not a dictionary.",
                )
            role = turn.get("role")
            content = turn.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                return JudgeVerdict(
                    score=0.0,
                    reasoning=f"History entry {i} missing or invalid 'role' or 'content'.",
                )
            formatted_history_lines.append(f"{role.capitalize()}: {content}")

        if not result.generated_answer:
            return JudgeVerdict(
                score=0.0, reasoning="No generated answer to evaluate."
            )

        history_text = "\n".join(formatted_history_lines)
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            history=history_text,
            question=test_case.question,
            answer=result.generated_answer,
        )
        return await self.judge.judge(_SYSTEM_PROMPT, user_prompt)