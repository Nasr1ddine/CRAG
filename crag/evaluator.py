"""Faithfulness judge — scores how well the answer is grounded in context."""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from crag.config import settings

logger = logging.getLogger(__name__)

_MAX_EVAL_CONTEXT_CHARS = 10_000
_MAX_EVAL_ANSWER_CHARS = 4_000

_JUDGE_SYSTEM = (
    "You are a faithfulness evaluator. Given a context and an answer, determine "
    "what fraction of the answer's claims are fully supported by the context.\n\n"
    "Instructions:\n"
    "1. Decompose the answer into individual factual claims.\n"
    "2. For each claim, check if the context provides supporting evidence.\n"
    "3. Return a score between 0.0 and 1.0 (fraction of supported claims).\n"
    "4. List every unsupported or partially supported claim in the issues list.\n"
    "5. If the answer explicitly states the context is insufficient, that is faithful — score 1.0."
)

_JUDGE_HUMAN = (
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Evaluate faithfulness."
)


class FaithfulnessResult(BaseModel):
    """Structured output from the faithfulness evaluation."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of answer claims supported by context (0.0–1.0).",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific unsupported or partially supported claims.",
    )


class FaithfulnessJudge:
    """Evaluates whether a generated answer is faithful to the provided context."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.grader_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        ).with_structured_output(FaithfulnessResult)

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _JUDGE_SYSTEM),
                ("human", _JUDGE_HUMAN),
            ]
        )

    def evaluate(self, context: str, answer: str) -> FaithfulnessResult:
        """Score the faithfulness of an answer against the context.

        Args:
            context: The verified context that was fed to the generator.
            answer: The generated answer to evaluate.

        Returns:
            ``FaithfulnessResult`` with a score and list of issues.
        """
        truncated_ctx = context[:_MAX_EVAL_CONTEXT_CHARS]
        truncated_ans = answer[:_MAX_EVAL_ANSWER_CHARS]
        try:
            result: FaithfulnessResult = (self._prompt | self._llm).invoke(
                {"context": truncated_ctx, "answer": truncated_ans}
            )
            logger.info("Faithfulness score: %.2f (%d issues)", result.score, len(result.issues))
            return result  # type: ignore[return-value]
        except Exception:
            logger.exception("Faithfulness evaluation failed — returning pessimistic score")
            return FaithfulnessResult(
                score=0.0, issues=["Evaluation failed due to an internal error."]
            )
