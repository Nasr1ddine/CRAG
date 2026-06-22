"""Answer generator — produces the final response grounded in verified context."""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from crag.config import settings

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 12_000

_GENERATOR_SYSTEM = (
    "You are a knowledgeable assistant. Answer the user's question based ONLY "
    "on the provided context. Follow these rules strictly:\n"
    "- If the context does not contain enough information, say so explicitly.\n"
    "- Do NOT hallucinate or invent facts not present in the context.\n"
    "- Cite the source when the context includes source metadata.\n"
    "- Be concise but thorough."
)

_GENERATOR_HUMAN = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class AnswerGenerator:
    """Generates a final answer grounded exclusively in the provided context."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.generator_model,
            temperature=0.2,
            openai_api_key=settings.openai_api_key,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _GENERATOR_SYSTEM),
                ("human", _GENERATOR_HUMAN),
            ]
        )
        self._chain = self._prompt | self._llm

    def generate(self, query: str, context: str) -> str:
        """Generate an answer from context.

        Args:
            query: The user question.
            context: Verified, refined context string.

        Returns:
            The generated answer text.
        """
        truncated_ctx = context[:_MAX_CONTEXT_CHARS]
        result = self._chain.invoke(
            {"question": query, "context": truncated_ctx}
        )
        answer = result.content.strip()
        logger.info("Generated answer (%d chars) for query: %s", len(answer), query)
        return answer
