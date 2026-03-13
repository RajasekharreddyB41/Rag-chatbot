"""RAG chain with Groq LLM, streaming, conversation memory, and source citations."""

from __future__ import annotations

import logging
from typing import Any, Generator, Iterator

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from app.config import AppConfig, get_config
from app.services.vector_store import VectorStoreManager
from app.utils.exceptions import ConfigurationError, LLMError, RetrievalError
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context documents.

Instructions:
- Answer questions using ONLY the information from the provided context.
- If the context doesn't contain enough information, say so clearly.
- Always cite your sources by referencing the document name and relevant section.
- Be concise, accurate, and professional.
- If asked a follow-up question, use both the conversation history and the context.

Context:
{context}"""


class RAGChain:
    """Orchestrates retrieval-augmented generation with Groq and ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStoreManager | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStoreManager(self.config)
        self._llm: ChatGroq | None = None
        self._chain = None
        self._session_histories: dict[str, ChatMessageHistory] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            if not self.config.groq.api_key:
                raise ConfigurationError("GROQ_API_KEY is not configured.")
            try:
                self._llm = ChatGroq(
                    api_key=self.config.groq.api_key,
                    model=self.config.groq.model,
                    temperature=self.config.groq.temperature,
                    max_tokens=self.config.groq.max_tokens,
                    streaming=self.config.groq.streaming,
                )
                logger.info("ChatGroq initialised with model '%s'", self.config.groq.model)
            except Exception as exc:
                raise LLMError("Failed to initialise ChatGroq", details=str(exc)) from exc
        return self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(
        self, question: str, session_id: str = "default"
    ) -> dict[str, Any]:
        """Non-streaming query. Returns answer + source documents."""
        history = self._get_session_history(session_id)
        retriever = self.vector_store.get_retriever()

        try:
            docs = retriever.invoke(question)
        except Exception as exc:
            raise RetrievalError("Document retrieval failed", details=str(exc)) from exc

        context = self._format_context(docs)
        messages = self._build_messages(history, question, context)

        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as exc:
            raise LLMError("LLM invocation failed", details=str(exc)) from exc

        # Update history
        history.add_user_message(question)
        history.add_ai_message(answer)

        return {
            "answer": answer,
            "source_documents": docs,
            "sources": self._extract_sources(docs),
        }

    def stream(
        self, question: str, session_id: str = "default"
    ) -> Generator[str, None, None]:
        """Streaming query. Yields text chunks."""
        history = self._get_session_history(session_id)
        retriever = self.vector_store.get_retriever()

        try:
            docs = retriever.invoke(question)
        except Exception as exc:
            raise RetrievalError("Document retrieval failed", details=str(exc)) from exc

        context = self._format_context(docs)
        messages = self._build_messages(history, question, context)

        full_answer = ""
        try:
            for chunk in self.llm.stream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield token
        except Exception as exc:
            raise LLMError("LLM streaming failed", details=str(exc)) from exc
        finally:
            # Persist history even if streaming is cut short
            if full_answer:
                history.add_user_message(question)
                history.add_ai_message(full_answer)

    def get_sources_for_query(self, query: str) -> list[Document]:
        """Retrieve source documents without generating an answer."""
        try:
            return self.vector_store.get_retriever().invoke(query)
        except Exception as exc:
            raise RetrievalError("Source retrieval failed", details=str(exc)) from exc

    def clear_session(self, session_id: str = "default") -> None:
        """Clear conversation history for a session."""
        if session_id in self._session_histories:
            self._session_histories[session_id].clear()
            logger.info("Cleared session history for '%s'", session_id)

    def clear_all_sessions(self) -> None:
        """Clear all conversation histories."""
        self._session_histories.clear()
        logger.info("Cleared all session histories.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._session_histories:
            self._session_histories[session_id] = ChatMessageHistory()
        return self._session_histories[session_id]

    def _build_messages(
        self,
        history: ChatMessageHistory,
        question: str,
        context: str,
    ) -> list:
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
        # Add recent history (last 10 turns = 5 exchanges)
        messages.extend(history.messages[-10:])
        messages.append(HumanMessage(content=question))
        return messages

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        if not docs:
            return "No relevant documents found."

        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f", page {page}" if page != "" else ""
            parts.append(f"[{i}] Source: {source}{page_info}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_sources(docs: list[Document]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        sources: list[dict[str, Any]] = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                seen.add(source)
                sources.append(
                    {
                        "source": source,
                        "page": doc.metadata.get("page", ""),
                        "file_type": doc.metadata.get("file_type", ""),
                    }
                )
        return sources
