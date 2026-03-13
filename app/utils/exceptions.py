"""Custom exceptions for the RAG chatbot."""

from __future__ import annotations


class RAGChatbotError(Exception):
    """Base exception for all RAG chatbot errors."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(RAGChatbotError):
    """Raised when configuration is invalid or missing."""


class DocumentIngestionError(RAGChatbotError):
    """Raised when document loading or parsing fails."""


class EmbeddingError(RAGChatbotError):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGChatbotError):
    """Raised when ChromaDB operations fail."""


class RetrievalError(RAGChatbotError):
    """Raised when document retrieval fails."""


class LLMError(RAGChatbotError):
    """Raised when the LLM call fails."""


class FileValidationError(RAGChatbotError):
    """Raised when file validation fails (size, type, etc.)."""
