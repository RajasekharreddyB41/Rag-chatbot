"""ChromaDB vector store manager with HuggingFace embeddings."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import AppConfig, get_config
from app.utils.exceptions import EmbeddingError, VectorStoreError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages ChromaDB collection with HuggingFace embeddings."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._vectorstore: Chroma | None = None
        self._client: chromadb.PersistentClient | None = None

    # ------------------------------------------------------------------
    # Properties / lazy init
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info(
                "Loading embedding model: %s (device=%s)",
                self.config.embedding.model_name,
                self.config.embedding.device,
            )
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embedding.model_name,
                    model_kwargs={"device": self.config.embedding.device},
                    encode_kwargs={
                        "normalize_embeddings": self.config.embedding.normalize_embeddings,
                        "batch_size": self.config.embedding.batch_size,
                    },
                )
            except Exception as exc:
                raise EmbeddingError("Failed to load embedding model", details=str(exc)) from exc
        return self._embeddings

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = self._init_vectorstore()
        return self._vectorstore

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document], batch_size: int = 100) -> int:
        """Add documents in batches. Returns total docs added."""
        if not documents:
            logger.warning("add_documents called with empty list.")
            return 0

        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                self.vectorstore.add_documents(batch)
                total += len(batch)
                logger.debug("Indexed batch %d-%d", i, i + len(batch) - 1)
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to index batch starting at {i}", details=str(exc)
                ) from exc

        logger.info("Successfully indexed %d documents.", total)
        return total

    def similarity_search(
        self, query: str, k: int | None = None
    ) -> list[Document]:
        """Plain similarity search."""
        k = k or self.config.retriever.k
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as exc:
            raise VectorStoreError("Similarity search failed", details=str(exc)) from exc

    def mmr_search(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float | None = None,
    ) -> list[Document]:
        """Maximum Marginal Relevance search for diverse results."""
        k = k or self.config.retriever.k
        fetch_k = fetch_k or self.config.retriever.fetch_k
        lambda_mult = lambda_mult if lambda_mult is not None else self.config.retriever.lambda_mult

        try:
            return self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        except Exception as exc:
            raise VectorStoreError("MMR search failed", details=str(exc)) from exc

    def get_retriever(self, search_type: str | None = None) -> Any:
        """Return a LangChain retriever."""
        search_type = search_type or self.config.retriever.search_type

        search_kwargs: dict[str, Any] = {"k": self.config.retriever.k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = self.config.retriever.fetch_k
            search_kwargs["lambda_mult"] = self.config.retriever.lambda_mult
        elif search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.config.retriever.score_threshold

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def collection_stats(self) -> dict[str, Any]:
        """Return basic stats about the current collection."""
        try:
            count = self.vectorstore._collection.count()
            return {
                "collection_name": self.config.chroma.collection_name,
                "document_count": count,
                "persist_directory": self.config.chroma.persist_directory,
                "embedding_model": self.config.embedding.model_name,
            }
        except Exception as exc:
            raise VectorStoreError("Failed to get collection stats", details=str(exc)) from exc

    def delete_collection(self) -> None:
        """Drop and recreate the collection."""
        try:
            client = chromadb.PersistentClient(
                path=self.config.chroma.persist_directory
            )
            client.delete_collection(self.config.chroma.collection_name)
            self._vectorstore = None
            logger.info("Collection '%s' deleted.", self.config.chroma.collection_name)
        except Exception as exc:
            raise VectorStoreError("Failed to delete collection", details=str(exc)) from exc

    def document_exists(self, content_hash: str) -> bool:
        """Check if a document with the given content hash already exists."""
        try:
            results = self.vectorstore._collection.get(
                where={"content_hash": content_hash}, limit=1
            )
            return len(results["ids"]) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_vectorstore(self) -> Chroma:
        logger.info(
            "Initialising ChromaDB at '%s', collection='%s'",
            self.config.chroma.persist_directory,
            self.config.chroma.collection_name,
        )
        try:
            return Chroma(
                collection_name=self.config.chroma.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.config.chroma.persist_directory,
                collection_metadata={"hnsw:space": self.config.chroma.distance_metric},
            )
        except Exception as exc:
            raise VectorStoreError("Failed to initialise ChromaDB", details=str(exc)) from exc
