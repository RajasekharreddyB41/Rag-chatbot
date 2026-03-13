"""Pytest tests for the RAG pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from app.config import AppConfig, ChunkingConfig, ChromaConfig, EmbeddingConfig, GroqConfig, RetrieverConfig
from app.services.document_ingestor import DocumentIngestor
from app.utils.exceptions import (
    DocumentIngestionError,
    FileValidationError,
    RAGChatbotError,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def test_config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        groq=GroqConfig(api_key="test-key", model="llama3-8b-8192"),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        chroma=ChromaConfig(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection",
        ),
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=20),
        retriever=RetrieverConfig(k=2, fetch_k=5),
        log_dir=str(tmp_path / "logs"),
        upload_dir=str(tmp_path / "uploads"),
    )
    cfg.validate()
    return cfg


@pytest.fixture()
def ingestor(test_config: AppConfig) -> DocumentIngestor:
    return DocumentIngestor(config=test_config)


@pytest.fixture()
def sample_txt(tmp_dir: Path) -> Path:
    p = tmp_dir / "sample.txt"
    p.write_text(
        "This is a test document about machine learning.\n\n"
        "Machine learning is a subset of artificial intelligence.\n\n"
        "Deep learning uses neural networks with many layers.\n\n"
        "Natural language processing handles text and speech.\n\n"
        "Computer vision handles images and video.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def sample_md(tmp_dir: Path) -> Path:
    p = tmp_dir / "sample.md"
    p.write_text(
        "# Introduction\n\nThis is a markdown document.\n\n"
        "## Section 1\n\nContent of section one.\n\n"
        "## Section 2\n\nContent of section two.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def sample_csv(tmp_dir: Path) -> Path:
    p = tmp_dir / "sample.csv"
    p.write_text("name,age,city\nAlice,30,New York\nBob,25,London\nCarol,35,Tokyo\n")
    return p


# ──────────────────────────────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────────────────────────────


class TestAppConfig:
    def test_default_config_creation(self) -> None:
        cfg = AppConfig()
        assert cfg.groq is not None
        assert cfg.embedding is not None
        assert cfg.chroma is not None
        assert cfg.chunking is not None
        assert cfg.retriever is not None

    def test_chunking_defaults(self) -> None:
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200

    def test_retriever_defaults(self) -> None:
        cfg = RetrieverConfig()
        assert cfg.k == 4
        assert cfg.search_type == "mmr"

    def test_groq_validation_fails_without_key(self) -> None:
        cfg = GroqConfig(api_key="")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            cfg.validate()

    def test_groq_validation_passes_with_key(self) -> None:
        cfg = GroqConfig(api_key="test-key")
        cfg.validate()  # Should not raise


# ──────────────────────────────────────────────────────────────────────
# DocumentIngestor tests
# ──────────────────────────────────────────────────────────────────────


class TestDocumentIngestor:
    def test_ingest_txt(self, ingestor: DocumentIngestor, sample_txt: Path) -> None:
        chunks = ingestor.ingest_file(sample_txt)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
        assert all(c.page_content.strip() for c in chunks)

    def test_ingest_md(self, ingestor: DocumentIngestor, sample_md: Path) -> None:
        chunks = ingestor.ingest_file(sample_md)
        assert len(chunks) > 0

    def test_ingest_csv(self, ingestor: DocumentIngestor, sample_csv: Path) -> None:
        chunks = ingestor.ingest_file(sample_csv)
        assert len(chunks) > 0

    def test_metadata_populated(self, ingestor: DocumentIngestor, sample_txt: Path) -> None:
        chunks = ingestor.ingest_file(sample_txt)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "content_hash" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_deduplication(self, ingestor: DocumentIngestor, sample_txt: Path) -> None:
        chunks1 = ingestor.ingest_file(sample_txt)
        chunks2 = ingestor.ingest_file(sample_txt)
        # Second ingest should return 0 unique chunks (all duplicates)
        assert len(chunks2) == 0

    def test_reset_seen_hashes(self, ingestor: DocumentIngestor, sample_txt: Path) -> None:
        ingestor.ingest_file(sample_txt)
        ingestor.reset_seen_hashes()
        chunks = ingestor.ingest_file(sample_txt)
        assert len(chunks) > 0

    def test_file_not_found_raises(self, ingestor: DocumentIngestor) -> None:
        with pytest.raises(FileValidationError):
            ingestor.ingest_file("/nonexistent/path/file.txt")

    def test_unsupported_extension_raises(
        self, ingestor: DocumentIngestor, tmp_dir: Path
    ) -> None:
        bad_file = tmp_dir / "file.xyz"
        bad_file.write_text("content")
        with pytest.raises(FileValidationError, match="Unsupported file type"):
            ingestor.ingest_file(bad_file)

    def test_file_too_large_raises(
        self, ingestor: DocumentIngestor, tmp_dir: Path
    ) -> None:
        large_file = tmp_dir / "large.txt"
        # Write a file that looks larger than the limit via mocked stat
        large_file.write_text("some content")
        ingestor.config.max_file_size_mb = 0  # Force 0 MB limit
        with pytest.raises(FileValidationError, match="MB"):
            ingestor.ingest_file(large_file)

    def test_ingest_multiple_files(
        self,
        ingestor: DocumentIngestor,
        sample_txt: Path,
        sample_md: Path,
    ) -> None:
        chunks = ingestor.ingest_files([sample_txt, sample_md])
        assert len(chunks) > 0

    def test_ingest_files_skips_invalid(
        self,
        ingestor: DocumentIngestor,
        sample_txt: Path,
        tmp_dir: Path,
    ) -> None:
        bad_file = tmp_dir / "bad.xyz"
        bad_file.write_text("content")
        # Should not raise; bad file is skipped
        chunks = ingestor.ingest_files([sample_txt, bad_file])
        assert len(chunks) > 0  # From sample_txt

    def test_compute_file_hash(self, sample_txt: Path) -> None:
        h1 = DocumentIngestor.compute_file_hash(sample_txt)
        h2 = DocumentIngestor.compute_file_hash(sample_txt)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


# ──────────────────────────────────────────────────────────────────────
# Custom exceptions tests
# ──────────────────────────────────────────────────────────────────────


class TestExceptions:
    def test_base_exception(self) -> None:
        exc = RAGChatbotError("test error")
        assert str(exc) == "test error"

    def test_exception_with_details(self) -> None:
        exc = RAGChatbotError("test error", details="some detail")
        assert "test error" in str(exc)
        assert "some detail" in str(exc)

    def test_exception_hierarchy(self) -> None:
        from app.utils.exceptions import (
            ConfigurationError,
            DocumentIngestionError,
            EmbeddingError,
            FileValidationError,
            LLMError,
            RetrievalError,
            VectorStoreError,
        )

        for exc_cls in [
            ConfigurationError,
            DocumentIngestionError,
            EmbeddingError,
            FileValidationError,
            LLMError,
            RetrievalError,
            VectorStoreError,
        ]:
            instance = exc_cls("msg")
            assert isinstance(instance, RAGChatbotError)


# ──────────────────────────────────────────────────────────────────────
# VectorStoreManager tests (mocked ChromaDB)
# ──────────────────────────────────────────────────────────────────────


class TestVectorStoreManager:
    @patch("app.services.vector_store.Chroma")
    @patch("app.services.vector_store.HuggingFaceEmbeddings")
    def test_add_documents(
        self,
        mock_embeddings_cls: MagicMock,
        mock_chroma_cls: MagicMock,
        test_config: AppConfig,
    ) -> None:
        from app.services.vector_store import VectorStoreManager

        mock_vs = MagicMock()
        mock_chroma_cls.return_value = mock_vs

        manager = VectorStoreManager(config=test_config)
        docs = [Document(page_content=f"doc {i}", metadata={"source": "test"}) for i in range(5)]
        count = manager.add_documents(docs, batch_size=3)
        assert count == 5
        assert mock_vs.add_documents.call_count == 2  # 5 docs / batch 3 = 2 batches

    @patch("app.services.vector_store.Chroma")
    @patch("app.services.vector_store.HuggingFaceEmbeddings")
    def test_add_empty_documents(
        self,
        mock_embeddings_cls: MagicMock,
        mock_chroma_cls: MagicMock,
        test_config: AppConfig,
    ) -> None:
        from app.services.vector_store import VectorStoreManager

        manager = VectorStoreManager(config=test_config)
        count = manager.add_documents([])
        assert count == 0


# ──────────────────────────────────────────────────────────────────────
# RAGChain tests (mocked LLM + vector store)
# ──────────────────────────────────────────────────────────────────────


class TestRAGChain:
    @patch("app.core.rag_chain.ChatGroq")
    def test_invoke(
        self,
        mock_groq_cls: MagicMock,
        test_config: AppConfig,
    ) -> None:
        from app.core.rag_chain import RAGChain

        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test answer")
        mock_groq_cls.return_value = mock_llm

        # Mock vector store
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Relevant content", metadata={"source": "test.txt"})
        ]
        mock_vs.get_retriever.return_value = mock_retriever

        chain = RAGChain(vector_store=mock_vs, config=test_config)
        result = chain.invoke("What is machine learning?")

        assert "answer" in result
        assert result["answer"] == "Test answer"
        assert "source_documents" in result
        assert "sources" in result

    @patch("app.core.rag_chain.ChatGroq")
    def test_stream(
        self,
        mock_groq_cls: MagicMock,
        test_config: AppConfig,
    ) -> None:
        from app.core.rag_chain import RAGChain

        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(
            [MagicMock(content="Hello"), MagicMock(content=" world")]
        )
        mock_groq_cls.return_value = mock_llm

        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs.get_retriever.return_value = mock_retriever

        chain = RAGChain(vector_store=mock_vs, config=test_config)
        tokens = list(chain.stream("test question"))
        assert "".join(tokens) == "Hello world"

    @patch("app.core.rag_chain.ChatGroq")
    def test_clear_session(
        self,
        mock_groq_cls: MagicMock,
        test_config: AppConfig,
    ) -> None:
        from app.core.rag_chain import RAGChain

        mock_vs = MagicMock()
        chain = RAGChain(vector_store=mock_vs, config=test_config)
        chain._get_session_history("s1").add_user_message("hi")
        chain.clear_session("s1")
        assert len(chain._get_session_history("s1").messages) == 0

    def test_missing_api_key_raises(self, test_config: AppConfig) -> None:
        from app.core.rag_chain import RAGChain
        from app.utils.exceptions import ConfigurationError

        test_config.groq.api_key = ""
        mock_vs = MagicMock()
        chain = RAGChain(vector_store=mock_vs, config=test_config)
        with pytest.raises(ConfigurationError):
            _ = chain.llm
