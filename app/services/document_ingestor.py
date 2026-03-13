"""Document loading, validation, SHA-256 deduplication, and chunking."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from app.config import AppConfig, ChunkingConfig, get_config
from app.utils.exceptions import DocumentIngestionError, FileValidationError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIngestor:
    """Handles document loading, validation, deduplication, and chunking."""

    LOADER_MAP: dict[str, type] = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._seen_hashes: set[str] = set()
        self._splitter = self._build_splitter(self.config.chunking)

    @staticmethod
    def _build_splitter(cfg: ChunkingConfig) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=cfg.separators,
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str | Path) -> list[Document]:
        """Load, validate, deduplicate, and chunk a single file."""
        path = Path(file_path)
        self._validate_file(path)

        try:
            raw_docs = self._load_file(path)
        except Exception as exc:
            raise DocumentIngestionError(
                f"Failed to load '{path.name}'", details=str(exc)
            ) from exc

        chunks = self._chunk_documents(raw_docs)
        unique_chunks = self._deduplicate(chunks)

        logger.info(
            "Ingested '%s': %d raw docs → %d chunks (%d unique after dedup)",
            path.name,
            len(raw_docs),
            len(chunks),
            len(unique_chunks),
        )
        return unique_chunks

    def ingest_files(self, file_paths: list[str | Path]) -> list[Document]:
        """Ingest multiple files, skipping any that fail."""
        all_chunks: list[Document] = []
        for fp in file_paths:
            try:
                all_chunks.extend(self.ingest_file(fp))
            except (DocumentIngestionError, FileValidationError) as exc:
                logger.warning("Skipping file due to error: %s", exc)
        return all_chunks

    def reset_seen_hashes(self) -> None:
        """Clear the in-memory deduplication cache."""
        self._seen_hashes.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_file(self, path: Path) -> None:
        if not path.exists():
            raise FileValidationError(f"File not found: {path}")
        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {path}")

        ext = path.suffix.lower()
        if ext not in self.config.allowed_extensions:
            raise FileValidationError(
                f"Unsupported file type '{ext}'",
                details=f"Allowed: {self.config.allowed_extensions}",
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise FileValidationError(
                f"File '{path.name}' is {size_mb:.1f} MB",
                details=f"Max allowed: {self.config.max_file_size_mb} MB",
            )

    def _load_file(self, path: Path) -> list[Document]:
        ext = path.suffix.lower()
        loader_cls = self.LOADER_MAP[ext]

        if ext == ".txt":
            loader = loader_cls(str(path), encoding="utf-8", autodetect_encoding=True)
        else:
            loader = loader_cls(str(path))

        docs = loader.load()

        # Enrich metadata
        for doc in docs:
            doc.metadata.setdefault("source", path.name)
            doc.metadata["file_path"] = str(path)
            doc.metadata["file_type"] = ext.lstrip(".")

        return docs

    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        return chunks

    def _deduplicate(self, chunks: list[Document]) -> list[Document]:
        unique: list[Document] = []
        for chunk in chunks:
            digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
            if digest not in self._seen_hashes:
                self._seen_hashes.add(digest)
                chunk.metadata["content_hash"] = digest
                unique.append(chunk)
        return unique

    @staticmethod
    def compute_file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file (for whole-file dedup)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()
