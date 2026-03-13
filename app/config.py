"""Centralized configuration using dataclasses and .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent


@dataclass
class GroqConfig:
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    temperature: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
    streaming: bool = True

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in environment or .env file.")


@dataclass
class EmbeddingConfig:
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    normalize_embeddings: bool = True


@dataclass
class ChromaConfig:
    persist_directory: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PERSIST_DIR",
            str(ROOT_DIR / "data" / "chroma_db"),
        )
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "rag_documents")
    )
    distance_metric: str = "cosine"


@dataclass
class ChunkingConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


@dataclass
class RetrieverConfig:
    search_type: str = field(
        default_factory=lambda: os.getenv("RETRIEVER_SEARCH_TYPE", "mmr")
    )
    k: int = int(os.getenv("RETRIEVER_K", "4"))
    fetch_k: int = int(os.getenv("RETRIEVER_FETCH_K", "20"))
    lambda_mult: float = float(os.getenv("RETRIEVER_LAMBDA_MULT", "0.5"))
    score_threshold: float = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.3"))


@dataclass
class AppConfig:
    groq: GroqConfig = field(default_factory=GroqConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = field(
        default_factory=lambda: os.getenv(
            "LOG_DIR", str(ROOT_DIR / "logs")
        )
    )
    upload_dir: str = field(
        default_factory=lambda: os.getenv(
            "UPLOAD_DIR", str(ROOT_DIR / "data" / "uploads")
        )
    )
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    allowed_extensions: list[str] = field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".csv"]
    )
    app_title: str = "RAG Chatbot"
    app_version: str = "1.0.0"

    def validate(self) -> None:
        self.groq.validate()
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma.persist_directory).mkdir(parents=True, exist_ok=True)


# Singleton instance
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
