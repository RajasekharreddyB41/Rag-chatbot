# 🧠 RAG Chatbot — Production-Ready Retrieval Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat&logo=groq&logoColor=white)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-E8A838?style=flat)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=flat)](LICENSE)

A production-grade RAG (Retrieval Augmented Generation) chatbot that lets you upload documents and ask questions about them in natural language. Built with LangChain, ChromaDB, and Groq's ultra-fast LLM inference — it streams answers in real time, cites its sources, and remembers your conversation history.

---

## ✨ Features

- **Multi-format document ingestion** — PDF, DOCX, TXT, Markdown, and CSV files supported out of the box
- **MMR retrieval** — Maximal Marginal Relevance search returns diverse, non-redundant context chunks
- **SHA-256 deduplication** — files are fingerprinted at ingest time so the same document is never indexed twice
- **Streaming responses** — tokens are pushed to the UI in real time via Groq's inference API
- **Conversation memory** — configurable sliding-window history keeps multi-turn context without ballooning token usage
- **Source citations** — every answer links back to the source file and page number
- **Production logging** — rotating file handlers with configurable log levels; logs persist across container restarts
- **Custom exception hierarchy** — typed exceptions (`DocumentIngestionError`, `LLMError`, `RetrievalError`, …) for clean error handling throughout the stack
- **Docker-ready** — multi-stage Dockerfile with a non-root user, health checks, and named volumes; deployable to Azure App Service in minutes

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐
│  Documents  │───▶│  Ingestion   │───▶│   Chunking   │───▶│    Embedding      │
│ PDF/DOCX/   │    │  Validation  │    │  Recursive   │    │  all-MiniLM-L6-v2 │
│ TXT/MD/CSV  │    │  SHA-256     │    │  Text Split  │    │  (HuggingFace)    │
└─────────────┘    └──────────────┘    └──────────────┘    └────────┬──────────┘
                                                                     │
                                                                     ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐
│  Response   │◀───│   Groq LLM   │◀───│   Retrieval  │◀───│     ChromaDB      │
│  Streaming  │    │ LLaMA 3.3 70B│    │  MMR Search  │    │  Vector Store     │
│  + Sources  │    │  + Memory    │    │  Top-K Docs  │    │  (Persistent)     │
└─────────────┘    └──────────────┘    └──────────────┘    └───────────────────┘
```

---

## 🛠️ Tech Stack

| Component           | Technology                                      |
|---------------------|-------------------------------------------------|
| **LLM**             | Groq API — LLaMA 3.3 70B Versatile              |
| **Embeddings**      | `sentence-transformers/all-MiniLM-L6-v2`        |
| **Vector Store**    | ChromaDB (persistent, local)                    |
| **Orchestration**   | LangChain 0.3 (LCEL pipeline)                   |
| **UI**              | Streamlit with custom light theme               |
| **Document Loaders**| LangChain Community (PDF, DOCX, MD, CSV, TXT)   |
| **Text Splitting**  | `langchain-text-splitters` — Recursive chunking |
| **Configuration**   | Python dataclasses + `python-dotenv`            |
| **Logging**         | Python `logging` with `RotatingFileHandler`     |
| **Testing**         | pytest                                          |
| **Containerization**| Docker (multi-stage) + Docker Compose           |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/)

### 1 — Clone and create a virtual environment

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key (the only required value):

```env
GROQ_API_KEY=gsk_your_key_here
```

### 4 — Run the app

```bash
python run.py
```

The Streamlit UI opens at **http://localhost:8501**.

---

## 📁 Project Structure

```
rag-chatbot/
├── app/
│   ├── config.py               # Dataclass-based config, loaded from .env
│   ├── core/
│   │   └── rag_chain.py        # LCEL RAG pipeline, streaming, memory
│   ├── services/
│   │   ├── document_ingestor.py # Loaders, SHA-256 dedup, chunking
│   │   └── vector_store.py     # ChromaDB wrapper, MMR retrieval
│   ├── ui/
│   │   └── streamlit_app.py    # Full Streamlit UI with sidebar + chat
│   └── utils/
│       ├── exceptions.py       # Typed exception hierarchy
│       └── logger.py           # Rotating file + console logger
├── tests/
│   └── test_pipeline.py        # pytest suite (config, ingestor, chain, store)
├── data/
│   ├── chroma_db/              # Persisted vector store (git-ignored)
│   └── documents/              # Uploaded files (git-ignored)
├── logs/                       # Rotating log files (git-ignored)
├── .streamlit/
│   └── config.toml             # Streamlit theme + server settings
├── .env.example                # Environment variable template
├── Dockerfile                  # Multi-stage production image
├── docker-compose.yml          # Compose with named volumes
└── run.py                      # Entry point
```

---

## ⚙️ Configuration

All settings are read from environment variables (or a `.env` file). Copy `.env.example` to get started.

| Variable                  | Default                                    | Description                                      |
|---------------------------|--------------------------------------------|--------------------------------------------------|
| `GROQ_API_KEY`            | *(required)*                               | Groq API key — get one free at console.groq.com  |
| `GROQ_MODEL`              | `llama-3.3-70b-versatile`                  | Groq model ID                                    |
| `GROQ_TEMPERATURE`        | `0.1`                                      | Sampling temperature (0 = deterministic)         |
| `GROQ_MAX_TOKENS`         | `1024`                                     | Maximum tokens per response                      |
| `EMBEDDING_MODEL`         | `sentence-transformers/all-MiniLM-L6-v2`   | HuggingFace embedding model                      |
| `EMBEDDING_DEVICE`        | `cpu`                                      | `cpu` or `cuda`                                  |
| `EMBEDDING_BATCH_SIZE`    | `32`                                       | Embedding batch size                             |
| `CHROMA_PERSIST_DIR`      | `data/chroma_db`                           | ChromaDB persistence directory                   |
| `CHROMA_COLLECTION`       | `rag_documents`                            | ChromaDB collection name                         |
| `CHUNK_SIZE`              | `1000`                                     | Characters per chunk                             |
| `CHUNK_OVERLAP`           | `200`                                      | Overlap between consecutive chunks              |
| `RETRIEVER_SEARCH_TYPE`   | `mmr`                                      | `mmr` or `similarity`                            |
| `RETRIEVER_K`             | `4`                                        | Number of chunks returned per query              |
| `RETRIEVER_FETCH_K`       | `20`                                       | Candidate pool size for MMR                      |
| `RETRIEVER_LAMBDA_MULT`   | `0.5`                                      | MMR diversity weight (0 = max diversity)         |
| `RETRIEVER_SCORE_THRESHOLD` | `0.3`                                    | Minimum similarity score to include a chunk      |
| `LOG_LEVEL`               | `INFO`                                     | `DEBUG` / `INFO` / `WARNING` / `ERROR`           |
| `LOG_DIR`                 | `logs`                                     | Directory for rotating log files                 |
| `MAX_FILE_SIZE_MB`        | `50`                                       | Maximum upload size per file                     |

---

## 🐳 Deployment

### Docker Compose (recommended)

```bash
# Build and start
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

The app is available at **http://localhost:8501**. Vector store data and logs are persisted in named Docker volumes (`chroma_data`, `logs`).

### Azure App Service

1. Build and push the image to Azure Container Registry:

```bash
az acr build --registry <your-registry> --image rag-chatbot:latest .
```

2. Create an App Service with the container image and set the environment variable:

```bash
az webapp config appsettings set \
  --name <your-app-name> \
  --resource-group <your-rg> \
  --settings GROQ_API_KEY=gsk_your_key_here
```

3. The Dockerfile exposes port `8501`; configure App Service to forward traffic accordingly:

```bash
az webapp config set \
  --name <your-app-name> \
  --resource-group <your-rg> \
  --generic-configurations '{"appCommandLine": "python run.py"}'
```

### Standalone Docker

```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 \
  -e GROQ_API_KEY=gsk_your_key_here \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  rag-chatbot
```

---

## 🧪 Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=app --cov-report=term-missing

# Run a specific test class
pytest tests/test_pipeline.py::TestDocumentIngestor -v
```

The test suite covers:

| Test Class              | Coverage                                              |
|-------------------------|-------------------------------------------------------|
| `TestAppConfig`         | Dataclass defaults, env override, API key validation  |
| `TestDocumentIngestor`  | TXT/MD/CSV ingest, deduplication, error paths         |
| `TestVectorStoreManager`| Document add, empty-list guard (mocked ChromaDB)      |
| `TestRAGChain`          | Invoke, stream, session clear, missing key error      |
| `TestExceptions`        | Exception hierarchy and `details` payload             |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built with [LangChain](https://www.langchain.com/) · [Groq](https://groq.com/) · [ChromaDB](https://www.trychroma.com/) · [Streamlit](https://streamlit.io/)

</div>
