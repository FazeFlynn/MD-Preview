
# RAG System — Comprehensive Project Explanation

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Architecture & Tech Stack](#2-architecture--tech-stack)
- [3. Project Structure](#3-project-structure)
- [4. Libraries & Dependencies](#4-libraries--dependencies)
- [5. How We Built the Project](#5-how-we-built-the-project)
- [6. System Workflow (End-to-End)](#6-system-workflow-end-to-end)
- [7. Core Components Deep Dive](#7-core-components-deep-dive)
  - [7.1 Configuration](#71-configuration)
  - [7.2 Document Processing & File Loading](#72-document-processing--file-loading)
  - [7.3 Chunking Strategy](#73-chunking-strategy)
  - [7.4 Embeddings](#74-embeddings)
  - [7.5 Vector Store (ChromaDB)](#75-vector-store-chromadb)
  - [7.6 Retrieval & Reranking](#76-retrieval--reranking)
  - [7.7 Query Router](#77-query-router)
  - [7.8 LLM Integration (Ollama)](#78-llm-integration-ollama)
  - [7.9 Web Search Pipeline](#79-web-search-pipeline)
  - [7.10 Conversation Memory](#710-conversation-memory)
  - [7.11 RAG Pipeline Orchestrator](#711-rag-pipeline-orchestrator)
- [8. API Endpoints](#8-api-endpoints)
- [9. Streamlit Frontend](#9-streamlit-frontend)
- [10. Data Models & Schemas](#10-data-models--schemas)
- [11. Testing](#11-testing)
- [12. Evaluation System](#12-evaluation-system)
- [13. How to Start the Project Locally](#13-how-to-start-the-project-locally)
  - [13.1 Local Development (Without Docker)](#131-local-development-without-docker)
  - [13.2 Using Docker (Recommended)](#132-using-docker-recommended)
- [14. How to Deploy for Everyone to Use](#14-how-to-deploy-for-everyone-to-use)
  - [14.1 Deploy on a Cloud VM](#141-deploy-on-a-cloud-vm)
  - [14.2 Deploy on AWS EC2](#142-deploy-on-aws-ec2)
  - [14.3 Deploy on Google Cloud (GCE)](#143-deploy-on-google-cloud-gce)
  - [14.4 Deploy on Azure VM](#144-deploy-on-azure-vm)
  - [14.5 Production Checklist](#145-production-checklist)
- [15. Environment Variables Reference](#15-environment-variables-reference)
- [16. Performance & Scalability Notes](#16-performance--scalability-notes)

---

## 1. Project Overview

This is a **full-featured Retrieval-Augmented Generation (RAG) system** — an AI-powered application that allows users to:

- **Upload documents** (PDF, TXT, CSV, XLSX, DOCX, HTML, JSON, Markdown) and ask questions about their content.
- **Perform intelligent web searches** using SearXNG meta-search engine with automatic content scraping.
- **Maintain conversational context** across multiple turns of dialogue.
- **Auto-detect query intent** — the system automatically routes queries to the right pipeline: document Q&A, web search, hybrid (both), or general chat.

The system is designed for **privacy-first** operation — all LLM inference runs locally via Ollama, documents stay on your machine, and web search uses the privacy-focused SearXNG engine.

---

## 2. Architecture & Tech Stack

```
┌───────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (:8501)                       │
│                    (Chat Interface + File Upload)                  │
└──────────────────────────┬────────────────────────────────────────┘
                           │ HTTP (REST API)
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND (:8011)                      │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  /chat/   │  │/documents│  │ /health  │  │ Query Router     │  │
│  │ endpoint  │  │ endpoint │  │ endpoint │  │ (Auto-classify)  │  │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └────────┬─────────┘  │
│       │              │                               │            │
│       ▼              ▼                               ▼            │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    RAG PIPELINE ORCHESTRATOR                │  │
│  │                                                             │  │
│  │  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐   │  │
│  │  │  Document │  │   Web    │  │  Hybrid  │  │  General  │   │  │
│  │  │    Q&A    │  │  Search  │  │  (Both)  │  │   Chat    │   │  │
│  │  └─────┬─────┘  └────┬─────┘  └────┬─────┘  └───────────┘   │  │
│  └────────┼──────────────┼─────────────┼───────────────────────┘  │
│           │              │             │                          │
└───────────┼──────────────┼─────────────┼──────────────────────────┘
            │              │             │
     ┌───────▼───────┐ ┌───▼─────────┐   │
     │   ChromaDB    │ │  SearXNG    │   │
     │ Vector Store  │ │ (:8080/8888)│   │
     │ (Persistent)  │ │ Meta-Search │   │
     └───────────────┘ └─────────────┘   │
             │                           │
     ┌───────▼───────────────────────────▼──┐
    │           OLLAMA LLM (:11434)         │
    │    (tinyllama / llama3.1 / phi3)      │
    └───────────────────────────────────────┘
```

| Layer         | Technology                                    |
|---------------|-----------------------------------------------|
| Frontend      | Streamlit                                     |
| Backend API   | FastAPI + Uvicorn                             |
| LLM Engine    | Ollama (local inference)                      |
| Embeddings    | sentence-transformers (`all-MiniLM-L6-v2`)   |
| Vector Store  | ChromaDB (persistent, cosine similarity)      |
| Reranker      | cross-encoder (`ms-marco-MiniLM-L-6-v2`)     |
| Web Search    | SearXNG (meta-search) + Trafilatura (scraper) |
| Containerization | Docker + Docker Compose                    |

---

## 3. Project Structure

```
rag-open/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker build for the API
├── docker-compose.yml               # Multi-service orchestration
├── README.md                        # Quick-start readme
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py              # Chat & streaming endpoints
│   │       ├── documents.py         # Upload, list, delete documents
│   │       └── health.py            # Health check endpoint
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Settings & environment variables
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic request/response models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py         # Main RAG orchestrator
│   │   ├── chunker.py              # Document chunking (recursive split)
│   │   ├── conversation.py         # In-memory conversation history
│   │   ├── embeddings.py           # Sentence-transformer embeddings
│   │   ├── file_loader.py          # Multi-format file parser
│   │   ├── llm.py                  # Ollama LLM client
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   ├── retriever.py            # Vector search + reranking
│   │   ├── router.py               # Query intent classification
│   │   ├── vector_store.py         # ChromaDB operations
│   │   └── web_search.py           # SearXNG + web scraping
│   │
│   └── ui/
│       └── streamlit_app.py        # Streamlit chat interface
│
├── data/
│   ├── chroma_db/                   # Persistent vector database
│   │   └── chroma.sqlite3
│   └── uploads/                     # Uploaded document storage
│
├── eval/
│   └── evaluate.py                  # RAG evaluation benchmarks
│
├── searxng/
│   ├── settings.yml                 # SearXNG configuration
│   └── limiter.toml                 # Rate limiting config
│
└── tests/
    ├── __init__.py
    ├── test_api.py                  # API endpoint tests
    ├── test_chunker.py              # Chunking logic tests
    ├── test_file_loader.py          # File loading tests
    └── test_router.py               # Query routing tests
```

---

## 4. Libraries & Dependencies

### Core Framework
| Library | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.115.0 | Web API framework |
| `uvicorn[standard]` | 0.30.6 | ASGI server |
| `python-multipart` | 0.0.9 | File upload handling |
| `python-dotenv` | 1.0.1 | `.env` file loading |
| `pydantic` | 2.9.2 | Data validation |
| `pydantic-settings` | 2.5.2 | Settings management |
| `sse-starlette` | 2.1.3 | Server-Sent Events (streaming) |
| `loguru` | 0.7.2 | Structured logging |
| `tenacity` | 8.2.3 | Retry logic with backoff |
| `aiofiles` | 24.1.0 | Async file I/O |

### Document Parsing
| Library | Version | Purpose |
|---------|---------|---------|
| `pymupdf` | 1.24.10 | PDF text extraction |
| `pdfplumber` | 0.11.4 | PDF parsing (fallback) |
| `python-docx` | 1.1.2 | Word document (.docx) |
| `openpyxl` | 3.1.5 | Excel spreadsheets |
| `pandas` | 2.2.3 | CSV/Excel data processing |
| `beautifulsoup4` | 4.12.3 | HTML parsing |
| `unstructured` | 0.15.13 | General document parsing |

### Machine Learning & Embeddings
| Library | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | 3.1.1 | Text embedding models |
| `torch` | 2.4.1 | Deep learning framework |
| `transformers` | 4.44.2 | HuggingFace model hub |

### Vector Store & Search
| Library | Version | Purpose |
|---------|---------|---------|
| `chromadb` | 0.5.7 | Vector database |
| `faiss-cpu` | 1.8.0.post1 | Vector similarity indexing |

### LLM & Web
| Library | Version | Purpose |
|---------|---------|---------|
| `ollama` | 0.3.3 | Ollama Python client |
| `httpx` | 0.27.2 | Async HTTP client |
| `duckduckgo-search` | 6.2.13 | DuckDuckGo fallback search |
| `trafilatura` | 1.12.2 | Web page content extraction |
| `requests` | 2.32.3 | HTTP requests |

### Frontend
| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.38.0 | Web UI framework |

### Testing & Evaluation
| Library | Version | Purpose |
|---------|---------|---------|
| `pytest` | 8.3.3 | Test framework |
| `pytest-asyncio` | 0.24.0 | Async test support |
| `ragas` | 0.1.21 | RAG evaluation metrics |
| `datasets` | 3.0.1 | HuggingFace datasets |

---

## 5. How We Built the Project

The project was built following a **modular, service-oriented architecture**. Here is how each piece was developed:

### Step 1: Core Configuration
We started with `app/core/config.py` — a centralized configuration system using Pydantic Settings. All configurable values (model names, ports, chunk sizes, etc.) are defined with sensible defaults and can be overridden via environment variables or a `.env` file.

### Step 2: Document Processing Pipeline
- **File Loader** (`file_loader.py`): Built parsers for each supported file type (PDF, DOCX, CSV, XLSX, TXT, HTML, JSON, MD). Each parser extracts text content and attaches metadata (source filename, page number, etc.).
- **Chunker** (`chunker.py`): Implemented recursive text splitting that tries paragraph breaks first, then sentences, then words. Each chunk preserves overlap with its neighbors for context continuity.

### Step 3: Embedding & Vector Store
- **Embeddings** (`embeddings.py`): Integrated HuggingFace sentence-transformers to convert text chunks into dense vectors (384 dimensions with `all-MiniLM-L6-v2`).
- **Vector Store** (`vector_store.py`): Set up ChromaDB with persistent storage, batch embedding (64 docs at a time), and cosine similarity search.

### Step 4: Retrieval & Reranking
- **Retriever** (`retriever.py`): Performs vector similarity search against ChromaDB, fetches 3x candidates when reranking is enabled.
- **Reranker** (`reranker.py`): Uses a cross-encoder model to re-score query-document pairs and select the most relevant chunks.

### Step 5: LLM Integration
- **LLM Service** (`llm.py`): Built an async client for Ollama that supports both regular and streaming responses. Includes connection checking and 120-second timeout.

### Step 6: Web Search
- **Web Search** (`web_search.py`): Integrated SearXNG as the meta-search engine, added Trafilatura for scraping full page content, and built query decomposition logic (using the LLM to break complex queries into sub-queries).

### Step 7: Query Router
- **Router** (`router.py`): Built keyword-based classification that detects whether a query should go to document Q&A, web search, hybrid, or general chat. It considers document availability, casual patterns, and web-specific keywords.

### Step 8: RAG Pipeline Orchestrator
- **Pipeline** (`rag_pipeline.py`): The central orchestrator that ties everything together — routes queries, builds context, manages conversation history, and generates answers with appropriate system prompts.

### Step 9: API Layer
- Built REST API endpoints with FastAPI for chat, document management, and health checks.
- Added streaming support via Server-Sent Events for real-time token delivery.

### Step 10: Frontend
- Built a Streamlit chat interface with file upload, document management, query mode selection, and source citations.

### Step 11: Docker Deployment
- Created Dockerfile for the API service and docker-compose.yml to orchestrate all 4 services (API, Ollama, SearXNG, Streamlit).

### Step 12: Testing & Evaluation
- Wrote unit tests for API endpoints, chunker, file loader, and router.
- Built an evaluation system to benchmark RAG performance using keyword coverage and answer quality metrics.

---

## 6. System Workflow (End-to-End)

### Document Upload Flow
```
User uploads file via UI or API
         │
         ▼
┌─────────────────────┐
│  File Validation     │  Check extension & size (max 50MB)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Save to Disk        │  Store in ./data/uploads/
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Parse Content       │  Extract text using appropriate parser
│  (file_loader.py)    │  (PyMuPDF for PDF, python-docx for DOCX, etc.)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Chunk Text          │  Recursive splitting (512 chars, 100 overlap)
│  (chunker.py)        │  Attach metadata (source, page, chunk_index)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Generate Embeddings │  sentence-transformers → 384-dim vectors
│  (embeddings.py)     │  Batch processing (64 docs at a time)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Store in ChromaDB   │  Persistent vector storage with metadata
│  (vector_store.py)   │  Cosine similarity indexing
└─────────────────────┘
```

### Query Processing Flow
```
User sends a query
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: Query Classification (router.py)            │
│                                                      │
│  Analyze query text + check document availability    │
│  → Classify as: DOCUMENT_QA / WEB_SEARCH /           │
│                  HYBRID / GENERAL                     │
└────────┬────────────────────────────────────────────┘
         │
         ├─── DOCUMENT_QA ─────────────────────────────┐
         │                                              │
         │    ┌─────────────────────────┐               │
         │    │ Embed query             │               │
         │    │ Search ChromaDB         │               │
         │    │ Rerank results          │               │
         │    │ Build context from top-K│               │
         │    │ Generate answer via LLM │               │
         │    │ Return answer + sources │               │
         │    └─────────────────────────┘               │
         │                                              │
         ├─── WEB_SEARCH ──────────────────────────────┐
         │                                              │
         │    ┌─────────────────────────┐               │
         │    │ Search via SearXNG      │               │
         │    │ Scrape web pages        │               │
         │    │ (Decompose query if     │               │
         │    │  few results found)     │               │
         │    │ Build context from web  │               │
         │    │ Generate answer via LLM │               │
         │    │ Return answer + URLs    │               │
         │    └─────────────────────────┘               │
         │                                              │
         ├─── HYBRID ──────────────────────────────────┐
         │                                              │
         │    ┌─────────────────────────┐               │
         │    │ Run BOTH document Q&A   │               │
         │    │ AND web search          │               │
         │    │ Synthesize results      │               │
         │    │ Merge all sources       │               │
         │    └─────────────────────────┘               │
         │                                              │
         └─── GENERAL ─────────────────────────────────┐
              ┌─────────────────────────┐               │
              │ Direct LLM generation   │               │
              │ No retrieval needed     │               │
              └─────────────────────────┘               │
                                                        │
         ┌──────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Conversation Memory                         │
│  Save query + response to conversation history       │
│  (Last 10 turns maintained per conversation)         │
└────────┬────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Return Response                             │
│  { answer, query_type, sources, web_sources,         │
│    conversation_id }                                 │
└─────────────────────────────────────────────────────┘
```

---

## 7. Core Components Deep Dive

### 7.1 Configuration

**File**: `app/core/config.py`

Uses Pydantic Settings to manage all configuration. Values can be set via environment variables or a `.env` file. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `APP_HOST` | `0.0.0.0` | Server bind address |
| `APP_PORT` | `8011` | Server port |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Device for embeddings (`cpu` or `cuda`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `tinyllama` | Default LLM model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector DB storage path |
| `COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Number of documents to retrieve |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `USE_RERANKER` | `true` | Enable/disable reranking |
| `SEARXNG_BASE_URL` | `http://localhost:8888` | SearXNG URL |
| `WEB_SEARCH_MAX_RESULTS` | `10` | Max search results |
| `UPLOAD_DIR` | `./data/uploads` | File upload directory |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |

### 7.2 Document Processing & File Loading

**File**: `app/services/file_loader.py`

Supports the following file formats with dedicated parsers:

| Format | Parser | Details |
|--------|--------|---------|
| `.pdf` | PyMuPDF | Extracts text page-by-page, includes page metadata |
| `.txt` | Built-in | Plain text reading |
| `.md` | Built-in | Markdown as plain text |
| `.json` | Built-in | JSON file as string |
| `.csv` | Pandas | Groups rows into text chunks |
| `.xlsx`, `.xls` | openpyxl + Pandas | Processes each sheet separately |
| `.docx` | python-docx | Extracts paragraphs as text |
| `.html`, `.htm` | BeautifulSoup | Strips scripts, nav, footer — extracts clean text |

Each parsed document produces a list of dictionaries with:
- `content`: The extracted text
- `metadata`: Source filename, page number (if applicable), and other context

### 7.3 Chunking Strategy

**File**: `app/services/chunker.py`

Uses a **recursive text splitting** algorithm:

1. **Try separators in priority order**:
   - `"\n\n"` — Paragraph breaks (best quality splits)
   - `"\n"` — Line breaks
   - `". "` — Sentence boundaries
   - `" "` — Word boundaries
   - `""` — Character-level (last resort)

2. **Parameters**:
   - `chunk_size`: 512 characters (default)
   - `chunk_overlap`: 100 characters (preserves context across boundaries)

3. **Overlap**: The last 100 characters of each chunk are prepended to the next chunk, ensuring no context is lost at chunk boundaries.

4. **Metadata preservation**: Each chunk retains its source filename, page number, chunk index, and total chunk count.

### 7.4 Embeddings

**File**: `app/services/embeddings.py`

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)
- **Device**: Configurable — CPU (default) or CUDA GPU
- **Normalization**: All embeddings are L2-normalized for cosine similarity
- **Batch processing**: Documents are embedded in batches of 64 for memory efficiency
- **Custom models**: Any HuggingFace sentence-transformer model can be used via the `EMBEDDING_MODEL` environment variable

### 7.5 Vector Store (ChromaDB)

**File**: `app/services/vector_store.py`

- **Backend**: ChromaDB with persistent SQLite storage
- **Collection**: Single collection (`rag_documents` by default)
- **Distance metric**: Cosine similarity
- **Batch operations**: Embeds and inserts 64 documents at a time
- **Operations supported**:
  - `add` — Insert documents with embeddings and metadata
  - `search` — Query by vector similarity, return top-K
  - `delete` — Remove all chunks for a given source file
  - `list_sources` — List all indexed document sources
- **Persistence**: Data stored at `./data/chroma_db/chroma.sqlite3`

### 7.6 Retrieval & Reranking

**Files**: `app/services/retriever.py`, `app/services/reranker.py`

**Retrieval Process**:
1. Generate embedding for the user's query
2. Search ChromaDB for similar vectors (cosine similarity)
3. If reranking is enabled, fetch **3× top_k** candidates
4. Apply cross-encoder reranking to score each query-document pair
5. Return the final **top_k** (default: 5) most relevant chunks

**Reranker**:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores every (query, document) pair and re-sorts by relevance
- Can be disabled via `USE_RERANKER=false` for faster (but less accurate) retrieval

### 7.7 Query Router

**File**: `app/services/router.py`

Classifies each user query into one of four types:

```
Query arrives
     │
     ├─ Matches casual patterns?           → GENERAL
     │  (hello, hi, thanks, bye, help...)
     │
     ├─ Contains document keywords          → DOCUMENT_QA
     │  AND documents are indexed?
     │  (document, file, uploaded, pdf...)
     │
     ├─ No documents indexed?
     │  ├─ Casual query?                    → GENERAL
     │  └─ Otherwise                        → WEB_SEARCH
     │
     ├─ Contains web/time keywords?         → WEB_SEARCH
     │  (latest, today, news, 2024,
     │   weather, stock, current...)
     │
     └─ Default (with documents)            → DOCUMENT_QA
```

### 7.8 LLM Integration (Ollama)

**File**: `app/services/llm.py`

- **Engine**: Ollama — runs LLMs locally on your machine
- **Supported models**: `tinyllama`, `llama3.1:8b`, `phi3`, or any model available in Ollama
- **Features**:
  - Async HTTP calls to Ollama's `/api/chat` endpoint
  - System prompt injection for context-aware responses
  - Streaming support (token-by-token delivery via SSE)
  - Connection health checking
  - 120-second generation timeout

**System Prompts by Query Type**:
- **DOCUMENT_QA**: "Answer ONLY based on the provided context. If the context doesn't contain the answer, say so."
- **WEB_SEARCH**: "Answer ONLY using web search results. Trust web results over your training data."
- **GENERAL**: "Give direct, concise answers."

### 7.9 Web Search Pipeline

**File**: `app/services/web_search.py`

Three-stage web search process:

1. **Search via SearXNG** — Queries the meta-search engine which aggregates results from Google, Bing, DuckDuckGo, and Wikipedia. Returns URLs, titles, and snippets.

2. **Content Scraping via Trafilatura** — For each search result URL, extracts the main article content (ignoring navigation, footers, ads). 15-second timeout per URL with retry logic.

3. **Query Decomposition (Adaptive)** — If the initial search returns fewer than 3 results, the LLM breaks the original query into 2-3 sub-queries and searches each independently. Results are deduplicated by URL.

### 7.10 Conversation Memory

**File**: `app/services/conversation.py`

- **Storage**: In-memory Python dictionary (per conversation_id)
- **Format**: List of `{role: "user"/"assistant", content: "..."}` messages
- **Max turns**: 20 (oldest turns are truncated when exceeded)
- **Conversation ID**: UUID generated automatically or provided by the user
- **History format**: Formatted as `"Q: ...\nA: ..."` strings and injected into the LLM prompt

> **Note**: Conversation memory resets on server restart. For production, use Redis or a database.

### 7.11 RAG Pipeline Orchestrator

**File**: `app/services/rag_pipeline.py`

The central orchestrator that ties everything together:

1. **Receives** user query + optional conversation_id + optional query_type
2. **Routes** the query (auto-detect or user-specified)
3. **Retrieves** context (documents, web, or both)
4. **Builds** the LLM prompt with system instructions + context + conversation history
5. **Generates** the answer via Ollama
6. **Saves** the exchange to conversation memory
7. **Returns** structured response with answer, sources, and metadata

---

## 8. API Endpoints

### Health Check
```
GET /health
```
Returns system status:
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "embedding_model_loaded": true,
  "documents_indexed": 42
}
```

### Chat
```
POST /chat/
```
**Request body**:
```json
{
  "query": "What is machine learning?",
  "query_type": "auto",           // Optional: "document_qa", "web_search", "hybrid", "general"
  "conversation_id": "uuid-here"  // Optional: for continuing a conversation
}
```
**Response**:
```json
{
  "answer": "Machine learning is...",
  "query_type": "web_search",
  "sources": [],
  "web_sources": [
    {
      "url": "https://example.com/ml",
      "title": "What is ML?",
      "snippet": "Machine learning is a branch..."
    }
  ],
  "conversation_id": "uuid-here"
}
```

### Chat (Streaming)
```
POST /chat/stream
```
Same request body as `/chat/`. Returns Server-Sent Events (SSE) with tokens streamed in real-time.

### Upload Document
```
POST /documents/upload
```
Multipart form data with a file. Supported extensions: `.pdf`, `.txt`, `.csv`, `.xlsx`, `.xls`, `.docx`, `.html`, `.htm`, `.json`, `.md`

### List Documents
```
GET /documents/
```
Returns list of all indexed documents with chunk counts.

### Delete Document
```
DELETE /documents/{filename}
```
Removes the document file and all its chunks from the vector store.

---

## 9. Streamlit Frontend

**File**: `app/ui/streamlit_app.py`

The web interface provides:

**Sidebar**:
- **File Upload Widget** — Drag & drop or browse for supported file types
- **Document List** — View all indexed documents with delete buttons
- **Query Mode Selector** — Choose between Auto-detect, Document Q&A, Web Search, or Hybrid
- **Health Indicators** — Shows Ollama connection status and number of indexed chunks

**Main Chat Area**:
- **Conversation History** — Full chat thread with user and assistant messages
- **Chat Input** — Text input with streaming response display
- **Source Citations** — Expandable sections showing:
  - 📚 Document sources (filename, page, relevance score)
  - 🌐 Web sources (URL, title, snippet)
- **Query Type Badge** — Icons indicating which pipeline was used (📄/🌐/🔀)
- **Clear Chat Button** — Reset the conversation

**Connection**: Posts to `http://localhost:8011` (or `API_URL` environment variable) with 180-second timeout.

---

## 10. Data Models & Schemas

**File**: `app/models/schemas.py`

```python
# Request
ChatRequest:
    query: str          # 1-5000 characters
    query_type: str     # Optional: "document_qa" | "web_search" | "hybrid" | "general"
    conversation_id: str  # Optional UUID

# Response
ChatResponse:
    answer: str
    query_type: QueryType
    sources: List[SourceDocument]
    web_sources: List[WebSource]
    conversation_id: Optional[str]

# Source from documents
SourceDocument:
    content: str        # First 500 chars of the chunk
    source: str         # Filename
    page: Optional[int]
    chunk_index: Optional[int]
    score: Optional[float]

# Source from web
WebSource:
    url: str
    title: str
    snippet: str        # First 300 chars
```

---

## 11. Testing

**Directory**: `tests/`

Run all tests:
```bash
pytest tests/ -v
```

### Test Coverage

**`test_api.py`** — API endpoint tests:
- Health endpoint returns valid status
- Empty document list returns correctly
- Unsupported file types are rejected
- Empty queries trigger validation errors
- Deleting nonexistent documents returns 404

**`test_chunker.py`** — Chunking logic:
- Short documents produce a single chunk
- Long documents produce multiple chunks with correct metadata
- Metadata is preserved through chunking
- Multiple documents are chunked correctly
- Empty input returns empty output

**`test_file_loader.py`** — File loading:
- All expected extensions are in supported list
- TXT files load correctly
- CSV files parse into text
- Unsupported extensions raise errors
- Empty files are handled gracefully

**`test_router.py`** — Query routing:
- Queries without documents route to web search
- Document keywords trigger document Q&A
- Web/time keywords trigger web search
- Ambiguous queries with documents default to document Q&A

---

## 12. Evaluation System

**File**: `eval/evaluate.py`

Benchmarks the RAG system with predefined test cases.

**Metrics**:
| Metric | Description | Threshold |
|--------|-------------|-----------|
| Type Correctness | Was the query routed to the correct pipeline? | Exact match |
| Keyword Coverage | Do expected keywords appear in the answer? | ≥ 40% |
| Answer Quality | Is the answer between 20-5000 characters? | Length check |
| Source Count | How many sources were cited? | Count |

**Run evaluation**:
```bash
python eval/evaluate.py
```

**Output**: Results saved to `eval/eval_results.json` with per-test breakdown and overall pass rates.

---

## 13. How to Start the Project Locally

### 13.1 Local Development (Without Docker)

#### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Git

#### Step-by-Step

**1. Clone the repository**:
```bash
git clone <repository-url>
cd rag-open
```

**2. Create a virtual environment**:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**:
```bash
pip install -r requirements.txt
```

**4. Set up environment variables** (optional — defaults work out of the box):
Create a `.env` file in the project root:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=tinyllama
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=100
TOP_K=5
USE_RERANKER=true
APP_PORT=8011
```

**5. Install and pull an Ollama model**:
```bash
# Install Ollama from https://ollama.ai/

# Pull a model (choose one):
ollama pull tinyllama        # Smallest, fastest (~1.1GB)
ollama pull phi3             # Good balance (~2.3GB)
ollama pull llama3.1:8b      # Best quality (~4.7GB)
```

**6. (Optional) Start SearXNG for web search**:
```bash
# Using Docker for just SearXNG:
docker run -d -p 8888:8080 \
  -v ./searxng:/etc/searxng \
  --name searxng \
  searxng/searxng:latest
```

**7. Start the API server**:
```bash
python main.py
# Server starts at http://localhost:8011
# API docs at http://localhost:8011/docs
```

**8. Start the Streamlit UI** (in a new terminal):
```bash
streamlit run app/ui/streamlit_app.py
# UI starts at http://localhost:8501
```

**9. Test it**:
- Open `http://localhost:8501` in your browser
- Upload a document via the sidebar
- Ask questions about your document
- Try web search queries like "latest news about AI"

### 13.2 Using Docker (Recommended)

This is the easiest way — everything runs in containers.

#### Prerequisites
- [Docker](https://www.docker.com/get-started) installed
- [Docker Compose](https://docs.docker.com/compose/) installed
- At least 8GB RAM available

#### Step-by-Step

**1. Clone and navigate**:
```bash
git clone <repository-url>
cd rag-open
```

**2. Build and start all services**:
```bash
docker-compose up --build
```

This starts 4 services:
| Service | Port | Description |
|---------|------|-------------|
| API | `8011` | FastAPI backend |
| Streamlit | `8501` | Web UI |
| Ollama | `11434` | LLM inference engine |
| SearXNG | `8888` | Meta-search engine |

**3. Pull an LLM model** (first time only):
```bash
docker exec -it rag-open-ollama-1 ollama pull tinyllama
```

**4. Access the application**:
- **Chat UI**: http://localhost:8501
- **API Docs**: http://localhost:8011/docs
- **Health Check**: http://localhost:8011/health

**5. Stop the services**:
```bash
docker-compose down
```

**6. Stop and remove volumes** (reset all data):
```bash
docker-compose down -v
```

---

## 14. How to Deploy for Everyone to Use

### 14.1 Deploy on a Cloud VM

This is the most straightforward deployment method — run Docker Compose on a cloud server.

#### General Steps (Any Cloud Provider)

**1. Provision a VM**:
- **Minimum**: 4 vCPUs, 8GB RAM, 50GB SSD
- **Recommended**: 8 vCPUs, 16GB RAM, 100GB SSD (for larger LLM models)
- **GPU** (optional): For faster inference, use a GPU instance (e.g., NVIDIA T4)
- **OS**: Ubuntu 22.04 LTS

**2. Install Docker on the VM**:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Add your user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**3. Clone and deploy**:
```bash
git clone <repository-url>
cd rag-open

# Start all services
docker-compose up -d --build

# Pull the LLM model
docker exec -it rag-open-ollama-1 ollama pull tinyllama
```

**4. Configure firewall** (open required ports):
```bash
# Allow Streamlit UI
sudo ufw allow 8501/tcp

# Allow API (if direct access needed)
sudo ufw allow 8011/tcp

# Enable firewall
sudo ufw enable
```

**5. Set up a reverse proxy with NGINX** (recommended for production):
```bash
sudo apt install nginx -y
```

Create `/etc/nginx/sites-available/rag`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Streamlit UI
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    # API
    location /api/ {
        proxy_pass http://localhost:8011/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**6. Add SSL with Let's Encrypt** (for HTTPS):
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

### 14.2 Deploy on AWS EC2

**1. Launch an EC2 instance**:
- AMI: Ubuntu 22.04 LTS
- Instance type: `t3.xlarge` (4 vCPU, 16GB RAM) or `g4dn.xlarge` (with GPU)
- Storage: 100GB gp3 SSD
- Security Group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

**2. SSH and deploy**:
```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# Install Docker (same as above)
# Clone repo and run docker-compose up -d --build
```

**3. (Optional) Use Elastic IP** for a static public IP address.

**4. (Optional) Use Route 53** to point your domain to the EC2 instance.

### 14.3 Deploy on Google Cloud (GCE)

**1. Create a Compute Engine VM**:
```bash
gcloud compute instances create rag-server \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --tags=http-server,https-server
```

**2. Allow firewall rules**:
```bash
gcloud compute firewall-rules create allow-rag \
  --allow=tcp:8501,tcp:8011 \
  --target-tags=http-server
```

**3. SSH and deploy** (same Docker steps as above).

### 14.4 Deploy on Azure VM

**1. Create a VM**:
```bash
az vm create \
  --resource-group rag-rg \
  --name rag-server \
  --image Ubuntu2204 \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

**2. Open ports**:
```bash
az vm open-port --port 8501 --resource-group rag-rg --name rag-server
az vm open-port --port 8011 --resource-group rag-rg --name rag-server --priority 1010
```

**3. SSH and deploy** (same Docker steps as above).

### 14.5 Production Checklist

Before making the system publicly available, ensure:

- [ ] **Change SearXNG secret key** in `searxng/settings.yml` (default is `rag-searxng-secret-key-change-me`)
- [ ] **Set `DEBUG=false`** in environment variables
- [ ] **Use HTTPS** — Set up SSL certificates with Let's Encrypt or your cloud provider
- [ ] **Set up a reverse proxy** (NGINX or Traefik) in front of the application
- [ ] **Use a larger LLM model** for better quality (e.g., `llama3.1:8b` instead of `tinyllama`)
- [ ] **Persistent conversation storage** — Replace in-memory storage with Redis or PostgreSQL
- [ ] **Add authentication** — Implement API key or OAuth authentication for the endpoints
- [ ] **Set up monitoring** — Use tools like Prometheus + Grafana or cloud-native monitoring
- [ ] **Configure log rotation** — Prevent disk from filling with logs
- [ ] **Set up backups** — Regular backups of `data/chroma_db/` and `data/uploads/`
- [ ] **Rate limiting** — Add rate limiting to prevent abuse
- [ ] **Resource limits** — Set Docker memory/CPU limits in docker-compose.yml
- [ ] **Domain name** — Point a domain to your server for a clean URL

---

## 15. Environment Variables Reference

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `APP_NAME` | `RAG System` | No | Application name |
| `APP_HOST` | `0.0.0.0` | No | Bind address |
| `APP_PORT` | `8011` | No | API port |
| `DEBUG` | `true` | No | Debug mode (set `false` in production) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | No | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | `cpu` | No | `cpu` or `cuda` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Yes | Ollama server URL |
| `OLLAMA_MODEL` | `tinyllama` | No | Default LLM model name |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | No | ChromaDB storage path |
| `COLLECTION_NAME` | `rag_documents` | No | ChromaDB collection name |
| `CHUNK_SIZE` | `512` | No | Text chunk size (chars) |
| `CHUNK_OVERLAP` | `100` | No | Overlap between chunks |
| `TOP_K` | `5` | No | Number of docs to retrieve |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | No | Reranking model |
| `USE_RERANKER` | `true` | No | Enable cross-encoder reranking |
| `SEARXNG_BASE_URL` | `http://localhost:8888` | No | SearXNG search engine URL |
| `WEB_SEARCH_MAX_RESULTS` | `10` | No | Max web search results |
| `WEB_SCRAPE_TIMEOUT` | `15` | No | Scraping timeout (seconds) |
| `UPLOAD_DIR` | `./data/uploads` | No | Document upload directory |
| `MAX_FILE_SIZE_MB` | `50` | No | Maximum upload file size |
| `API_URL` | `http://localhost:8011` | No | API URL (for Streamlit) |

---

## 16. Performance & Scalability Notes

### Current Performance Characteristics

| Operation | Approximate Time |
|-----------|-----------------|
| Document upload + indexing (10-page PDF) | 5-15 seconds |
| Embedding generation (per chunk) | ~10ms (CPU) |
| Vector search (ChromaDB) | ~50ms |
| Cross-encoder reranking (5 docs) | ~200ms |
| LLM generation (tinyllama) | 2-10 seconds |
| Web search + scraping | 3-15 seconds |

### Known Bottlenecks

1. **Embedding generation** — Batch-processed (64 docs at a time) but CPU-bound. Use `EMBEDDING_DEVICE=cuda` with a GPU for significant speedup.
2. **LLM inference** — Depends on Ollama and the chosen model. Larger models = better quality but slower. GPU acceleration helps dramatically.
3. **Web scraping** — 15-second timeout per URL. Multiple URLs are processed sequentially.
4. **Reranking** — Quadratic with document count. Consider disabling for large collections.

### Scaling Recommendations

- **For more users**: Put NGINX in front as a load balancer, run multiple API instances
- **For more documents**: Consider switching from ChromaDB to Qdrant or Weaviate
- **For better quality**: Use `llama3.1:8b` or larger models, increase `TOP_K` and `CHUNK_SIZE`
- **For faster inference**: Use GPU instances, consider vLLM instead of Ollama for production
- **For persistent conversations**: Replace in-memory storage with Redis or PostgreSQL
- **For high availability**: Deploy across multiple VMs with shared storage (e.g., S3 for uploads, managed vector DB)




---
---

# nanoGPT — Full Codebase Explanation

## What Is This Project?

nanoGPT is a minimal, from-scratch implementation of the **GPT (Generative Pre-trained Transformer)** language model by Andrej Karpathy. It can:

1. **Train a GPT model from scratch** on any text dataset
2. **Fine-tune** OpenAI's pre-trained GPT-2 models on custom text
3. **Generate text** (sample) from a trained model

The entire core is just two files: `model.py` (~300 lines) defines the GPT architecture, and `train.py` (~300 lines) runs the training loop. Everything else is support code.

---

## High-Level Workflow

```
[1] Prepare Data          [2] Train Model          [3] Generate Text
    prepare.py        -->     train.py         -->     sample.py

  Raw text                 Reads train.bin/val.bin     Loads checkpoint
      |                    Builds/loads GPT model      Generates tokens
  Tokenize to integers     Trains with backprop        Decodes to text
      |                    Saves checkpoints
  Save as train.bin            |
  and val.bin              out/ckpt.pt
```

---

## File-by-File Breakdown

### `model.py` — The GPT Model (The Brain)

This file contains the **entire GPT-2 architecture** in pure PyTorch. Here's how the pieces fit together:

#### Building Blocks (bottom-up):

| Class | What It Does |
|-------|-------------|
| `LayerNorm` | Normalizes activations within a layer. Custom version that supports optional bias. |
| `CausalSelfAttention` | The core attention mechanism. Each token looks at all **previous** tokens (not future ones — that's the "causal" part) to decide what's relevant. Uses Flash Attention when available for speed. |
| `MLP` | A two-layer feed-forward network (expand 4x → GELU activation → project back). Processes each token independently after attention has mixed information across tokens. |
| `Block` | One Transformer block = LayerNorm → Attention → LayerNorm → MLP, with residual (skip) connections around each. |
| `GPTConfig` | A dataclass holding model hyperparameters: number of layers, heads, embedding size, vocab size, etc. |
| `GPT` | The full model. Stacks N `Block`s together with token + position embeddings at the input and a classification head at the output. |

#### How a Forward Pass Works:

```
Input token IDs [batch, sequence_length]
        |
   Token Embedding (lookup table: token → vector)
   + Position Embedding (lookup table: position → vector)
        |
   Dropout
        |
   Block 0: LayerNorm → Self-Attention → LayerNorm → MLP
   Block 1: LayerNorm → Self-Attention → LayerNorm → MLP
   ...
   Block N: LayerNorm → Self-Attention → LayerNorm → MLP
        |
   Final LayerNorm
        |
   Linear Head (project back to vocabulary size)
        |
   Output logits [batch, sequence_length, vocab_size]
   (probability distribution over next token)
```

#### Key Methods on `GPT`:

- **`forward(idx, targets)`** — Run input tokens through the model. If targets are given, also computes cross-entropy loss.
- **`from_pretrained(model_type)`** — Load official OpenAI GPT-2 weights from HuggingFace and map them into this model's format.
- **`configure_optimizers(...)`** — Creates an AdamW optimizer with weight decay for 2D parameters (weights) and no decay for 1D parameters (biases, layer norms).
- **`generate(idx, max_new_tokens)`** — Autoregressive text generation: predict next token, append it, repeat.
- **`crop_block_size(block_size)`** — Shrink the model's context window (useful when loading a pretrained model but wanting a smaller context).
- **`estimate_mfu(...)`** — Estimates Model FLOPs Utilization (what % of your GPU's theoretical peak you're achieving).

---

### `train.py` — The Training Loop (The Coach)

This is the main script you run to train or fine-tune a model. It handles everything:

#### Configuration (lines 1-70):
All hyperparameters are defined as plain Python variables at the top:
- **I/O**: `out_dir`, `eval_interval`, `always_save_checkpoint`
- **Model**: `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`
- **Optimization**: `learning_rate`, `max_iters`, `weight_decay`, `grad_clip`
- **LR Schedule**: cosine decay with linear warmup
- **System**: `device`, `dtype`, `compile` (PyTorch 2.0 compilation)

These get overridden by config files and/or command-line args via `configurator.py`.

#### Initialization:
1. **DDP Setup** — Detects if running with `torchrun` for multi-GPU training. If so, sets up distributed data parallel.
2. **Data Loading** — `get_batch()` reads random chunks from `train.bin`/`val.bin` using memory-mapped files (efficient for huge datasets).
3. **Model Init** — Three modes:
   - `'scratch'`: Random initialization, train from zero
   - `'resume'`: Load from a saved checkpoint and continue training
   - `'gpt2'`/`'gpt2-medium'`/etc.: Load pretrained OpenAI weights for fine-tuning
4. **Optimizer** — AdamW with separate weight-decay groups
5. **Compilation** — Optionally uses `torch.compile()` for faster execution

#### Training Loop:
```
For each iteration:
  1. Set learning rate (cosine schedule with warmup)
  2. Every eval_interval steps:
     - Estimate train/val loss (average over eval_iters batches)
     - Save checkpoint if val loss improved
     - Log to wandb (optional)
  3. Forward pass:
     - Run micro-batches for gradient accumulation
     - Each micro-batch: forward → loss → backward
  4. Gradient clipping
  5. Optimizer step
  6. Log loss, time, MFU
  7. Repeat until max_iters
```

#### Key Concepts in the Training Loop:
- **Gradient Accumulation**: Simulates larger batch sizes by accumulating gradients over multiple micro-batches before stepping the optimizer.
- **Mixed Precision (AMP)**: Uses `float16` or `bfloat16` for faster training with `GradScaler` to prevent underflow.
- **DDP Gradient Sync**: Only syncs gradients across GPUs on the last micro-step for efficiency.

---

### `sample.py` — Text Generation (The Output)

Loads a trained model and generates text:

1. Load model from checkpoint (`out/ckpt.pt`) or from pretrained GPT-2
2. Set up the tokenizer:
   - If a `meta.pkl` exists (character-level), use its encoder/decoder
   - Otherwise, use GPT-2's BPE tokenizer (tiktoken)
3. Encode the starting prompt into token IDs
4. Call `model.generate()` autoregressively in a loop
5. Decode and print the results

Key parameters: `temperature` (randomness), `top_k` (limit sampling pool), `num_samples`, `max_new_tokens`.

---

### `configurator.py` — Configuration System (The Settings)

A clever "poor man's configurator" — it's not imported as a module; instead, it's `exec()`-ed inside `train.py` so it can directly override global variables.

Two ways to override defaults:
1. **Config file**: `python train.py config/train_gpt2.py` — executes the Python file, which sets variables
2. **CLI args**: `python train.py --batch_size=32 --learning_rate=1e-4` — parses and overrides individual values

---

### `bench.py` — Benchmarking (The Speedometer)

A stripped-down version of `train.py` for measuring training speed:
- Runs a few training iterations (10 burn-in + 20 measured)
- Reports time per iteration and MFU (Model FLOPs Utilization)
- Optional PyTorch profiler support for detailed analysis
- Can run with real data or random synthetic data

---

### `config/` — Preset Configurations

| File | Purpose |
|------|---------|
| `train_gpt2.py` | Train GPT-2 124M from scratch on OpenWebText (8×A100, ~4 days) |
| `train_shakespeare_char.py` | Train a tiny character-level GPT on Shakespeare (single GPU, ~3 min) |
| `finetune_shakespeare.py` | Fine-tune GPT-2 XL on Shakespeare (single GPU, minutes) |
| `eval_gpt2.py` | Evaluate pretrained GPT-2 124M on OpenWebText |
| `eval_gpt2_medium.py` | Evaluate pretrained GPT-2 350M |
| `eval_gpt2_large.py` | Evaluate pretrained GPT-2 774M |
| `eval_gpt2_xl.py` | Evaluate pretrained GPT-2 1.5B |

These are just Python files that set variables — they get exec'd by the configurator.

---

### `data/` — Dataset Preparation

Each subfolder has a `prepare.py` script that downloads and preprocesses text:

| Folder | Tokenization | Data Size | What It Does |
|--------|-------------|-----------|-------------|
| `openwebtext/` | GPT-2 BPE (tiktoken) | ~17GB train, ~8.5MB val | Downloads the OpenWebText dataset from HuggingFace, tokenizes ~8M documents, writes `train.bin` + `val.bin` |
| `shakespeare/` | GPT-2 BPE (tiktoken) | ~300K tokens | Downloads Tiny Shakespeare, encodes with GPT-2 tokenizer. Used for **fine-tuning** pretrained GPT-2. |
| `shakespeare_char/` | Character-level | ~1M characters | Downloads Tiny Shakespeare, maps each character to an integer. Used for **training from scratch**. Also saves `meta.pkl` with the character→int mapping. |

All output the same format: `train.bin` and `val.bin` files containing uint16 token IDs that `train.py` can directly memory-map.

---

## How to Understand the Code (Reading Order)

If you want to understand nanoGPT, read the files in this order:

### Step 1: Understand the Model (`model.py`)
Start here. Read bottom-up:
1. `GPTConfig` — what hyperparameters define a GPT
2. `LayerNorm` — simple normalization
3. `CausalSelfAttention` — the heart of the Transformer. Understand Q/K/V projections, multi-head splitting, causal masking, and how attention scores are computed.
4. `MLP` — straightforward feed-forward network
5. `Block` — puts attention + MLP together with residual connections
6. `GPT.__init__` — how the full model is assembled from blocks
7. `GPT.forward` — how data flows through: embeddings → blocks → output head → loss
8. `GPT.generate` — token-by-token autoregressive generation

### Step 2: Understand Data Prep (`data/shakespeare_char/prepare.py`)
The simplest example. See how raw text becomes binary token files.

### Step 3: Understand Training (`train.py`)
Read the main training loop. Key things to understand:
- How `get_batch()` loads random chunks from binary files
- The gradient accumulation loop (inner `for micro_step`)
- Mixed precision context (`ctx`)
- The learning rate schedule (`get_lr`)
- Checkpointing logic

### Step 4: Understand Sampling (`sample.py`)
See how a trained model generates text token by token.

### Step 5: Understand Configuration (`configurator.py`)
Short file. See how config files and CLI args override defaults.

---

## Key Concepts You Need to Know

### Tokenization
Text must be converted to numbers. Two approaches used here:
- **Character-level**: Each character is a token (vocab size = ~65 for Shakespeare)
- **BPE (Byte Pair Encoding)**: Subword tokenization used by GPT-2 (vocab size = 50,257)

### Autoregressive Language Modeling
The model is trained to predict the **next token** given all previous tokens. During generation, it predicts one token at a time and feeds it back as input.

### Causal Masking
In self-attention, each token can only attend to tokens **before** it (not after). This is what makes the model generate text left-to-right.

### Residual Connections
Each block adds its output to its input (`x = x + block(x)`). This helps gradients flow through deep networks.

### Weight Tying
The token embedding matrix and the final output projection share the same weights. This reduces parameter count and improves performance.

### Mixed Precision Training
Using `float16` or `bfloat16` instead of `float32` for most operations. Faster and uses less memory, with `GradScaler` preventing numerical issues (only needed for `float16`).

### Gradient Accumulation
When you can't fit a large batch in GPU memory, you process smaller micro-batches and accumulate gradients before updating weights. Mathematically equivalent to a larger batch.

### Distributed Data Parallel (DDP)
For multi-GPU training. Each GPU processes different data, gradients are averaged across GPUs before the optimizer step.

---

## Common Workflows

### Train a small model from scratch (quickstart):
```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

### Train on YOUR OWN custom text (books, Wikipedia, articles):
```bash
# 1. Put your .txt files in data/custom/input/
# 2. Prepare the data:
python data/custom/prepare.py
# 3. Train:
python train.py config/train_custom.py
# 4. Generate:
python sample.py --out_dir=out-custom
```

### RAG-enhanced generation (search the web + generate):
```bash
# Single query:
python rag_sample.py --init_from=gpt2-xl --start="What is quantum computing?"

# Interactive mode:
python rag_sample.py --out_dir=out-custom

# With more search depth:
python rag_sample.py --init_from=gpt2 --num_search_results=10 --max_pages=8
```

### Fine-tune GPT-2 on custom text:
```bash
python data/shakespeare/prepare.py
python train.py config/finetune_shakespeare.py
python sample.py --out_dir=out-shakespeare
```

### Reproduce GPT-2 124M:
```bash
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
python sample.py
```

### Evaluate pretrained GPT-2:
```bash
python train.py config/eval_gpt2.py
```

---

## New Features Added

### 1. Custom Data Training Pipeline (`data/custom/prepare.py`)

Train on **any text data** — books, Wikipedia articles, your own writing, anything:

```
data/custom/
  input/           ← Drop your .txt / .md files here
  prepare.py       ← Run this to tokenize everything
  train.bin        ← Created automatically
  val.bin          ← Created automatically
```

**Features:**
- Reads all `.txt`, `.md`, `.text`, `.csv`, `.html` files recursively from a directory
- Supports both **BPE** (GPT-2 tokenizer) and **character-level** encoding
- Automatic train/val split at document boundaries
- Handles encoding issues (UTF-8 with Latin-1 fallback)
- Configurable via CLI args: `--encoding=char`, `--val_fraction=0.1`, `--input_dir=path/`, etc.

**Examples:**
```bash
# BPE tokenization (default, for fine-tuning GPT-2):
python data/custom/prepare.py

# Character-level (for training from scratch):
python data/custom/prepare.py --encoding=char

# From a single file:
python data/custom/prepare.py --input_file=mybook.txt

# From a custom directory:
python data/custom/prepare.py --input_dir=C:/my_wikipedia_dumps/
```

### 2. RAG-Enhanced Generation (`rag_sample.py` + `search_utils.py`)

**Retrieval-Augmented Generation**: the model searches the web before answering.

```
User Prompt
    |
    ├── [1] Extract Topics  →  "quantum computing", "quantum applications"
    |
    ├── [2] Search Web      →  DuckDuckGo search for each topic
    |
    ├── [3] Fetch Pages     →  Top 5-10 links, extract text content
    |
    ├── [4] Build Context   →  Assemble retrieved text as context
    |
    └── [5] Generate        →  Context + Question → Model → Answer
```

**How it works:**
1. `search_utils.py` extracts keywords from your prompt (removes stopwords, identifies topics)
2. Searches DuckDuckGo for each topic (no API key needed)
3. Fetches the top pages and extracts clean text (strips HTML, scripts, nav, etc.)
4. Prepends the gathered context to your prompt
5. The model generates a response informed by the web content

**Files:**
- `search_utils.py` — Topic extraction, web search, page scraping, context assembly
- `rag_sample.py` — Ties it all together: loads model + runs RAG pipeline + generates

### 3. Training Config for Custom Data (`config/train_custom.py`)

Pre-configured settings for training on custom datasets with different hardware:
- **~8GB GPU** (default): 8-layer model, 512 context, batch 8
- **~4GB GPU**: 6-layer model, 256 context (commented preset)
- **~16GB+ GPU**: 12-layer model, 1024 context (commented preset)
- **CPU/MacBook**: 4-layer model, 256 context (commented preset)

### Required Additional Packages

```bash
pip install duckduckgo-search beautifulsoup4 requests
```

These are only needed for the RAG feature. Training on custom data only needs the original dependencies.
