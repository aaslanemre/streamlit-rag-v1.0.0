# RAG-Based Academic Document Analysis System

A Retrieval-Augmented Generation (RAG) system developed for automated technical document retrieval and synthesis, supporting academic research activities at the Universidade Federal do Rio de Janeiro (UFRJ). The system enables structured querying of domain-specific document corpora through a locally deployed, privacy-preserving architecture.

---

## Architecture Overview

```
┌─────────────────┐     HTTP/Webhook     ┌──────────────────────────────┐
│  Streamlit UI   │ ──────────────────► │         n8n AI Agent         │
│  (Frontend)     │ ◄────────────────── │  (Orchestration & Workflow)  │
└─────────────────┘      JSON Response   └──────────────┬───────────────┘
                                                        │
                          ┌─────────────────────────────┼───────────────────┐
                          │                             │                   │
                 ┌────────▼────────┐       ┌────────────▼──────┐  ┌────────▼────────┐
                 │  Ollama (LLM)   │       │  Qdrant           │  │  HuggingFace    │
                 │  Llama 3.2 /    │       │  Vector Database  │  │  Inference API  │
                 │  Mistral        │       │  (Dockerized)     │  │  (Embeddings)   │
                 └─────────────────┘       └───────────────────┘  └─────────────────┘
```

| Layer | Technology | Role |
|---|---|---|
| Frontend | Streamlit | Chat interface; communicates with n8n via HTTP webhook |
| Orchestration | n8n (AI Agent + Webhooks) | Workflow automation; routes queries through retrieval and generation pipeline |
| Vector Database | Qdrant (Docker) | Stores and retrieves document embeddings via cosine similarity search |
| Embeddings | HuggingFace Inference API | Encodes queries and documents into dense vector representations |
| LLM | Ollama — Llama 3.2 / Mistral | Local inference; generates responses grounded in retrieved context |

---

## Repository Structure

```
streamlit-rag-v1.0.0/
├── docker-compose.yml          # Service definitions: n8n, PostgreSQL, Ollama, Qdrant
├── ingest.py                   # Document ingestion pipeline (chunking → embedding → Qdrant)
├── requirements-ingest.txt     # Python dependencies for ingestion
├── knowledge_base/             # Drop source documents here (.pdf, .txt, .md, .rst)
├── qdrant_data/                # Qdrant persistent storage (bind-mounted volume)
├── streamlit_app/
│   ├── main.py                 # Streamlit chat interface
│   └── requirements.txt        # Python dependencies for the frontend
└── n8n_workflow/
    └── rag_workflow.json       # Importable n8n workflow (Webhook → Agent → Qdrant → LLM)
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (v24+)
- Python 3.10 or higher
- A HuggingFace account with an API token (huggingface.co/settings/tokens)

---

## Setup Guide

### Step 1 — Start the Docker Stack

```bash
docker compose up -d
```

This starts four services on a shared internal network (`rag_network`):

| Container | Port | Purpose |
|---|---|---|
| `rag_postgres` | — | n8n metadata database |
| `rag_n8n` | `5678` | Workflow orchestration UI |
| `rag_ollama` | `11434` | Local LLM inference |
| `rag_qdrant` | `6333` / `6334` | Vector database (REST / gRPC) |

### Step 2 — Pull the LLM

```bash
docker exec rag_ollama ollama pull llama3.2
```

To use Mistral instead:

```bash
docker exec rag_ollama ollama pull mistral
```

### Step 3 — Configure Environment Variables

The following variables can be set in the shell before running ingestion:

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `rag_documents` | Target collection name |

Example for a production or remote environment:

```bash
export QDRANT_URL=http://production-host:6333
export QDRANT_COLLECTION=rag_documents_prod
```

For n8n, the following credentials must be configured via the UI at `http://localhost:5678`:

- **Ollama API** — Base URL: `http://ollama:11434`
- **Qdrant API** — URL: `http://qdrant:6333`
- **HuggingFace Token** — API token from HuggingFace settings

### Step 4 — Install Python Dependencies

**Ingestion pipeline:**

```bash
pip install -r requirements-ingest.txt
```

**Streamlit frontend:**

```bash
pip install -r streamlit_app/requirements.txt
```

### Step 5 — Ingest Documents

Place source documents (`.pdf`, `.txt`, `.md`, or `.rst`) in the `knowledge_base/` directory, then execute:

```bash
python ingest.py
```

The ingestion pipeline will:
1. Detect the optimal execution backend (CoreML → CUDA → CPU)
2. Load the configured embedding model via FastEmbed
3. Extract text and split into overlapping chunks (512 characters, 64-character overlap)
4. Embed each chunk and upsert into the Qdrant collection

To specify a custom Qdrant endpoint or collection at runtime:

```bash
python ingest.py --qdrant-url http://production-host:6333 --collection my_collection
```

### Step 6 — Import the n8n Workflow

1. Open `http://localhost:5678` and complete the initial account setup
2. Navigate to **Workflows → Import from file**
3. Select `n8n_workflow/rag_workflow.json`
4. Map the three credentials (Ollama, Qdrant, HuggingFace) when prompted
5. Activate the workflow using the toggle in the top-right corner

### Step 7 — Launch the Frontend

```bash
cd streamlit_app
streamlit run main.py
```

The interface is accessible at `http://localhost:8501`.

---

## Technical Specifications

### Embedding Models

| Version | Model | Dimensions | Language |
|---|---|---|---|
| v1.0.0 — v1.2.0 | `BAAI/bge-small-en-v1.5` | 384 | English |
| v1.1.3+ | `neuralmind/bert-base-portuguese-cased` (BERTimbau) | 768 | Brazilian Portuguese |

From version `v1.1.3` onward, the system transitions to **BERTimbau** (`neuralmind/bert-base-portuguese-cased`), a BERT-base model pre-trained on a large Portuguese corpus by Neuralmind. This model delivers significantly improved semantic retrieval accuracy for technical and academic documents in Brazilian Portuguese (PT-BR).

> **Note:** Changing the embedding model requires deleting and recreating the Qdrant collection, as vector dimensionality changes from 384 to 768. Execute the following before re-ingesting:
> ```bash
> curl -X DELETE http://localhost:6333/collections/rag_documents
> python ingest.py
> ```

### Chunking Strategy

- **Method:** Overlapping character-level sliding window
- **Chunk size:** 512 characters
- **Overlap:** 64 characters
- **ID scheme:** Deterministic UUID v5 — re-ingestion overwrites existing chunks without duplication

### Vector Similarity

All collections use **cosine similarity**, which is appropriate for normalized BERT-family embeddings and remains stable across document lengths.

### LLM Options

The system supports any model available through Ollama. Recommended configurations:

| Model | Use Case |
|---|---|
| `llama3.2` | General-purpose retrieval-augmented generation |
| `mistral` | Higher instruction-following precision |

A Gemini API integration path is supported by replacing the Ollama Chat Model node in the n8n workflow with the Google Gemini node and supplying the corresponding API key credential.

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready releases |
| `feature/v1.1.3-portuguese-rag` | PT-BR configuration with BERTimbau embeddings |

---

## License

This project is developed for academic research purposes. All usage must comply with the terms of the respective third-party services and models referenced herein (Ollama, Qdrant, HuggingFace, n8n).
