# Academic RAG System — Portuguese Edition (v1.1.3)

> **Branch:** `feature/v1.1.3-portuguese-rag`
> **Base:** `main` @ `v1.2.0-qdrant-integration-complete`
> **Status:** Production-Ready — Brazilian Portuguese (PT-BR)

This branch constitutes the production-ready Portuguese localization of the Academic RAG System. It supersedes the English-language baseline (`main`) with full PT-BR interface localization, a higher-dimensional embedding strategy optimized for multilingual semantic retrieval, and a neutral, institutionally appropriate user interface designed for deployment in formal academic environments.

---

## Core Updates — Architectural Decisions

### 1. Full PT-BR Localization

The system has been localized end-to-end for Brazilian Portuguese at a formal academic register (nível culto). This covers three distinct layers:

- **Streamlit frontend (`streamlit_app/main.py`):** All interface strings, status messages, and error outputs are rendered exclusively in PT-BR. No titles, personal identifiers, or informal greetings are present.
- **n8n AI Agent system prompt (`n8n_workflow/rag_workflow.json`):** The agent operates under a strict PT-BR specialist persona, instructed to respond formally and to explicitly acknowledge when retrieved context is insufficient rather than hallucinate.
- **Ingestion pipeline (`ingest.py`):** Model selection and all log output are aligned with the PT-BR production configuration.

### 2. Embedding Strategy — Model Selection Rationale

**Target model (original):** `neuralmind/bert-base-portuguese-cased` (BERTimbau)

BERTimbau was the initial candidate for this branch due to its pre-training on a large Brazilian Portuguese corpus. However, FastEmbed — the ONNX-based inference library used for local, GPU-accelerated embedding — does not include a pre-built ONNX export for this model in its registry. Attempting to instantiate it raises:

```
ValueError: Model neuralmind/bert-base-portuguese-cased is not supported in TextEmbedding.
```

**Adopted model:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

| Property | Value |
|---|---|
| Architecture | Multilingual BERT (mBERT) + mean pooling |
| Training data | 50+ languages, including Brazilian Portuguese |
| Output dimensions | **768** |
| Similarity metric | Cosine similarity |
| FastEmbed ONNX support | Yes — native registry entry |
| Execution backends | CoreML (Apple MPS), CUDA (NVIDIA), CPU |

This model is fine-tuned on paraphrase data across 50+ languages using a sentence-transformers training pipeline, making it well-suited for semantic retrieval tasks in PT-BR technical and academic corpora. Its 768-dimensional output matches the dimensionality originally specified for BERTimbau, preserving the intended vector space configuration.

### 3. Vector Database Configuration

- **Engine:** Qdrant (Dockerized, container: `rag_qdrant`)
- **Collection:** `rag_documents`
- **Vector size:** 768 dimensions
- **Distance metric:** Cosine similarity
- **ID scheme:** Deterministic UUID v5 — re-ingestion is idempotent; existing chunks are overwritten, not duplicated
- **Persistence:** Bind-mounted volume at `./qdrant_data:/qdrant/storage`

### 4. Neutral UI Design — Titleless / Zero-Greeting Interface

The Streamlit interface has been redesigned to conform to institutional deployment standards:

- **No page title rendered in the UI body** — `st.title()` is absent
- **No personal identifiers, names, or informal greetings**
- **Professional landing caption:** `"Sistema de análise documental ativo. Insira sua consulta técnica abaixo."`
- **All interaction strings in formal PT-BR:**

| Element | String |
|---|---|
| Input placeholder | `"Digite sua pergunta técnica..."` |
| Processing spinner | `"Consultando base de dados técnica..."` |
| Connection error | `"Erro de conexão com o servidor de processamento."` |
| Timeout error | `"A solicitação excedeu o tempo limite. O modelo pode estar sendo inicializado."` |

---

## System Requirements

### Runtime Dependencies

| Component | Version | Purpose |
|---|---|---|
| Docker Desktop | 24+ | Container orchestration |
| Python | 3.10+ | Ingestion pipeline and frontend |
| `fastembed` | ≥ 0.3.6 | ONNX-accelerated local embedding |
| `qdrant-client` | ≥ 1.9.0 | Qdrant REST/gRPC client |
| `pypdf` | ≥ 4.3.0 | PDF text extraction |
| `onnxruntime` | ≥ 1.18.0 | ONNX Runtime — CPU (default) |
| `streamlit` | ≥ 1.35.0 | Frontend framework |
| `requests` | ≥ 2.31.0 | HTTP client for webhook communication |

> **NVIDIA GPU:** Replace `onnxruntime` with `onnxruntime-gpu>=1.18.0` in `requirements-ingest.txt`.
> **Apple Silicon:** CoreML support is bundled in the standard `onnxruntime` package on macOS — no additional packages required.

### Install

```bash
# Ingestion pipeline
pip install -r requirements-ingest.txt

# Streamlit frontend
pip install -r streamlit_app/requirements.txt
```

---

## Installation & Migration Guide

### Step 1 — Check Out This Branch

```bash
git checkout feature/v1.1.3-portuguese-rag
git pull origin feature/v1.1.3-portuguese-rag
```

### Step 2 — Start the Docker Stack

```bash
docker compose up -d
```

Verify all four containers are running:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected output:

```
NAMES          STATUS           PORTS
rag_qdrant     Up               0.0.0.0:6333-6334->6333-6334/tcp
rag_n8n        Up               0.0.0.0:5678->5678/tcp
rag_postgres   Up (healthy)     5432/tcp
rag_ollama     Up               0.0.0.0:11434->11434/tcp
```

### Step 3 — Migrate the Qdrant Collection

> **This step is mandatory when upgrading from any previous branch.**
> The baseline `main` branch used `BAAI/bge-small-en-v1.5` (384 dimensions). This branch uses a 768-dimensional model. Qdrant enforces strict dimensionality per collection — inserting 768-dim vectors into a 384-dim collection will raise a validation error.

Delete the existing collection:

```bash
curl -X DELETE http://localhost:6333/collections/rag_documents
```

Expected response: `{"result":true,"status":"ok"}`

### Step 4 — Local Execution Backend (Apple Silicon)

The ingestion pipeline detects the available ONNX Runtime execution provider at startup and selects the optimal backend automatically:

```
Priority: CoreML (Apple MPS / Neural Engine) → CUDA (NVIDIA) → CPU
```

On Apple Silicon (M-series), the pipeline will log:

```
[INFO] Execution backend: CoreML (Apple MPS / Neural Engine)
```

No additional configuration is required. CoreML support is included in the standard `onnxruntime` package distributed for `macosx_14_0_arm64`.

> **Note:** CoreML does not accelerate all ONNX operators. The runtime will log partition warnings (e.g., `number of nodes supported by CoreML: 448 / 637`). This is expected behavior — unsupported operators fall back to CPU transparently. `Context leak detected` messages from CoreAnalytics are macOS system-level logging artifacts and do not affect correctness or output.

### Step 5 — Ingest Documents

Place source documents in `knowledge_base/` (supported formats: `.pdf`, `.txt`, `.md`, `.rst`), then run:

```bash
python3 ingest.py
```

Expected terminal output:

```
[INFO] Found N document(s) to ingest.
[INFO] Execution backend: CoreML (Apple MPS / Neural Engine)
[INFO] Loading embedding model 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'…
[INFO] Vector dimensionality: 768
[INFO] Connecting to Qdrant at http://localhost:6333…
[INFO] Created collection 'rag_documents' (dim=768, metric=cosine).
[INFO] Processing: knowledge_base/<document>.pdf
[INFO]   Upserted N chunk(s).
[INFO] Ingestion complete. N total chunk(s) upserted into 'rag_documents'.
```

To override the Qdrant endpoint (e.g., Linux VM or remote server):

```bash
QDRANT_URL=http://production-host:6333 python3 ingest.py
```

### Step 6 — Import the n8n Workflow

1. Open `http://localhost:5678`
2. Navigate to **Workflows → Import from file**
3. Select `n8n_workflow/rag_workflow.json`
4. Map credentials when prompted:
   - **Ollama API** → Base URL: `http://ollama:11434`
   - **Qdrant API** → URL: `http://qdrant:6333`
   - **HuggingFace Token** → API token from huggingface.co/settings/tokens
5. Activate the workflow

### Step 7 — Launch the Frontend

```bash
cd streamlit_app
streamlit run main.py
```

Interface available at `http://localhost:8501`.

---

## Reproducibility Notes

All changes in this branch are deterministic and reproducible:

- **Chunk IDs** are UUID v5 hashes derived from `(filename, chunk_index, text_prefix)` — identical inputs always produce identical IDs
- **Model weights** are downloaded once by FastEmbed and cached locally; subsequent runs use the cached ONNX model
- **Qdrant collection** is created with fixed parameters (`dim=768`, `Distance.COSINE`) — no runtime variation
- **n8n workflow** is fully defined in `rag_workflow.json` and version-controlled — workflow state is reproducible via import

---

## Diff Summary vs. `main`

| File | Change |
|---|---|
| `ingest.py` | Model: `BAAI/bge-small-en-v1.5` → `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`; `VECTOR_DIM=768`; BERTimbau rationale documented |
| `streamlit_app/main.py` | Full PT-BR localization; title removed; neutral institutional UI |
| `n8n_workflow/rag_workflow.json` | System prompt → PT-BR specialist persona; embedding model updated; temperature `0.2` → `0.1` |

---

## License

Developed for academic research purposes at UFRJ. All third-party components (Ollama, Qdrant, HuggingFace, n8n, FastEmbed) are subject to their respective licenses.
