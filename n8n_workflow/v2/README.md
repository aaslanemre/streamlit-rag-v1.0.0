# Power Systems Orchestrator v2.1.0 — Multi-Agent + PostgreSQL Session State

> **Branch:** `feature/v2.1.0-postgres-orchestrator`
> **Base:** `feature/v1.1.3-portuguese-rag`
> **Status:** Production-Ready — Brazilian Portuguese (PT-BR)

This version supersedes the single-agent RAG architecture (`v1.1.3`) with a full **Multi-Agent Orchestration System** backed by **PostgreSQL session state management**. The orchestrator classifies each query into one of three power system study domains and routes it to the appropriate specialist sub-workflow.

---

## Architecture Overview

```
Chat Webhook
     │
     ▼
Fetch Session  ──── PostgreSQL (SELECT session state)
     │
     ▼
AI Agent (Tools Agent)  ◄── Ollama Chat Model (llama3.2)
     │                  ◄── Window Buffer Memory (10 turns)
     │                  ◄── anarede_tool  ──► ANAREDE sub-workflow
     │                  ◄── anatem_tool   ──► ANATEM sub-workflow
     │                  ◄── plexos_tool   ──► PLEXOS sub-workflow
     │
     ▼
Update Session  ──── PostgreSQL (UPSERT session state)
     │
     ▼
Respond to Webhook
     │
     { output, session_id, study_scope, message_count }
```

---

## What's New vs v1.1.3

| Feature | v1.1.3 | v2.1.0 |
|---|---|---|
| Agent type | `conversationalRetrievalAgent` | `toolsAgent` |
| Routing | Single RAG chain | 3 specialist sub-workflows |
| Session memory | Window Buffer Memory only | PostgreSQL persistent state + Window Buffer Memory |
| Scope detection | None | SQL `ILIKE` keyword matching on agent output |
| Response payload | `{ output, session_id }` | `{ output, session_id, study_scope, message_count }` |
| Node versions | Mixed | All latest (no Deprecated warnings) |

---

## Node Versions

| Node | Type | typeVersion |
|---|---|---|
| Chat Webhook | `n8n-nodes-base.webhook` | 2 |
| Fetch Session | `n8n-nodes-base.postgres` | 2.5 |
| AI Agent | `@n8n/n8n-nodes-langchain.agent` | 1.8 |
| Update Session | `n8n-nodes-base.postgres` | 2.5 |
| Respond to Webhook | `n8n-nodes-base.respondToWebhook` | 1.1 |
| Ollama Chat Model | `@n8n/n8n-nodes-langchain.lmChatOllama` | 1.3 |
| Window Buffer Memory | `@n8n/n8n-nodes-langchain.memoryBufferWindow` | 1.3 |
| Anarede / Anatem / Plexos Tool | `@n8n/n8n-nodes-langchain.toolWorkflow` | 1.2 |

---

## PostgreSQL Session Table

Run once against the `n8n` database before activating the workflow:

```bash
docker exec -it rag_postgres psql -U postgres -d n8n -c "
CREATE TABLE IF NOT EXISTS power_system_sessions (
  session_id     TEXT PRIMARY KEY,
  study_scope    TEXT DEFAULT 'undefined',
  last_tool_used TEXT,
  last_response  TEXT,
  message_count  INTEGER DEFAULT 0,
  created_at     TIMESTAMPTZ DEFAULT NOW(),
  updated_at     TIMESTAMPTZ DEFAULT NOW()
);"
```

### Schema

| Column | Type | Description |
|---|---|---|
| `session_id` | TEXT (PK) | Unique session identifier from the Streamlit frontend |
| `study_scope` | TEXT | Detected domain: `electrical-static`, `electrical-dynamic`, `energetic`, `undefined` |
| `last_tool_used` | TEXT | Last specialist tool called: `anarede_tool`, `anatem_tool`, `plexos_tool` |
| `last_response` | TEXT | Full text of the agent's last response |
| `message_count` | INTEGER | Total messages in session — incremented on every turn |
| `created_at` | TIMESTAMPTZ | Session creation timestamp |
| `updated_at` | TIMESTAMPTZ | Last activity timestamp |

### Scope Detection Logic

The `Update Session` node detects the study scope from the agent's output using PostgreSQL `ILIKE` keyword matching — no extra Code node required:

| Detected scope | Trigger keywords |
|---|---|
| `electrical-static` | `anarede`, `fluxo de pot`, `PWF`, `power flow` |
| `electrical-dynamic` | `anatem`, `estabilidade`, `stability`, `transient` |
| `energetic` | `plexos`, `despacho`, `dispatch`, `energético` |
| `undefined` | No keyword match — scope preserved from previous turn |

---

## Setup Guide

### Step 1 — Check out this branch

```bash
git checkout feature/v2.1.0-postgres-orchestrator
git pull origin feature/v2.1.0-postgres-orchestrator
```

### Step 2 — Ensure the Docker stack is running

```bash
docker compose up -d
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected:

```
NAMES          STATUS           PORTS
rag_qdrant     Up               0.0.0.0:6333-6334->6333-6334/tcp
rag_n8n        Up               0.0.0.0:5678->5678/tcp
rag_postgres   Up (healthy)     5432/tcp
rag_ollama     Up               0.0.0.0:11434->11434/tcp
```

### Step 3 — Create the sessions table

```bash
docker exec -it rag_postgres psql -U postgres -d n8n -c "
CREATE TABLE IF NOT EXISTS power_system_sessions (
  session_id TEXT PRIMARY KEY,
  study_scope TEXT DEFAULT 'undefined',
  last_tool_used TEXT,
  last_response TEXT,
  message_count INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);"
```

### Step 4 — Import the workflow

1. Open `http://localhost:5678`
2. **Workflows → Import from file**
3. Select `n8n_workflow/v2/orchestrator_v2.1.0.json`
4. Map credentials:
   - **Ollama — Local** → Base URL: `http://ollama:11434`
   - **PostgreSQL — RAG** → Host: `rag_postgres`, Port: `5432`, Database: `n8n`, User: `postgres`

### Step 5 — Create the 3 specialist sub-workflows

Each tool node points to a separate n8n workflow. Create three workflows, each starting with a **When Called by Another Workflow** trigger node that receives a `query` field, then chains into its own RAG pipeline (Tools Agent + Qdrant Vector Store Tool + Ollama Embeddings).

Replace the placeholder IDs in each tool node:
- `REPLACE_WITH_ANAREDE_WORKFLOW_ID`
- `REPLACE_WITH_ANATEM_WORKFLOW_ID`
- `REPLACE_WITH_PLEXOS_WORKFLOW_ID`

### Step 6 — Activate and test

Enable the workflow toggle in n8n, then send a test request:

```bash
curl -X POST http://localhost:5678/webhook/power-systems-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Como faço uma análise de fluxo de potência no ANAREDE?", "session_id": "test-session-001"}'
```

Expected response:

```json
{
  "output": "...",
  "session_id": "test-session-001",
  "study_scope": "electrical-static",
  "message_count": 1
}
```

---

## Credential Summary

| Credential name | Type | Value (inside Docker) |
|---|---|---|
| Ollama — Local | Ollama API | `http://ollama:11434` |
| PostgreSQL — RAG | Postgres | Host: `rag_postgres`, DB: `n8n` |

---

## License

Developed for academic research purposes at UFRJ. All third-party components are subject to their respective licenses.
