#!/usr/bin/env python3
"""
ingest.py — Document ingestion pipeline for the RAG system (v1.1.3).

Embedding backend: Ollama (nomic-embed-text, 768 dims)
Language:          Brazilian Portuguese (PT-BR)

Using Ollama for embeddings ensures the ingestion vector space is identical
to the n8n query vector space — both call the same local Ollama endpoint
with the same model. This eliminates the inference-path mismatch that occurs
when using FastEmbed (local ONNX) for ingestion and HuggingFace Inference
API for query-time retrieval.

Execution is fully local — no external API keys required.
Qdrant URL is configurable via --qdrant-url flag or QDRANT_URL env var.
"""

import os
import sys
import uuid
import logging
import argparse
from pathlib import Path
from typing import Generator

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pypdf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR: Path  = Path(__file__).parent / "knowledge_base"
DEFAULT_QDRANT_URL: str   = "http://localhost:6333"
DEFAULT_OLLAMA_URL: str   = "http://localhost:11434"
DEFAULT_COLLECTION: str   = "rag_documents"

# nomic-embed-text — 768-dim, multilingual, already pulled in rag_ollama.
# Must match the model configured in the n8n Ollama Embeddings node.
DEFAULT_MODEL: str = "nomic-embed-text"
VECTOR_DIM: int    = 768

CHUNK_SIZE: int    = 512
CHUNK_OVERLAP: int = 64
BATCH_SIZE: int    = 32

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".rst"})


# ── Ollama Embedding Client ───────────────────────────────────────────────────
def embed_texts(texts: list[str], model: str, ollama_url: str) -> list[list[float]]:
    """
    Call Ollama's /api/embed endpoint and return a list of float vectors.
    Uses the batch endpoint (Ollama ≥ 0.1.32) for efficiency.
    Falls back to sequential /api/embeddings calls if batch is unavailable.
    """
    try:
        resp = requests.post(
            f"{ollama_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]
    except (KeyError, requests.HTTPError):
        # Fallback: sequential single-text calls
        vectors = []
        for text in texts:
            r = requests.post(
                f"{ollama_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60,
            )
            r.raise_for_status()
            vectors.append(r.json()["embedding"])
        return vectors


def probe_dimension(model: str, ollama_url: str) -> int:
    """Return the embedding dimensionality for a given Ollama model."""
    vectors = embed_texts(["__probe__"], model, ollama_url)
    return len(vectors[0])


# ── Text Extraction ───────────────────────────────────────────────────────────
def extract_text(path: Path) -> str:
    """Return full plain text for non-PDF files. Returns empty string on failure."""
    try:
        if path.suffix.lower() in {".txt", ".md", ".rst"}:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        log.error("Failed to read '%s': %s", path.name, exc)
    return ""


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Generator[str, None, None]:
    text = text.strip()
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == len(text):
            break
        start += size - overlap


# ── Qdrant Helpers ────────────────────────────────────────────────────────────
def ensure_collection(client: QdrantClient, name: str, vector_size: int = VECTOR_DIM) -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        log.info("Collection '%s' already exists — skipping creation.", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    log.info("Created collection '%s' (dim=%d, metric=cosine).", name, vector_size)


def stable_id(source: str, chunk_index: int, text_prefix: str) -> str:
    """Deterministic UUID v5 — re-ingestion is idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID,
                          f"{source}::{chunk_index}::{text_prefix[:64]}"))


# ── Argument Parser ───────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant via Ollama embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--qdrant-url",
                        default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL))
    parser.add_argument("--ollama-url",
                        default=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--collection",
                        default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama embedding model name.")
    parser.add_argument("--knowledge-base", type=Path, default=KNOWLEDGE_BASE_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = build_arg_parser().parse_args()

    kb_dir: Path = args.knowledge_base.resolve()
    if not kb_dir.is_dir():
        log.error("Knowledge base directory not found: %s", kb_dir)
        sys.exit(1)

    files = [
        f for f in kb_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        log.warning("No supported documents found in %s.", kb_dir)
        sys.exit(0)
    log.info("Found %d document(s) to ingest.", len(files))

    # ── Validate Ollama connection and model
    log.info("Probing Ollama at %s — model: %s…", args.ollama_url, args.model)
    try:
        actual_dim = probe_dimension(args.model, args.ollama_url)
    except Exception as exc:
        log.error("Cannot reach Ollama: %s", exc)
        log.error("Is rag_ollama running? Try: docker compose up -d ollama")
        sys.exit(1)

    if actual_dim != VECTOR_DIM:
        log.warning("Model returned %d dims; expected %d.", actual_dim, VECTOR_DIM)
    log.info("Embedding backend: Ollama (%s) — %d dims", args.model, actual_dim)

    # ── Qdrant connection
    log.info("Connecting to Qdrant at %s…", args.qdrant_url)
    client = QdrantClient(url=args.qdrant_url)
    ensure_collection(client, args.collection, vector_size=actual_dim)

    # ── Process files
    total_chunks = 0

    for file_path in files:
        log.info("Processing: %s", file_path.relative_to(kb_dir.parent))
        file_chunks = 0
        global_idx = 0

        if file_path.suffix.lower() == ".pdf":
            # Page-by-page to keep peak RAM bounded on large PDFs.
            reader = pypdf.PdfReader(str(file_path))
            total_pages = len(reader.pages)
            log.info("  PDF — %d page(s), batch size %d.", total_pages, BATCH_SIZE)

            pending: list[tuple[int, str]] = []

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                for chunk in chunk_text(page_text, args.chunk_size, args.chunk_overlap):
                    pending.append((global_idx, chunk))
                    global_idx += 1

                    if len(pending) >= BATCH_SIZE:
                        idxs, batch = zip(*pending)
                        vectors = embed_texts(list(batch), args.model, args.ollama_url)
                        points = [
                            PointStruct(
                                id=stable_id(file_path.name, idx, ch),
                                vector=vec,
                                payload={"source": file_path.name,
                                         "chunk_index": idx, "text": ch},
                            )
                            for idx, ch, vec in zip(idxs, batch, vectors)
                        ]
                        client.upsert(collection_name=args.collection, points=points)
                        file_chunks += len(points)
                        log.info("  Page %d/%d — %d points (%d total).",
                                 page_num, total_pages, len(points), file_chunks)
                        pending = []

            if pending:
                idxs, batch = zip(*pending)
                vectors = embed_texts(list(batch), args.model, args.ollama_url)
                points = [
                    PointStruct(
                        id=stable_id(file_path.name, idx, ch),
                        vector=vec,
                        payload={"source": file_path.name,
                                 "chunk_index": idx, "text": ch},
                    )
                    for idx, ch, vec in zip(idxs, batch, vectors)
                ]
                client.upsert(collection_name=args.collection, points=points)
                file_chunks += len(points)
                log.info("  Final batch upserted (%d points).", len(points))

        else:
            text = extract_text(file_path)
            if not text.strip():
                log.warning("  No text extracted — skipping.")
                continue
            chunks = list(chunk_text(text, args.chunk_size, args.chunk_overlap))
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i: i + BATCH_SIZE]
                vectors = embed_texts(batch, args.model, args.ollama_url)
                points = [
                    PointStruct(
                        id=stable_id(file_path.name, i + j, ch),
                        vector=vec,
                        payload={"source": file_path.name,
                                 "chunk_index": i + j, "text": ch},
                    )
                    for j, (ch, vec) in enumerate(zip(batch, vectors))
                ]
                client.upsert(collection_name=args.collection, points=points)
                file_chunks += len(points)

        log.info("  File complete — %d chunk(s) total.", file_chunks)
        total_chunks += file_chunks

    log.info("Ingestion complete. %d total chunk(s) upserted into '%s'.",
             total_chunks, args.collection)


if __name__ == "__main__":
    main()
