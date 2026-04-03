#!/usr/bin/env python3
"""
ingest.py — Document ingestion pipeline for the RAG system.

Reads documents from knowledge_base/, chunks them, embeds via FastEmbed,
and upserts vectors into Qdrant.

Execution provider priority:  CoreML (Apple MPS) → CUDA (NVIDIA) → CPU
Qdrant URL is configurable via --qdrant-url flag or QDRANT_URL env var.
"""

import os
import sys
import uuid
import logging
import argparse
from pathlib import Path
from typing import Generator

import onnxruntime as ort
from fastembed import TextEmbedding
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
# Use Path(__file__) so all paths resolve correctly on Mac, Linux, and Windows.
KNOWLEDGE_BASE_DIR: Path = Path(__file__).parent / "knowledge_base"
DEFAULT_QDRANT_URL: str   = "http://localhost:6333"
DEFAULT_COLLECTION: str   = "rag_documents"
DEFAULT_MODEL: str        = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE: int           = 512   # characters per chunk
CHUNK_OVERLAP: int        = 64    # overlap between consecutive chunks

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".rst"})


# ── Provider Detection ────────────────────────────────────────────────────────
def detect_providers() -> list[str]:
    """
    Detect the best available ONNX Runtime execution provider.

    Priority:
        1. CoreMLExecutionProvider  — Apple Silicon / MPS (macOS)
        2. CUDAExecutionProvider    — NVIDIA GPU (Linux / Windows)
        3. CPUExecutionProvider     — universal fallback

    CPUExecutionProvider is always appended as the final fallback so
    ONNX Runtime never fails to find a valid backend.
    """
    available: list[str] = ort.get_available_providers()
    log.debug("Available ORT providers: %s", available)

    providers: list[str] = []

    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
        log.info("Execution backend: CoreML (Apple MPS / Neural Engine)")
    elif "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        log.info("Execution backend: CUDA (NVIDIA GPU)")
    else:
        log.info("Execution backend: CPU (no GPU provider found)")

    providers.append("CPUExecutionProvider")
    return providers


# ── Text Extraction ───────────────────────────────────────────────────────────
def _extract_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text(path: Path) -> str:
    """Return plain text from a supported file. Returns empty string on failure."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix in {".txt", ".md", ".rst"}:
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
    """
    Yield overlapping character-level chunks from *text*.
    Strips leading/trailing whitespace from each chunk.
    """
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
def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
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
    """
    Return a deterministic UUID v5 string for a chunk.
    Using UUID5 means re-running ingest on the same file produces identical
    IDs, so Qdrant's upsert will overwrite rather than duplicate.
    """
    name = f"{source}::{chunk_index}::{text_prefix[:64]}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))


# ── Main ──────────────────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest documents from knowledge_base/ into Qdrant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL),
        help="Qdrant REST endpoint. Override via QDRANT_URL env var.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
        help="Qdrant collection name. Override via QDRANT_COLLECTION env var.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="FastEmbed model name.",
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=KNOWLEDGE_BASE_DIR,
        help="Path to the directory containing documents to ingest.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Character overlap between consecutive chunks.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # Resolve knowledge base path (works on Mac, Linux, Windows)
    kb_dir: Path = args.knowledge_base.resolve()
    if not kb_dir.is_dir():
        log.error("Knowledge base directory not found: %s", kb_dir)
        sys.exit(1)

    files = [
        f for f in kb_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        log.warning("No supported documents found in %s. Supported: %s",
                    kb_dir, ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        sys.exit(0)

    log.info("Found %d document(s) to ingest.", len(files))

    # ── Embedding model
    providers = detect_providers()
    log.info("Loading embedding model '%s'…", args.model)
    embedder = TextEmbedding(model_name=args.model, providers=providers)

    # Probe vector dimensionality
    probe_vector = next(iter(embedder.embed(["__probe__"])))
    vector_size = len(probe_vector)
    log.info("Vector dimensionality: %d", vector_size)

    # ── Qdrant connection
    log.info("Connecting to Qdrant at %s…", args.qdrant_url)
    client = QdrantClient(url=args.qdrant_url)
    ensure_collection(client, args.collection, vector_size)

    # ── Process files
    total_chunks = 0

    for file_path in files:
        relative = file_path.relative_to(kb_dir.parent)
        log.info("Processing: %s", relative)

        text = extract_text(file_path)
        if not text.strip():
            log.warning("  No text extracted from '%s' — skipping.", file_path.name)
            continue

        chunks = list(chunk_text(text, args.chunk_size, args.chunk_overlap))
        vectors = list(embedder.embed(chunks))

        points = [
            PointStruct(
                id=stable_id(file_path.name, idx, chunk),
                vector=vector.tolist(),
                payload={
                    "source":      file_path.name,
                    "chunk_index": idx,
                    "text":        chunk,
                },
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        client.upsert(collection_name=args.collection, points=points)
        log.info("  Upserted %d chunk(s).", len(points))
        total_chunks += len(points)

    log.info("Ingestion complete. %d total chunk(s) upserted into '%s'.",
             total_chunks, args.collection)


if __name__ == "__main__":
    main()
