"""
Production-Grade RAG System — FastAPI Backend
Handles document ingestion, retrieval-augmented QA, and S3 file management.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.rag_pipeline import RAGPipeline
from app.utils import setup_logging, save_upload_file_tmp
from app.s3_utils import upload_file_to_s3, download_file_from_s3, list_files
from app.config import get_settings

# ─── Logging ─────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

# ─── Global RAG Pipeline ─────────────────────────────────────────────────────
rag: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize & teardown the RAG pipeline."""
    global rag
    logger.info("Starting up — loading RAG pipeline …")
    rag = RAGPipeline()
    rag.load_existing_vectorstore()  # loads persisted index if present
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down RAG pipeline.")


# ─── App ─────────────────────────────────────────────────────────────────────
cfg = get_settings()

app = FastAPI(
    title="Production-Grade RAG API",
    description=(
        "LLMOps RAG system powered by LangChain, FAISS & Sentence-Transformers.\n\n"
        "**Endpoints:**\n"
        "- `GET /ask?query=…` — Query the RAG pipeline (convenience)\n"
        "- `POST /ask` — Query the RAG pipeline (JSON body)\n"
        "- `POST /upload` — Upload a document and index it into the vector store\n"
        "- `GET /files` — List documents stored in S3\n"
        "- `GET /documents` — List documents indexed in the local vector store\n"
        "- `GET /health` — Health check\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    source_documents: list[str] = []


# ─── Health ──────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="Health check")
def health_check():
    """Returns service health and whether the vector store is loaded."""
    return {
        "status": "ok",
        "vectorstore_loaded": rag is not None and rag.is_ready(),
        "s3_enabled": cfg.s3_enabled,
    }


# ─── Upload Documents ────────────────────────────────────────────────────────
@app.post("/upload", tags=["Documents"], summary="Upload & index a document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF, TXT, or DOCX file.

    The file is:
    1. Saved temporarily on disk.
    2. Uploaded to S3 (if configured).
    3. Chunked, embedded, and indexed into FAISS (background task).
    """
    allowed = {".pdf", ".txt", ".docx"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    tmp_path = await save_upload_file_tmp(file)

    # S3 upload (non-blocking — if it fails, we still index locally)
    s3_key = upload_file_to_s3(tmp_path, file.filename)

    # Index in background so the HTTP response is fast
    background_tasks.add_task(_index_document, tmp_path, file.filename)
    logger.info("Queued '%s' for indexing (s3_key=%r).", file.filename, s3_key)

    return {
        "message": f"'{file.filename}' accepted for indexing.",
        "status": "processing",
        "s3_key": s3_key or None,
    }


def _index_document(tmp_path: str, filename: str):
    """Background task: load, chunk & embed document into FAISS."""
    try:
        rag.ingest(tmp_path, filename)
        logger.info("Successfully indexed '%s'.", filename)
    except Exception as exc:
        logger.exception("Failed to index '%s': %s", filename, exc)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── Ask (POST) ───────────────────────────────────────────────────────────────
@app.post("/ask", response_model=QueryResponse, tags=["QA"], summary="Query the RAG pipeline")
def ask(request: QueryRequest):
    """Submit a question via JSON body. Returns the LLM-generated answer and source snippets."""
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Vector store not yet ready. Please upload documents first.",
        )
    try:
        answer, sources = rag.query(request.query, top_k=request.top_k)
        logger.info("Query answered | query=%r | sources=%d", request.query, len(sources))
        return QueryResponse(answer=answer, source_documents=sources)
    except Exception as exc:
        logger.exception("Error during query: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Ask (GET — convenience) ──────────────────────────────────────────────────
@app.get("/ask", response_model=QueryResponse, tags=["QA"], summary="Query via GET param")
def ask_get(
    query: str = Query(..., description="Your question for the RAG system"),
    top_k: int = Query(4, ge=1, le=20, description="Number of retrieved chunks"),
):
    """GET-based query endpoint — useful for quick browser testing."""
    return ask(QueryRequest(query=query, top_k=top_k))


# ─── List S3 Files ────────────────────────────────────────────────────────────
@app.get("/files", tags=["Documents"], summary="List documents stored in S3")
def list_s3_files():
    """
    Return all document objects stored in the configured S3 bucket.
    Returns an empty list when S3 is not configured.
    """
    files = list_files()
    logger.info("Listed %d files from S3.", len(files))
    return {"files": files, "count": len(files)}


# ─── List Indexed Docs ────────────────────────────────────────────────────────
@app.get("/documents", tags=["Documents"], summary="List locally indexed documents")
def list_documents():
    """List documents currently indexed in the local FAISS vector store."""
    docs = rag.list_ingested_docs() if rag else []
    return {"documents": docs, "count": len(docs)}
