"""
Production-Grade RAG System - FastAPI Backend
Handles document ingestion and retrieval-augmented question answering.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.rag_pipeline import RAGPipeline
from app.utils import setup_logging, save_upload_file_tmp

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
app = FastAPI(
    title="Production-Grade RAG API",
    description="LLMOps RAG system powered by LangChain, FAISS & Sentence-Transformers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "vectorstore_loaded": rag is not None and rag.is_ready()}


# ─── Upload Documents ────────────────────────────────────────────────────────
@app.post("/upload", tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF, TXT, or DOCX file and index it into the vector store.
    Heavy processing is done in a background task so the response is fast.
    """
    allowed = {".pdf", ".txt", ".docx"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    tmp_path = await save_upload_file_tmp(file)
    background_tasks.add_task(_index_document, tmp_path, file.filename)
    logger.info("Queued '%s' for indexing.", file.filename)
    return {"message": f"'{file.filename}' accepted for indexing.", "status": "processing"}


def _index_document(tmp_path: str, filename: str):
    """Background task: load, chunk & embed document."""
    try:
        rag.ingest(tmp_path, filename)
        logger.info("Successfully indexed '%s'.", filename)
    except Exception as exc:
        logger.exception("Failed to index '%s': %s", filename, exc)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── Ask ─────────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=QueryResponse, tags=["QA"])
def ask(request: QueryRequest):
    """Query the RAG pipeline and return the LLM-generated answer."""
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Vector store not yet ready. Please upload documents first.",
        )
    try:
        answer, sources = rag.query(request.query, top_k=request.top_k)
        return QueryResponse(answer=answer, source_documents=sources)
    except Exception as exc:
        logger.exception("Error during query: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── GET convenience endpoint ─────────────────────────────────────────────────
@app.get("/ask", tags=["QA"])
def ask_get(query: str, top_k: int = 4):
    """GET-based query endpoint (convenience)."""
    return ask(QueryRequest(query=query, top_k=top_k))


# ─── List indexed docs ────────────────────────────────────────────────────────
@app.get("/documents", tags=["Documents"])
def list_documents():
    """List all documents currently indexed in the vector store."""
    if not rag:
        return {"documents": []}
    return {"documents": rag.list_ingested_docs()}
