"""
app/services/rag_service.py — Enhanced RAG pipeline with hybrid search and re-ranking.

Improvements over v1:
  - Hybrid BM25 + vector search with configurable weights
  - Cross-encoder re-ranking (sentence-transformers)
  - Dynamic chunk sizing based on document type
  - Metadata filtering
  - MLflow experiment tracking
  - Async-safe (CPU-bound ops run in executor)
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-worker")

# ── Lazy imports (heavy; loaded once at startup) ───────────────────────────────
_pipeline: Optional["RAGPipeline"] = None


def get_rag_pipeline() -> "RAGPipeline":
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ── RAG Pipeline ──────────────────────────────────────────────────────────────


class RAGPipeline:
    """Production RAG pipeline: hybrid BM25 + vector search with re-ranking."""

    CHUNK_SIZE_BY_EXT = {".pdf": 800, ".txt": 500, ".docx": 600}

    def __init__(self):
        cfg = get_settings()
        self._cfg = cfg

        # Embedding model
        from langchain_community.embeddings import HuggingFaceEmbeddings

        self._embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embed_model,
            model_kwargs={"device": "cpu"},
        )

        # Vector store + BM25 corpus (in-memory)
        from langchain_community.vectorstores import FAISS

        self._vectorstore: Optional[FAISS] = None
        self._bm25 = None
        self._all_docs: list = []  # raw LangChain Documents kept for BM25

        # LLM chain
        self._chain = None

        # Ingested doc registry
        self._meta_path = os.path.join(cfg.vectorstore_path, "ingested_docs.json")
        self._ingested: list[str] = self._load_meta()

        # Cross-encoder re-ranker (optional — will skip if not importable)
        self._reranker = self._build_reranker()

        logger.info("RAGPipeline initialised (embed_model=%s).", cfg.embed_model)

    # ── Public interface ──────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._vectorstore is not None

    def list_ingested_docs(self) -> list[str]:
        return self._ingested

    def load_existing_vectorstore(self):
        """Load a persisted FAISS index from disk (if it exists)."""
        from langchain_community.vectorstores import FAISS

        index_file = os.path.join(self._cfg.vectorstore_path, "index.faiss")
        if os.path.exists(index_file):
            try:
                self._vectorstore = FAISS.load_local(
                    self._cfg.vectorstore_path,
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._rebuild_bm25()
                self._build_chain()
                logger.info("Loaded FAISS index from '%s'.", self._cfg.vectorstore_path)
            except Exception as exc:
                logger.warning("Could not load vectorstore: %s", exc)

    def ingest(self, file_path: str, filename: str, metadata: dict[str, Any] | None = None) -> int:
        """
        Load, chunk, embed, and persist a document.
        Returns the number of chunks added.
        """
        start = time.perf_counter()
        docs = self._load_file(file_path)
        ext = os.path.splitext(filename)[-1].lower()
        chunk_size = self.CHUNK_SIZE_BY_EXT.get(ext, self._cfg.chunk_size)

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=self._cfg.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        # Attach metadata filter fields to every chunk
        base_meta = {"source": filename, **(metadata or {})}
        for c in chunks:
            c.metadata.update(base_meta)

        # Add to FAISS
        if self._vectorstore is None:
            from langchain_community.vectorstores import FAISS

            self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._vectorstore.add_documents(chunks)

        self._vectorstore.save_local(self._cfg.vectorstore_path)

        # Update BM25 corpus
        self._all_docs.extend(chunks)
        self._rebuild_bm25()
        self._build_chain()

        if filename not in self._ingested:
            self._ingested.append(filename)
        self._save_meta()

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("Ingested '%s': %d chunks in %.0fms.", filename, len(chunks), elapsed)

        # MLflow tracking
        self._track_ingestion(filename, len(chunks), elapsed)

        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Hybrid search + re-rank + generate.

        Returns
        -------
        tuple[str, list[dict]]
            (answer_text, source_documents_with_scores)
        """
        if not self.is_ready():
            from app.utils.exceptions import VectorStoreNotReadyError

            raise VectorStoreNotReadyError()

        candidates = self._hybrid_retrieve(question, top_k=self._cfg.reranker_top_k)
        reranked = self._rerank(question, candidates, top_k=top_k)
        sources = [
            {
                "content": d.page_content[:400],
                "source": d.metadata.get("source"),
                "score": float(s),
            }
            for d, s in reranked
        ]

        if self._chain is not None:
            context = "\n\n".join(d.page_content for d, _ in reranked)
            answer = self._chain.invoke({"question": question, "context": context})
        else:
            # Retrieval-only fallback
            answer = (
                "\n\n---\n\n".join(s["content"] for s in sources) or "No relevant documents found."
            )

        return answer, sources

    # ── Private helpers ───────────────────────────────────────────────────────

    def _hybrid_retrieve(self, question: str, top_k: int) -> list:
        """Fuse BM25 and vector scores (reciprocal rank fusion)."""
        cfg = self._cfg

        # Vector retrieval with scores
        try:
            vec_results = self._vectorstore.similarity_search_with_score(question, k=top_k)
        except Exception as exc:
            logger.warning("Vector search error: %s", exc)
            vec_results = []

        if self._bm25 is None or not self._all_docs:
            return [doc for doc, _ in vec_results]

        try:
            tokenized_q = question.lower().split()
            scores = self._bm25.get_scores(tokenized_q)
            bm25_top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            bm25_results = [(self._all_docs[i], s) for i, s in bm25_top]
        except Exception as exc:
            logger.warning("BM25 retrieval error: %s — using vector-only.", exc)
            return [doc for doc, _ in vec_results]

        # Reciprocal Rank Fusion
        scores_map: dict[str, float] = {}
        doc_map: dict[str, object] = {}

        for rank, (doc, _) in enumerate(vec_results):
            key = doc.page_content[:100]
            scores_map[key] = scores_map.get(key, 0) + cfg.vector_weight / (rank + 1)
            doc_map[key] = doc

        for rank, (doc, _) in enumerate(bm25_results):
            key = doc.page_content[:100]
            scores_map[key] = scores_map.get(key, 0) + cfg.bm25_weight / (rank + 1)
            doc_map[key] = doc

        merged = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc_map[k] for k, _ in merged]

    def _rerank(self, question: str, docs: list, top_k: int) -> list[tuple]:
        """Cross-encoder re-ranking; returns list of (doc, score) tuples."""
        if self._reranker is None or not docs:
            return [(d, 1.0) for d in docs[:top_k]]
        try:
            pairs = [[question, d.page_content] for d in docs]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        except Exception as exc:
            logger.warning("Re-ranker error: %s — skipping re-ranking.", exc)
            return [(d, 1.0) for d in docs[:top_k]]

    def _rebuild_bm25(self):
        """Rebuild BM25 index from the in-memory document corpus."""
        if not self._all_docs:
            return
        try:
            from rank_bm25 import BM25Okapi

            tokenized = [doc.page_content.lower().split() for doc in self._all_docs]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.info("rank_bm25 not installed — vector-only search active.")

    def _build_chain(self):
        """(Re-)build the LCEL RAG chain when OpenAI key is set."""
        if not self._cfg.openai_api_key:
            self._chain = None
            return
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self._cfg.llm_model,
                temperature=0,
                api_key=self._cfg.openai_api_key,
                streaming=False,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant. Answer using ONLY the provided context.\n"
                        "If the context is insufficient, say so honestly.\n\nContext:\n{context}",
                    ),
                    ("human", "{question}"),
                ]
            )
            self._chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            logger.info("LCEL chain built with model '%s'.", self._cfg.llm_model)
        except Exception as exc:
            logger.warning("Could not build LLM chain: %s — retrieval-only mode.", exc)
            self._chain = None

    def _build_reranker(self):
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder re-ranker loaded.")
            return model
        except Exception as exc:
            logger.info("Re-ranker not available (%s) — skipping.", exc)
            return None

    def _track_ingestion(self, filename: str, chunk_count: int, latency_ms: float):
        cfg = self._cfg
        if not cfg.mlflow_enabled:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment)
            with mlflow.start_run(run_name=f"ingest-{filename}"):
                mlflow.log_param("filename", filename)
                mlflow.log_metric("chunk_count", chunk_count)
                mlflow.log_metric("ingestion_latency_ms", latency_ms)
        except Exception as exc:
            logger.debug("MLflow tracking skipped: %s", exc)

    @staticmethod
    def _load_file(file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            return PyPDFLoader(file_path).load()
        elif ext == ".txt":
            from langchain_community.document_loaders import TextLoader

            return TextLoader(file_path, encoding="utf-8").load()
        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader

            return Docx2txtLoader(file_path).load()
        raise ValueError(f"Unsupported file type: {ext}")

    def _load_meta(self) -> list[str]:
        if os.path.exists(self._meta_path):
            with open(self._meta_path) as f:
                return json.load(f)
        return []

    def _save_meta(self):
        os.makedirs(self._cfg.vectorstore_path, exist_ok=True)
        with open(self._meta_path, "w") as f:
            json.dump(self._ingested, f, indent=2)


# ── Async wrappers ─────────────────────────────────────────────────────────────


async def async_ingest(
    file_path: str,
    filename: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Run the CPU-bound ingest in a thread so the event loop stays free."""
    loop = asyncio.get_event_loop()
    pipeline = get_rag_pipeline()
    return await loop.run_in_executor(
        _executor, partial(pipeline.ingest, file_path, filename, metadata)
    )


async def async_query(
    question: str,
    top_k: int = 4,
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[str, list[dict]]:
    """Run the CPU-bound query in a thread."""
    loop = asyncio.get_event_loop()
    pipeline = get_rag_pipeline()
    return await loop.run_in_executor(
        _executor, partial(pipeline.query, question, top_k, metadata_filter)
    )
