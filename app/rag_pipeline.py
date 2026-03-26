"""
RAG Pipeline — core embedding, indexing & retrieval logic.
"""

import os
import json
import logging
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
DOCS_META_PATH = os.path.join(VECTORSTORE_PATH, "ingested_docs.json")


class RAGPipeline:
    """Encapsulates the full Retrieve-and-Generate pipeline."""

    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
        )
        self._vectorstore: Optional[FAISS] = None
        self._qa_chain = None
        self._ingested_docs: list[str] = []
        self._load_docs_meta()

    # ── Persistence helpers ──────────────────────────────────────────────────

    def _load_docs_meta(self):
        if os.path.exists(DOCS_META_PATH):
            with open(DOCS_META_PATH) as f:
                self._ingested_docs = json.load(f)

    def _save_docs_meta(self):
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        with open(DOCS_META_PATH, "w") as f:
            json.dump(self._ingested_docs, f, indent=2)

    # ── Public interface ─────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._vectorstore is not None

    def list_ingested_docs(self) -> list[str]:
        return self._ingested_docs

    def load_existing_vectorstore(self):
        """Load a previously persisted FAISS index (if it exists)."""
        index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
        if os.path.exists(index_file):
            try:
                self._vectorstore = FAISS.load_local(
                    VECTORSTORE_PATH,
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._build_qa_chain()
                logger.info("Loaded existing vectorstore from '%s'.", VECTORSTORE_PATH)
            except Exception as exc:
                logger.warning("Could not load existing vectorstore: %s", exc)

    def ingest(self, file_path: str, filename: str):
        """Load, chunk, embed, and persist a document."""
        docs = self._load_file(file_path)
        chunks = self._splitter.split_documents(docs)
        logger.info(
            "Ingesting '%s': %d raw docs → %d chunks.", filename, len(docs), len(chunks)
        )

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._vectorstore.add_documents(chunks)

        self._vectorstore.save_local(VECTORSTORE_PATH)
        self._build_qa_chain()

        if filename not in self._ingested_docs:
            self._ingested_docs.append(filename)
        self._save_docs_meta()
        logger.info("Indexing of '%s' complete.", filename)

    def query(self, question: str, top_k: int = 4) -> tuple[str, list[str]]:
        """Run a RAG query; returns (answer, [source page contents])."""
        if not self.is_ready():
            raise RuntimeError("Vector store not initialised.")

        retriever = self._vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        result = self._qa_chain({"query": question})
        answer = result.get("result", "")

        source_docs = result.get("source_documents", [])
        sources = [doc.page_content[:300] for doc in source_docs]
        return answer, sources

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_qa_chain(self):
        """(Re-)build the RetrievalQA chain after vectorstore updates."""
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})

        if OPENAI_API_KEY:
            llm = ChatOpenAI(
                model_name=LLM_MODEL,
                temperature=0,
                openai_api_key=OPENAI_API_KEY,
            )
        else:
            # Fallback: use a simple prompt-based retrieval without LLM
            # (returns top-k chunks only – useful for dev without API key)
            llm = _FallbackLLM()

        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

    @staticmethod
    def _load_file(file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path).load()
        elif ext == ".txt":
            return TextLoader(file_path, encoding="utf-8").load()
        elif ext == ".docx":
            return Docx2txtLoader(file_path).load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")


# ──────────────────────────────────────────────────────────────────────────────
# Fallback "LLM" for when no OpenAI key is configured
# ──────────────────────────────────────────────────────────────────────────────
from langchain.llms.base import LLM
from typing import Any


class _FallbackLLM(LLM):
    """Returns the retrieved context directly (no actual LLM call)."""

    @property
    def _llm_type(self) -> str:
        return "fallback"

    def _call(self, prompt: str, stop: Any = None, **kwargs) -> str:
        # strip preamble and return the embedded context
        marker = "Context:"
        if marker in prompt:
            return prompt.split(marker, 1)[-1].strip()
        return prompt
