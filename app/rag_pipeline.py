"""
RAG Pipeline — core embedding, indexing & retrieval logic.
Compatible with LangChain 1.x (LCEL-based, no deprecated chains).
"""

import os
import json
import logging
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
DOCS_META_PATH = os.path.join(VECTORSTORE_PATH, "ingested_docs.json")

# Prompt template for RAG chain
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Answer the user's question using ONLY the context "
        "provided below. If the context does not contain enough information to answer, "
        "say 'I don't have enough information to answer that based on the uploaded documents.'\n\n"
        "Context:\n{context}"
    )),
    ("human", "{question}"),
])


class RAGPipeline:
    """Encapsulates the full Retrieve-and-Generate pipeline (LangChain 1.x LCEL)."""

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
        self._chain = None
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
                self._build_chain()
                logger.info(
                    "Loaded existing vectorstore from '%s'.", VECTORSTORE_PATH)
            except Exception as exc:
                logger.warning("Could not load existing vectorstore: %s", exc)

    def ingest(self, file_path: str, filename: str):
        """Load, chunk, embed, and persist a document."""
        docs = self._load_file(file_path)
        chunks = self._splitter.split_documents(docs)
        logger.info(
            "Ingesting '%s': %d raw docs → %d chunks.", filename, len(
                docs), len(chunks)
        )

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._vectorstore.add_documents(chunks)

        self._vectorstore.save_local(VECTORSTORE_PATH)
        self._build_chain()

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
        # Retrieve source docs for the response
        source_docs = retriever.invoke(question)
        sources = [doc.page_content[:300] for doc in source_docs]

        if self._chain is not None:
            answer = self._chain.invoke(
                {"question": question, "context": _format_docs(source_docs)})
        else:
            # Fallback: return the retrieved context directly
            answer = "\n\n---\n\n".join(
                sources) if sources else "No relevant documents found."

        return answer, sources

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_chain(self):
        """(Re-)build the LCEL RAG chain after vectorstore updates."""
        if not OPENAI_API_KEY:
            # No LLM configured — queries will return raw retrieved context
            logger.info("No OPENAI_API_KEY set — using retrieval-only mode.")
            self._chain = None
            return
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=0,
                api_key=OPENAI_API_KEY,
            )
            self._chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | RAG_PROMPT
                | llm
                | StrOutputParser()
            )
            logger.info("LCEL RAG chain built with model '%s'.", LLM_MODEL)
        except Exception as exc:
            logger.warning(
                "Could not build LLM chain: %s — falling back to retrieval-only.", exc)
            self._chain = None

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


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
