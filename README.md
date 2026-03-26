# 🚀 Production-Grade RAG LLMOps Pipeline

A **production-ready** Retrieval-Augmented Generation (RAG) system built with
**LangChain · FAISS · FastAPI · Docker · AWS EC2 · GitHub Actions**.

---

## ✨ Features

| Feature | Details |
|---|---|
| Document ingestion | PDF, TXT, DOCX via `/upload` |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, runs locally — no API key needed) |
| Vector store | FAISS (persisted to disk) |
| LLM | OpenAI GPT (configurable) or built-in fallback |
| REST API | FastAPI with `/ask`, `/upload`, `/documents`, `/health` |
| Docker | Multi-stage build with non-root user & health checks |
| CI/CD | GitHub Actions: lint → build → push → SSH deploy to EC2 |
| Cloud storage | Optional S3 integration for document persistence |

---

## 🗂️ Project Structure

```
rag-llmops/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── rag_pipeline.py   # Core RAG logic (embed, index, retrieve, generate)
│   └── utils.py          # Logging, file helpers, S3 integration
├── data/
│   └── sample.txt        # Sample document for testing
├── tests/
│   ├── __init__.py
│   └── test_api.py       # Pytest suite
├── .github/
│   └── workflows/
│       └── deploy.yml    # CI/CD pipeline
├── vectorstore/          # FAISS index (auto-created, git-ignored)
├── .env.example          # Template — copy to .env and fill in values
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (Local)

### 1 — Clone & configure

```bash
git clone https://github.com/YOUR_USER/rag-llmops.git
cd rag-llmops
cp .env.example .env          # fill in OPENAI_API_KEY (optional) and others
```

### 2 — Create virtualenv & install deps

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3 — Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 4 — Upload a document

```bash
curl -X POST http://localhost:8000/upload \
     -F "file=@data/sample.txt"
```

### 5 — Ask a question

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is RAG and what are its benefits?"}'
```

---

## 🐳 Docker

```bash
# Build & run with Docker Compose (recommended)
docker-compose up --build

# Or plain Docker
docker build -t rag-llmops .
docker run -d -p 8000:8000 --env-file .env rag-llmops
```

---

## ☁️ AWS EC2 Deployment

### 1 — Launch EC2 instance
- AMI: **Ubuntu 22.04 LTS**
- Instance type: `t3.medium` (minimum; use `t3.large` for production)
- Security Group: open port **8000** (and optionally 80 via nginx reverse-proxy)

### 2 — Bootstrap the server

```bash
sudo apt update && sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
```

### 3 — Pull & run

```bash
docker pull ghcr.io/YOUR_GITHUB_USER/rag-llmops:latest
docker run -d --name rag_api --restart unless-stopped \
  -p 8000:8000 \
  -v rag_vectorstore:/app/vectorstore \
  -e OPENAI_API_KEY="sk-..." \
  ghcr.io/YOUR_GITHUB_USER/rag-llmops:latest
```

### 4 — CI/CD (GitHub Actions)

Add these **repository secrets** in GitHub → Settings → Secrets:

| Secret | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI key |
| `EC2_HOST` | Public IP / DNS of your EC2 instance |
| `EC2_USER` | SSH user (usually `ubuntu`) |
| `EC2_SSH_KEY` | Private SSH key (PEM content) |
| `AWS_ACCESS_KEY_ID` | IAM user key (for S3) |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `S3_BUCKET_NAME` | S3 bucket name for document storage |

Push to `main` → GitHub Actions automatically tests, builds the Docker image, pushes to GHCR, and deploys to EC2.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | OpenAI key — leave empty to use fallback |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI model name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `VECTORSTORE_PATH` | `vectorstore` | Directory to persist the FAISS index |
| `CHUNK_SIZE` | `500` | Token chunk size for text splitting |
| `CHUNK_OVERLAP` | `50` | Chunk overlap for text splitting |
| `AWS_ACCESS_KEY_ID` | *(empty)* | IAM key for S3 |
| `AWS_SECRET_ACCESS_KEY` | *(empty)* | IAM secret for S3 |
| `AWS_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET_NAME` | *(empty)* | S3 bucket — leave empty to disable |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload & index a document |
| `POST` | `/ask` | Ask a question (JSON body) |
| `GET` | `/ask?query=...` | Ask a question (URL param) |
| `GET` | `/documents` | List all indexed documents |
| `GET` | `/docs` | Swagger UI |

---

## 🏗️ Architecture

```
User Query
   │
   ▼
FastAPI  (/ask)
   │
   ▼
LangChain RetrievalQA
   │
   ├──[embed query]──► HuggingFace Embeddings (all-MiniLM-L6-v2)
   │
   ├──[vector search]──► FAISS Vector Store
   │                         │
   │                    top-k chunks
   │
   ▼
LLM (GPT-4 / GPT-3.5 / Fallback)
   │
   ▼
Answer + Source Snippets
```

---

## 📜 License

```
MIT License

Copyright (c) 2026 Omkar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

Made with ❤️ by **Omkar** &nbsp;|&nbsp; MIT Licensed &nbsp;|&nbsp; 2026

</div>
