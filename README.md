# RAG Knowledge Assistant API

Production-style retrieval-augmented generation backend with vector search, structured outputs, and hallucination guardrails.

**Stack:** FastAPI · PostgreSQL + pgvector · Redis · sentence-transformers · Groq (OpenAI-compatible) · Pydantic v2 · Docker

## What it does

The API turns source documents into searchable knowledge and answers questions against that knowledge. Document embedding runs through FastAPI background tasks, so large ingestion jobs leave the HTTP response cycle immediately:

`ingest -> chunk -> embed -> store vectors -> semantic retrieve -> guarded LLM answer with citations`

```mermaid
flowchart LR
    A[Document API] --> B[Chunk with overlap]
    B --> C[Local sentence-transformer]
    C --> D[(PostgreSQL + pgvector)]
    Q[Question API] --> E[Embed query]
    E --> F[Cosine similarity search]
    D --> F
    F --> G[Similarity threshold]
    G --> H[Groq structured output]
    H --> I[Pydantic response + server-side source IDs]
```

## Features

- Free local embeddings with `all-MiniLM-L6-v2` (384 dimensions; no embedding API cost).
- Hybrid retrieval combining cosine-similarity search in pgvector with PostgreSQL full-text keyword matching.
- Exact keyword matching for entities such as error codes and SKUs alongside semantic retrieval.
- Local CrossEncoder reranking of 15 retrieved candidates before the LLM sees the top 3.
- Similarity-threshold filtering to reject weak context.
- Pydantic-validated JSON responses.
- Explicit hallucination flag for unsupported answers.
- Source document IDs injected server-side, never trusted from the LLM.
- Deterministic `temperature=0` generation.
- Background-task ingestion with a status endpoint for long-running embedding workloads.
- Redis semantic caching with cosine similarity to bypass duplicate LLM calls.

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/documents` | Store a source document |
| GET | `/documents` | List source documents |
| POST | `/documents/{id}/process` | Queue chunking, embedding, and persistence |
| GET | `/documents/{id}/status` | Check whether processing is `pending`, `processing`, `completed`, or `failed` |
| POST | `/search` | Retrieve hybrid semantic and keyword matches |
| POST | `/ask` | Retrieve context, use the semantic cache, and generate a guarded answer |

Example `/ask` request:

```json
{
  "query": "What is the return policy?"
}
```

Example response:

```json
{
  "answer": "The return policy allows returns within 30 days.",
  "is_hallucination": false,
  "source_document_ids": [7]
}
```

## Quickstart

Prerequisites: Python 3.11+ and Docker Desktop.

From PowerShell:

```powershell
docker compose up -d
py -3.11 -m venv fastapi-rag\venv
fastapi-rag\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example fastapi-rag\.env
notepad fastapi-rag\.env
Set-Location fastapi-rag
uvicorn main:app --reload
```

Add `GROQ_API_KEY` and `DATABASE_URL` to `fastapi-rag\.env`, then open <http://127.0.0.1:8000/docs>.
Redis runs locally at `redis://localhost:6379/0` through Docker Compose.

Run the test suite from the repository root:

```powershell
pip install -r requirements-dev.txt
pytest
```

## Project structure

```text
.
|-- fastapi-rag/
|   |-- main.py         # FastAPI routes and request schemas
|   |-- database.py     # SQLAlchemy engine, sessions, and metadata
|   |-- models.py       # Document and vector chunk models
|   |-- chunking.py     # Overlapping text chunking
|   |-- embeddings.py   # Local sentence-transformer embeddings
|   |-- llm.py          # Groq structured generation and guardrails
|   |-- reranker.py     # Local CrossEncoder relevance reranking
|   |-- semantic_cache.py # Redis vector-similarity response cache
|   `-- .env            # Local secrets; ignored by Git
|-- tests/              # DB- and API-key-free automated tests
|-- docker-compose.yml  # Persistent pgvector service
|-- requirements.txt    # Runtime dependencies
|-- requirements-dev.txt# Test dependencies
|-- .env.example        # Safe environment template
`-- LICENSE             # MIT license
```

## Design decisions

- **Chunking with overlap:** preserves context across boundaries so a fact split between chunks remains retrievable.
- **Local embeddings:** removes embedding API cost and keeps ingestion data local while retaining a compact 384-dimensional index.
- **Similarity threshold:** prevents unrelated nearest neighbors from being presented as evidence.
- **Server-side source IDs:** makes citations authoritative by deriving them from retrieved database rows instead of model output.
- **Temperature 0:** favors repeatable answers and makes evaluation and debugging easier.
- **Background ingestion:** moves chunking and embedding out of the request/response cycle to prevent long-document HTTP timeouts and keep the API responsive to concurrent requests.
- **Persistent job status:** stores ingestion state on the document row so status remains consistent across workers and process restarts; failures are recorded as `failed`.
- **Hybrid search:** combines dense embeddings for semantic meaning with sparse PostgreSQL full-text matching for exact entities such as `ERR-4042`.
- **Reranking:** expands `/ask` retrieval to 15 candidates, then uses a local CrossEncoder to select the 3 most relevant context chunks for generation.
- **Semantic caching:** stores query embeddings and validated answers in Redis for one hour; a cosine similarity above `0.95` returns the cached answer without calling Groq.

## Roadmap

- Next.js chat UI
- Hybrid search (BM25 + vector)
- Reranking
- Golden evaluation dataset
- Multi-tenant metadata filters
- LangGraph agent mode

## License

MIT. See [LICENSE](LICENSE).