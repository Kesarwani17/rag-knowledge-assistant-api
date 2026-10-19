# RAG Knowledge Assistant API

Production-style retrieval-augmented generation backend with vector search, structured outputs, and hallucination guardrails.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-vector%20search-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Redis](https://img.shields.io/badge/Redis-semantic%20cache-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-local%20services-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-structured%20outputs-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Pytest](https://img.shields.io/badge/Pytest-13%20tests-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-local%20models-FCC624?logo=huggingface&logoColor=black)](https://www.sbert.net/)
[![Jev](https://img.shields.io/badge/Jev-optional%20answerability%20guard-7B61FF?logo=shield&logoColor=white)](fastapi-rag/jev_guard.py)
[![Tests](https://github.com/Kesarwani17/rag-knowledge-assistant-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Kesarwani17/rag-knowledge-assistant-api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Stack:** FastAPI · PostgreSQL + pgvector · Redis · sentence-transformers · Groq (OpenAI-compatible) · Pydantic v2 · Docker · bcrypt · structlog

This repository is a focused backend reference implementation for grounded question answering: documents are ingested into a tenant-scoped knowledge base, retrieved with both semantic and keyword search, reranked locally, and answered with structured LLM output that carries server-verified citations.

<table>
  <tr>
    <td><strong>Knowledge base</strong><br>Tenant-scoped documents with vector and full-text search</td>
    <td><strong>Grounded answers</strong><br>Structured responses with server-verified citations</td>
    <td><strong>Operations</strong><br>Docker-backed PostgreSQL and Redis services</td>
  </tr>
</table>

> **Pipeline:** ingest, retrieve, rerank, optionally verify with Jev, then generate.
> The Jev guard is disabled by default and fails open when enabled but unavailable.

<table>
  <tr>
    <td><strong>API engineering</strong><br>FastAPI routes, Pydantic schemas, background tasks, and OpenAPI docs</td>
    <td><strong>Search and RAG</strong><br>Hybrid retrieval, pgvector, full-text search, local embeddings, and reranking</td>
    <td><strong>Reliability</strong><br>Tenant isolation, semantic caching, validation, citations, and offline tests</td>
  </tr>
</table>

> **Optional guardrail — Jev System-1:** a typed, confidence-aware answerability check that can refuse out-of-scope questions before LLM generation. It is disabled by default and fails open when enabled but unavailable.

## What it does

The API turns tenant-scoped source documents into searchable knowledge and answers questions against that knowledge. Document embedding runs through FastAPI background tasks, so large ingestion jobs leave the HTTP response cycle immediately:

`ingest -> chunk -> embed + full-text vector -> store -> hybrid retrieve -> rerank -> semantic cache or guarded LLM answer`

The design keeps expensive work explicit: ingestion is asynchronous, retrieval combines dense and sparse signals, reranking narrows context before generation, and Redis can bypass repeated LLM inference.

The implementation keeps the RAG flow observable while enforcing tenant boundaries at the API and database query layers.

```mermaid
flowchart TB
  subgraph INGEST[Document ingestion]
    A[POST /documents] --> B[Background task]
    B --> C[Chunk with overlap]
    C --> D[Local sentence-transformer]
    C --> E[PostgreSQL TSVECTOR]
    D --> F[(PostgreSQL + pgvector)]
    E --> F
    F --> G[Status: completed or failed]
  end

  subgraph ANSWER[Question answering]
    H[POST /ask] --> I[Embed query]
    I --> J[Hybrid retrieval]
    F --> J
    J --> K[Cosine + GIN keyword search]
    K --> L[Similarity threshold]
    L --> M[CrossEncoder reranker]
    M --> N[(Redis semantic cache)]
    N -- cache hit --> O[Validated cached response]
    N -- cache miss --> P{Jev enabled?}
    P -- no or fail-open --> Q[Groq structured output]
    P -- high-confidence refusal --> R[Grounded refusal]
    P -- yes --> Q
    Q --> S[Server-side source IDs]
    O --> S
  end
```

## Features

- Free local embeddings with `all-MiniLM-L6-v2` (384 dimensions; no embedding API cost).
- Hybrid retrieval combining cosine-similarity search in pgvector with PostgreSQL full-text keyword matching.
- Exact keyword matching for entities such as error codes and SKUs alongside semantic retrieval.
- Local CrossEncoder reranking of retrieved candidates before the LLM sees the top results (configurable via `RERANK_TOP_K`).
- Configurable similarity-threshold filtering to reject weak context (`SIMILARITY_THRESHOLD`).
- Pydantic-validated JSON responses with explicit request/response models.
- Bounded request inputs at the API boundary (configurable `MAX_REQUEST_SIZE`).
- Explicit hallucination flag for unsupported answers.
- Source document IDs injected server-side, never trusted from the LLM.
- Deterministic `temperature=0` generation.
- Background-task ingestion with a status endpoint for long-running embedding workloads.
- Tenant-scoped document update and delete; updates clear old chunks and return the document to `pending` for reprocessing.
- Redis semantic caching with cosine similarity to bypass duplicate LLM calls.
- Optional Jev System-1 answerability guard (feature-flagged, fail-open) refuses out-of-scope queries before LLM generation.
- Tenant-scoped retrieval and cache isolation through `tenant_id`.
- Tenant-scoped document listing, processing, and status checks.
- Persistent ingestion lifecycle states: `pending`, `processing`, `completed`, and `failed`.
- GIN-indexed PostgreSQL `TSVECTOR` search column populated when chunks are ingested.
- Reproducible seeding from official OWASP Cheat Sheet Series Markdown documents, with source and license attribution.
- **Structured JSON logging** via `structlog` with configurable level.
- **Production-ready password hashing** via bcrypt (replaces demo hash).
- **Database connection pooling** configurable via `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`.
- **Readiness endpoint** (`/health/ready`) verifying DB, Redis, and Groq connectivity.
- **Configurable retrieval limits** (`RETRIEVE_LIMIT`) and rerank top-K (`RERANK_TOP_K`).
- **Token-based authentication** with SHA-256 bearer-token hashes, bcrypt password hashes, configurable expiry (`TOKEN_EXPIRY_HOURS`), and revocation (logout).
- Bearer tokens are stored as SHA-256 hashes; raw tokens are returned only when issued.
- Tenant-scoped fixed-window rate limits on `/documents`, `/search`, and `/ask`.
- **Multi-tenant isolation** via Organization → User → APIToken hierarchy with `tenant_id` on every row.

On startup, `init_db()` migrates existing `api_tokens` rows by hashing their bearer tokens with SHA-256, then removes the legacy plaintext `token` column. Back up the database before upgrading; downgrading to a version that expects plaintext tokens requires restoring that backup.

## Request flow

The main question-answering path is intentionally ordered from inexpensive checks to deeper work:

1. Resolve the active tenant and query embedding.
2. Return a sufficiently similar cached answer when available.
3. Retrieve tenant-scoped candidates using pgvector and PostgreSQL full-text search.
4. Reject empty or weak context, then rerank the best candidates locally.
5. Optionally ask Jev whether the top context can answer the question. A high-confidence refusal stops before generation; Jev errors fail open.
6. Generate a Pydantic-validated answer and derive source document IDs from retrieved rows.

The guard is disabled by default, so deployments that leave `USE_JEV` unset retain the original retrieval, cache, and generation behavior.

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/health/ready` | Readiness check (DB, Redis, Groq) |
| GET | `/me` | Current user's tenant context (requires Bearer token) |
| POST | `/admin/organizations` | Create a tenant organization |
| POST | `/admin/users` | Create a user in an organization |
| POST | `/admin/tokens` | Issue a bearer token (login) |
| DELETE | `/admin/tokens/{token_id}` | Revoke a token (logout) |
| POST | `/admin/tokens/cleanup` | Clean up expired tokens (admin only) |
| POST | `/documents` | Store a source document |
| GET | `/documents` | List source documents |
| PUT | `/documents/{id}` | Replace source content, clear chunks, and mark pending |
| DELETE | `/documents/{id}` | Delete a document and its chunks |
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

Example `/documents` request:

```json
{
  "title": "Return Policy",
  "content": "Returns are accepted within 30 days."
}
```

The API ignores extra `tenant_id` fields. In normal mode (`USE_MOCK_TENANT=false`), it resolves the tenant from the bearer token and never accepts a client-selected tenant. `USE_MOCK_TENANT=true` is a development-only authentication bypass that assigns every request the configured `MOCK_TENANT_ID`.

`POST /admin/tokens` issues an opaque bearer token. The raw value is returned once; only its SHA-256 hash is stored. Login failures for unknown users and incorrect passwords both return `401 Invalid credentials`. Only admins can create admins or organization owners; organization owners can create members only in their own organization. Other tenants' document IDs return 404 for status, processing, update, and delete routes.

Example `/search` request:

```json
{
  "query": "ERR-4042",
  "top_k": 3
}
```

## Quickstart

Prerequisites: Python 3.11+ and Docker Desktop.

### Option A: PowerShell

```powershell
docker compose up -d
py -3.11 -m venv fastapi-rag\venv
fastapi-rag\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example fastapi-rag\.env
notepad fastapi-rag\.env
Set-Location fastapi-rag
python -m uvicorn main:app --reload
```

### Option B: cmd (Command Prompt)

```cmd
docker compose up -d
python -m venv fastapi-rag\venv
fastapi-rag\venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example fastapi-rag\.env
notepad fastapi-rag\.env
cd fastapi-rag
python -m uvicorn main:app --reload
```

Add `GROQ_API_KEY`, `DATABASE_URL`, and `REDIS_URL` to `fastapi-rag\.env`, then open <http://127.0.0.1:8000/docs>.
For the current single-tenant demo configuration, set `USE_MOCK_TENANT=true` and `MOCK_TENANT_ID=acme_corp`. Redis runs locally at `redis://localhost:6379/0` through Docker Compose.

Key environment variables (see `.env.example` for full list):

| Variable | Default | Purpose |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.35` | Min cosine similarity for vector results |
| `RERANK_TOP_K` | `3` | Chunks kept after CrossEncoder rerank |
| `RETRIEVE_LIMIT` | `15` | Initial candidates before rerank |
| `DB_POOL_SIZE` | `10` | SQLAlchemy connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Extra connections above pool size |
| `MAX_REQUEST_SIZE` | `1048576` | Max request body (1 MB) |
| `RATE_LIMIT_REQUESTS` | `60` | Requests per tenant, route, and fixed window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Fixed-window duration |
| `TOKEN_EXPIRY_HOURS` | `24` | Token lifetime in hours (0 = never expires) |

### Optional Jev configuration

Jev is disabled by default and is designed to fail open. Enable it only when a TypeSafe-compatible endpoint and API key are available:

| Variable | Default | Purpose |
|---|---|---|
| `USE_JEV` | `false` | Enables the answerability check before LLM generation. |
| `JEV_API_KEY` | empty | Bearer token sent to the Jev endpoint. |
| `JEV_API_URL` | `https://api.typesafe.ai/v1/decide` | Jev decision endpoint. |
| `JEV_CONFIDENCE_THRESHOLD` | `0.9` | Minimum refusal confidence required to stop generation. |

The request schema in `fastapi-rag/jev_guard.py` is intentionally isolated because the TypeSafe API contract is still provisional.

To seed the database with real public security guidance from the official OWASP Cheat Sheet Series, keep the API running and execute this from the repository root:

**PowerShell:**
```powershell
fastapi-rag\venv\Scripts\python.exe scripts\seed.py
```

**cmd:**
```cmd
fastapi-rag\venv\Scripts\python.exe scripts\seed.py
```

The seeder downloads 17 curated OWASP Markdown documents, stores their canonical source URLs and CC BY-SA 4.0 attribution, and waits for each document to reach `completed`. It skips documents whose titles already exist for the active tenant. For a clean disposable database, truncate the document tables before running it again.

To exercise health checks, the full RAG pipeline, semantic caching, hallucination protection, and server-side citations in one run, execute the demo after seeding:

**PowerShell:**
```powershell
fastapi-rag\venv\Scripts\python.exe scripts\demo.py
```

**cmd:**
```cmd
fastapi-rag\venv\Scripts\python.exe scripts\demo.py
```

The demo requires the API, PostgreSQL, pgvector, Redis, and `GROQ_API_KEY` to be available. Stage 2 repeats the exact Stage 1 query so the semantic-cache assertion is deterministic. A previous run may already have cached Stage 1, so its label describes the expected pipeline rather than guaranteeing a cache miss.

Run unit/API tests from the repository root:

**PowerShell / cmd:**
```cmd
pip install -r requirements-dev.txt
python -m pytest -q -m "not integration"
```

Run real PostgreSQL/pgvector and Redis integration tests with Docker services:

```powershell
docker compose up -d postgres redis
$env:DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/ragdb"
$env:REDIS_URL = "redis://localhost:6379/0"
python -m pytest -q -m integration
```

Integration tests use a unique PostgreSQL schema and tenant IDs per test, real Redis, deterministic vectors, and no Groq calls. CI provisions the same services and runs unit and integration markers separately.

## Project structure

```text
.
|-- fastapi-rag/
|   |-- main.py         # FastAPI routes, request schemas, pipeline logic
|   |-- database.py     # SQLAlchemy engine, sessions, metadata, connection pool
|   |-- models.py       # Document, chunk, org, user, token models
|   |-- chunking.py     # Overlapping text chunking
|   |-- embeddings.py   # Local sentence-transformer embeddings
|   |-- llm.py          # Groq structured generation and guardrails
|   |-- reranker.py     # Local CrossEncoder relevance reranking
|   |-- semantic_cache.py # Redis vector-similarity response cache
|   |-- jev_guard.py      # Optional fail-open answerability guard
|   `-- .env            # Local secrets; ignored by Git
|-- scripts/
|   |-- seed.py         # Seeds official OWASP source documents through the API
|   `-- demo.py         # Exercises the main RAG features end to end
|-- tests/              # DB- and API-key-free automated tests
|   |-- test_api.py     # FastAPI contract and background-task tests
|   |-- test_chunking.py # Chunk overlap behavior tests
|   |-- test_semantic_cache.py # Cache similarity and tenant-isolation tests
|   `-- test_jev_guard.py # Offline Jev decision and fail-open tests
|-- docker-compose.yml  # Persistent pgvector service
|-- requirements.txt    # Runtime dependencies
|-- requirements-dev.txt # Test dependencies
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
- **Semantic caching:** stores query embeddings and validated answers in Redis for one hour; a cosine similarity above `0.95` returns a recent cached answer without calling Groq. Lookup compares at most the newest 200 entries per tenant and corpus version; older paraphrases can miss the cache and trigger generation. A Redis vector index is the next step if cache volume outgrows this bounded list.
- **Tenant isolation:** derives tenant IDs from authenticated context and scopes document/chunk reads and writes by both document ID and tenant ID.
- **Full-text indexing:** stores PostgreSQL `TSVECTOR` values on chunks and declares a GIN index so keyword search does not recompute vectors for every row.
- **Guard cheap, verify deep:** the optional Jev System-1 guard checks whether the top retrieved context can answer the question before the expensive LLM call; fail-open semantics allow generation to continue when Jev is unavailable or returns malformed data.
- **Operational simplicity:** PostgreSQL and Redis are the only services required by the local stack; embedding and reranking models run in the API process, while Docker provides the stateful dependencies.
- **Explicit API contracts:** Pydantic response models make route outputs visible in generated OpenAPI documentation, while input limits prevent unexpectedly large requests from entering the pipeline.
- **Continuous verification:** GitHub Actions runs unit tests and real Postgres/Redis integration tests on pushes and pull requests.
- **Structured logging:** JSON output via `structlog` enables log aggregation and querying.
- **Secure password storage:** bcrypt via `passlib` replaces the demo hash; `_verify_password` uses constant-time comparison.
- **Connection pooling:** SQLAlchemy pool sizing via env prevents connection exhaustion under load.
- **Readiness probing:** `/health/ready` distinguishes liveness from dependency health for container orchestration.
- **Token authentication with expiry & revocation:** Random bearer tokens are SHA-256 hashed at rest, validated by hash, expire according to `TOKEN_EXPIRY_HOURS`, and can be revoked via `DELETE /admin/tokens/{id}`.
- **Multi-tenant hierarchy:** `Organization` → `User` → `APIToken`; documents and chunks carry `tenant_id`, while authentication rows are scoped by `organization_id` and role.

## Security

Covered: token hash-at-rest and legacy-column migration; bcrypt password verification; generic login failures; role/organization authorization; tenant-scoped document and chunk access; request body limits including chunked requests; fixed-window rate limits; Redis cache tenant/version isolation; server-derived citations.

Known limits: rate limiting and cache invalidation fail open when Redis is unavailable; the rate limit is per tenant and route, not per user/token; the recent-answer index retains at most 200 entries per tenant/version, so older equivalent questions may miss; ingestion uses in-process `BackgroundTasks`; JEV remains an optional provisional fail-open integration. Back up the database before startup migrates existing plaintext tokens.

## Code-reading guide

Follow one question through the system in this order:

1. Start at `fastapi-rag/main.py` and read the `/ask` route.
2. Trace `get_embedding()` into `embeddings.py` to see how the query becomes a vector.
3. Read `retrieve_chunks()` in `main.py` to compare vector search with PostgreSQL full-text search.
4. Follow `rerank_chunks()` into `reranker.py` to see how the top context is selected.
5. Read `semantic_cache.py` to understand how repeated questions can avoid generation.
6. Read `jev_guard.py` to see the optional answerability check and fail-open error handling.
7. Finish in `llm.py` to see structured generation, hallucination signaling, and response validation.

**Production patterns to study:**

- `database.py`: Connection pool config (`pool_size`, `max_overflow`, `pool_pre_ping`)
- `main.py:30-50`: Structured logging setup with `structlog`
- `main.py:75-85`: Bcrypt password hashing and verification
- `main.py`: Request body limit and rate-limit dependencies
- `main.py:117-140`: Readiness endpoint checking DB/Redis/Groq
- `main.py:370-380`: Configurable thresholds via environment variables

Useful learning exercises:

- Change the retrieval limit and observe how the reranker input changes.
- Add a test for a malformed Jev response and verify that the request still proceeds.
- Add a typed response model for one currently dictionary-based endpoint.
- Measure the first `/ask` request against an identical cached request.
- Add one question to an evaluation dataset and verify its expected source document.
- Toggle `SIMILARITY_THRESHOLD` and observe retrieval precision/recall tradeoff.
- Test `/health/ready` with Redis or PostgreSQL stopped.

## Roadmap

- Next.js chat UI
- Golden evaluation dataset
- Multi-tenant metadata filters
- LangGraph agent mode
- Redis vector index to replace the bounded recent-entry list at very large cache sizes
- Durable background job queue to replace FastAPI in-process `BackgroundTasks`
- Alembic migrations for schema versioning
- OpenTelemetry tracing and Prometheus metrics

## License

MIT. See [LICENSE](LICENSE).