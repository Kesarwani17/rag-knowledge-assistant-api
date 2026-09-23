"""FastAPI application exposing document ingestion, retrieval, and RAG answers."""

import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import desc, func, select

from llm import generate_answer, RAGResponse
from database import SessionLocal, init_db
import models
from chunking import chunk_text
from embeddings import get_embedding
from reranker import rerank_chunks
from semantic_cache import cache_response, get_cached_response
from jev_guard import CONFIDENCE_THRESHOLD, USE_JEV, is_answerable

load_dotenv()

init_db()

app = FastAPI(title="RAG Learning API")
security = HTTPBearer(auto_error=False)


def get_current_tenant_id() -> str:
    """Return the configured tenant identifier for the current deployment."""
    return os.getenv("MOCK_TENANT_ID", "default")


def _hash_password(password: str) -> str:
    """Create a minimal deterministic password hash for demo/admin flows."""
    return f"hashed::{password}::{len(password)}"


def _is_valid_token(token_value: str | None) -> bool:
    """Return whether the token exists in the product-style auth registry."""
    if not token_value:
        return False
    db = SessionLocal()
    try:
        token = db.query(models.APIToken).filter(models.APIToken.token == token_value).first()
        return token is not None
    finally:
        db.close()


def get_authenticated_context(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, object]:
    """Resolve the active tenant and user from a bearer token if one is supplied."""
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token_value = creds.credentials
    if not _is_valid_token(token_value):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")

    db = SessionLocal()
    try:
        token = db.query(models.APIToken).filter(models.APIToken.token == token_value).first()
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
        return {"tenant_id": str(token.organization_id), "user_id": token.user_id, "role": token.role}
    finally:
        db.close()


class DocumentIn(BaseModel):
    """Request body for storing a source document."""

    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=100_000)
    tenant_id: str = Field(..., min_length=1, max_length=100)


class DocumentResponse(BaseModel):
    """Response returned after a document is stored."""

    id: int
    title: str
    status: str


class DocumentDetailResponse(BaseModel):
    """Document fields returned by the tenant-scoped listing endpoint."""

    id: int
    title: str
    content: str


class ProcessResponse(BaseModel):
    """Response returned when document processing is queued."""

    status: str
    doc_id: int
    message: str


class StatusResponse(BaseModel):
    """Current processing status for a document."""

    status: str


class ChunkResponse(BaseModel):
    """Retrieved chunk and its relevance metadata."""

    text: str
    document_id: int
    similarity_score: float


class HealthResponse(BaseModel):
    """Lightweight service health response."""

    status: str


class OrganizationCreateRequest(BaseModel):
    """Admin request to create a new organization."""

    name: str = Field(..., min_length=1, max_length=200)


class OrganizationResponse(BaseModel):
    """Public organization metadata returned by admin routes."""

    id: int
    name: str
    slug: str


class UserCreateRequest(BaseModel):
    """Admin request to create a user in an organization."""

    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=200)
    organization_id: int = Field(..., ge=1)
    role: str = Field(default="member", min_length=1, max_length=64)


class UserResponse(BaseModel):
    """User summary returned after admin creation."""

    id: int
    email: str
    organization_id: int
    role: str


class TokenCreateRequest(BaseModel):
    """Admin request to issue a bearer token."""

    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=200)
    organization_id: int = Field(..., ge=1)


class TokenResponse(BaseModel):
    """Issued token payload for client authentication."""

    token: str
    user_id: int
    organization_id: int
    role: str


class SearchQuery(BaseModel):
    """Request body for semantic document search."""

    query: str = Field(..., min_length=1, max_length=1_000)
    top_k: int = Field(3, ge=1, le=10)


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""
    return {"status": "ok"}


@app.post("/admin/organizations", response_model=OrganizationResponse)
def create_organization(req: OrganizationCreateRequest) -> dict[str, object]:
    """Create a tenant organization for product-style multi-tenancy."""
    db = SessionLocal()
    try:
        slug = req.name.strip().lower().replace(" ", "-")
        org = models.Organization(name=req.name.strip(), slug=slug)
        db.add(org)
        db.commit()
        db.refresh(org)
        return {"id": org.id, "name": org.name, "slug": org.slug}
    finally:
        db.close()


@app.post("/admin/users", response_model=UserResponse)
def create_user(req: UserCreateRequest) -> dict[str, object]:
    """Create a user and associate them with an organization."""
    db = SessionLocal()
    try:
        user = models.User(
            email=req.email.strip().lower(),
            password_hash=_hash_password(req.password),
            organization_id=req.organization_id,
            role=req.role,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"id": user.id, "email": user.email, "organization_id": user.organization_id, "role": user.role}
    finally:
        db.close()


@app.post("/admin/tokens", response_model=TokenResponse)
def issue_token(req: TokenCreateRequest) -> dict[str, object]:
    """Issue a bearer token for a user in an organization."""
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == req.email.strip().lower()).first()
        if not user or user.organization_id != req.organization_id:
            raise HTTPException(status_code=404, detail="User not found")

        if user.password_hash != _hash_password(req.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token_value = f"tenant_{uuid.uuid4().hex}"
        token = models.APIToken(
            token=token_value,
            user_id=user.id,
            organization_id=user.organization_id,
            role=user.role,
        )
        db.add(token)
        db.commit()
        db.refresh(token)
        return {"token": token.token, "user_id": token.user_id, "organization_id": token.organization_id, "role": token.role}
    finally:
        db.close()


@app.get("/me", response_model=dict[str, object])
def me(context: dict[str, object] = Depends(get_authenticated_context)) -> dict[str, object]:
    """Return the authenticated user's tenant context."""
    return {"tenant_id": context["tenant_id"], "user_id": context["user_id"], "role": context["role"]}


@app.post("/documents", response_model=DocumentResponse)
def create_document(doc: DocumentIn) -> dict[str, object]:
    """Store a source document and return its generated identifier."""
    db = SessionLocal()
    
    use_mock_tenant = os.getenv("USE_MOCK_TENANT", "false").lower() == "true"
    if use_mock_tenant:
        tenant_id = os.getenv("MOCK_TENANT_ID")
        if not tenant_id:
            raise RuntimeError("MOCK_TENANT_ID must be set")
    else:
        tenant_id = doc.tenant_id

    try:
        new_doc = models.Document(title=doc.title, content=doc.content, tenant_id=tenant_id)
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        return {"id": new_doc.id, "title": new_doc.title, "status": "stored"}
    finally:
        db.close()


@app.get("/documents", response_model=list[DocumentDetailResponse])
def list_documents() -> list[dict[str, object]]:
    """Return all stored source documents."""
    db = SessionLocal()
    try:
        tenant_id = get_current_tenant_id()
        docs = (
            db.query(models.Document)
            .filter(models.Document.tenant_id == tenant_id)
            .all()
        )
        return [{"id": d.id, "title": d.title, "content": d.content} for d in docs]
    finally:
        db.close()


def retrieve_chunks(
    db,
    query: str,
    tenant_id: str,
    limit: int,
    query_embedding: list[float] | None = None,
) -> tuple[list[dict[str, object]], list[float]]:
    """Retrieve tenant-scoped chunks with dense and full-text search."""
    if query_embedding is None:
        query_embedding = get_embedding(query)

    vector_stmt = (
        select(
            models.DocumentChunk,
            models.DocumentChunk.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .where(models.DocumentChunk.tenant_id == tenant_id)
        .order_by("distance")
        .limit(limit)
    )
    vector_results = db.execute(vector_stmt).all()

    keyword_stmt = (
        select(
            models.DocumentChunk,
            func.ts_rank(
                models.DocumentChunk.search_vector,
                func.plainto_tsquery("english", query),
            ).label("keyword_score"),
        )
        .where(
            models.DocumentChunk.tenant_id == tenant_id,
            models.DocumentChunk.search_vector.match(query),
        )
        .order_by(desc("keyword_score"))
        .limit(limit)
    )
    keyword_results = db.execute(keyword_stmt).all()

    chunks: list[dict[str, object]] = []
    chunk_ids: set[int] = set()
    for chunk, distance in vector_results:
        similarity = 1.0 - float(distance)
        if similarity >= 0.35:
            chunks.append({
                "text": chunk.chunk_text,
                "document_id": chunk.document_id,
                "similarity_score": round(similarity, 4),
            })
            chunk_ids.add(chunk.id)

    for chunk, keyword_score in keyword_results:
        if chunk.id not in chunk_ids:
            chunks.append({
                "text": chunk.chunk_text,
                "document_id": chunk.document_id,
                "similarity_score": round(float(keyword_score), 4),
            })
            chunk_ids.add(chunk.id)

    return chunks, query_embedding


def process_heavy_document(doc_id: int) -> None:
    """Chunk and embed a stored document in a background task."""
    db = SessionLocal()
    doc = None
    try:
        doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
        if not doc:
            return

        chunks = chunk_text(doc.content)

        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            db.add(
                models.DocumentChunk(
                    document_id=doc.id,
                    chunk_index=index,
                    chunk_text=chunk,
                    embedding=embedding,
                    tenant_id=doc.tenant_id,
                    search_vector=func.to_tsvector("english", chunk),
                )
            )

        db.commit()
        doc.status = "completed"
        db.commit()
    except Exception as exc:
        db.rollback()
        if doc is None:
            doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
        print(f"Error processing doc {doc_id}: {exc}")
    finally:
        db.close()


@app.post("/documents/{doc_id}/process", response_model=ProcessResponse)
def trigger_process(doc_id: int, background_tasks: BackgroundTasks) -> dict[str, object]:
    """Queue document processing and return before embedding work begins."""
    db = SessionLocal()
    try:
        tenant_id = get_current_tenant_id()
        doc = (
            db.query(models.Document)
            .filter(
                models.Document.id == doc_id,
                models.Document.tenant_id == tenant_id,
            )
            .first()
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        if doc.status == "processing":
            raise HTTPException(status_code=409, detail="Document is already processing")
        if doc.status == "completed":
            raise HTTPException(status_code=409, detail="Document is already processed")

        doc.status = "processing"
        db.commit()
        background_tasks.add_task(process_heavy_document, doc_id)
        return {
            "status": "processing",
            "doc_id": doc_id,
            "message": "Embedding started. Check /documents/{doc_id}/status for updates.",
        }
    finally:
        db.close()


@app.get("/documents/{doc_id}/status", response_model=StatusResponse)
def get_document_status(doc_id: int) -> dict[str, object]:
    """Return the current background processing status for a document."""
    db = SessionLocal()
    try:
        tenant_id = get_current_tenant_id()
        doc = (
            db.query(models.Document)
            .filter(
                models.Document.id == doc_id,
                models.Document.tenant_id == tenant_id,
            )
            .first()
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": doc.status}
    finally:
        db.close()


@app.post("/search", response_model=list[ChunkResponse])
def search_documents(req: SearchQuery) -> list[dict[str, object]]:
    """Combine dense vector and sparse keyword retrieval for a query."""
    db = SessionLocal()
    try:
        tenant_id = get_current_tenant_id()
        chunks, _ = retrieve_chunks(db, req.query, tenant_id, req.top_k)
        return chunks[: req.top_k]
    finally:
        db.close()

class AskQuery(BaseModel):
    """Request body for a guarded question-answering request."""

    query: str = Field(..., min_length=1, max_length=1_000)

@app.post("/ask", response_model=RAGResponse)
def ask_question(req: AskQuery) -> RAGResponse:
    """Retrieve, rerank, and generate a grounded answer from relevant chunks."""
    db = SessionLocal()
    try:
        query_embedding = get_embedding(req.query)
        tenant_id = get_current_tenant_id()
        cached_response = get_cached_response(tenant_id, query_embedding)
        if cached_response:
            return cached_response

        chunks, _ = retrieve_chunks(db, req.query, tenant_id, 15, query_embedding)

        if not chunks:
            return RAGResponse(
                answer="I cannot answer this based on the provided documents.",
                is_hallucination=True,
                source_document_ids=[]
            )
        # 2. RERANKING: Let the local CrossEncoder select the best context.
        chunks = rerank_chunks(req.query, chunks)[:3]

        if USE_JEV:
            context_text = "\n\n".join(c["text"] for c in chunks)
            answerable, confidence = is_answerable(req.query, context_text)
            if not answerable and confidence >= CONFIDENCE_THRESHOLD:
                return RAGResponse(
                    answer="I cannot answer this based on the provided documents.",
                    is_hallucination=True,
                    source_document_ids=[],
                )

        # 3. GENERATION: Send only the top 3 chunks to Groq with guardrails
        llm_response = generate_answer(req.query, chunks)

        # --- SENIOR FIX: Force the correct IDs from the database ---
        # Don't rely on the LLM to remember IDs. We inject the exact ones we retrieved.
        llm_response.source_document_ids = list(set(c["document_id"] for c in chunks))
        cache_response(tenant_id, query_embedding, llm_response)

        return llm_response

    finally:
        db.close()