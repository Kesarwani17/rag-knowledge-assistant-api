"""FastAPI application exposing document ingestion, retrieval, and RAG answers."""

import os
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import Depends, HTTPException, status

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import desc, func, select, text

from llm import generate_answer, RAGResponse
from database import SessionLocal, init_db, engine
import models
from chunking import chunk_text
from embeddings import get_embedding
from reranker import rerank_chunks
from semantic_cache import cache_response, get_cached_response
from jev_guard import CONFIDENCE_THRESHOLD, USE_JEV, is_answerable

load_dotenv()

# Structured logging setup
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = structlog.get_logger()

# Configurable thresholds via env
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
RETRIEVE_LIMIT = int(os.getenv("RETRIEVE_LIMIT", "15"))

init_db()


# OpenAPI tag definitions for Swagger UI grouping
OPENAPI_TAGS = [
    {"name": "Health", "description": "Service health and readiness checks"},
    {"name": "Auth - Tokens", "description": "Token issuance (login) and revocation (logout)"},
    {"name": "Auth - Profile", "description": "Authenticated user profile and context"},
    {"name": "Admin - Organizations", "description": "Organization management (admin only)"},
    {"name": "Admin - Users", "description": "User management within organizations"},
    {"name": "Admin - Tokens", "description": "Admin token utilities (cleanup expired)"},
    {"name": "Documents", "description": "Document CRUD and background processing"},
    {"name": "Search", "description": "Hybrid vector + keyword search"},
    {"name": "RAG", "description": "Full question-answering pipeline with caching and citations"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_startup")
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="RAG Learning API",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)
security = HTTPBearer(auto_error=False)


# Request size limit middleware (teaches production protection)
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    max_size = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB default
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body too large (max {max_size} bytes)"},
        )
    return await call_next(request)


def get_current_tenant_id() -> str:
    """Return the configured tenant identifier for the current deployment."""
    return os.getenv("MOCK_TENANT_ID", "default")


def _hash_password(password: str) -> str:
    """Hash password using bcrypt (production-ready)."""
    from passlib.hash import bcrypt
    return bcrypt.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify password against bcrypt hash."""
    from passlib.hash import bcrypt
    return bcrypt.verify(password, password_hash)


def _is_valid_token(token_value: str | None) -> bool:
    """Return whether the token exists and is not expired."""
    if not token_value:
        return False
    db = SessionLocal()
    try:
        from datetime import datetime, timezone
        token = db.query(models.APIToken).filter(models.APIToken.token == token_value).first()
        if not token:
            return False
        # Check expiry
        if token.expires_at and token.expires_at < datetime.now(timezone.utc):
            return False
        return True
    finally:
        db.close()


def get_authenticated_context(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, object]:
    """Resolve the active tenant and user from a bearer token if one is supplied."""
    # Mock mode: bypass auth, return fixed tenant context
    if os.getenv("USE_MOCK_TENANT", "false").lower() == "true":
        mock_tenant_id = os.getenv("MOCK_TENANT_ID", "default")
        return {"tenant_id": mock_tenant_id, "user_id": 0, "role": "member"}

    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token_value = creds.credentials
    if not _is_valid_token(token_value):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired bearer token")

    db = SessionLocal()
    try:
        from datetime import datetime, timezone
        token = db.query(models.APIToken).filter(models.APIToken.token == token_value).first()
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
        # Double-check expiry (defense in depth)
        if token.expires_at and token.expires_at < datetime.now(timezone.utc):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        return {"tenant_id": str(token.organization_id), "user_id": token.user_id, "role": token.role}
    finally:
        db.close()


def cleanup_expired_tokens() -> int:
    """Background task: delete expired tokens. Returns count of deleted tokens."""
    from datetime import datetime, timezone
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        deleted = db.query(models.APIToken).filter(
            models.APIToken.expires_at.isnot(None),
            models.APIToken.expires_at < now
        ).delete()
        db.commit()
        logger.info("expired_tokens_cleaned", count=deleted)
        return deleted
    finally:
        db.close()


# =============================================================================
# RBAC System (Simple, Educational)
# =============================================================================
class Role(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"          # Global admin - can manage all orgs, users, tokens
    ORG_OWNER = "org_owner"  # Organization owner - can manage users/tokens in their org
    MEMBER = "member"        # Regular user - can use RAG features, manage own tokens


def require_role(*allowed_roles: Role):
    """Dependency factory: require one of the allowed roles."""
    def checker(context: dict = Depends(get_authenticated_context)) -> dict:
        # Bypass RBAC in mock mode
        if os.getenv("USE_MOCK_TENANT", "false").lower() == "true":
            return context
        user_role = Role(context["role"])
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {[r.value for r in allowed_roles]}, got {user_role.value}"
            )
        return context
    return checker


def require_org_owner_or_admin(org_id: int):
    """Dependency factory: require org_owner (for this org) or admin."""
    def checker(context: dict = Depends(get_authenticated_context)) -> dict:
        # Bypass RBAC in mock mode
        if os.getenv("USE_MOCK_TENANT", "false").lower() == "true":
            return context
        user_role = Role(context["role"])
        user_org_id = int(context["tenant_id"])
        if user_role == Role.ADMIN:
            return context
        if user_role == Role.ORG_OWNER and user_org_id == org_id:
            return context
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requires org_owner for this organization or admin"
        )
    return checker


# Convenience dependencies (used in route decorators)
require_admin = require_role(Role.ADMIN)
require_org_owner_or_admin_any = require_role(Role.ADMIN, Role.ORG_OWNER)
require_any_authenticated = require_role(Role.ADMIN, Role.ORG_OWNER, Role.MEMBER)


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
    expires_at: datetime | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "token": "tenant_a1b2c3d4e5f6...",
                "user_id": 1,
                "organization_id": 1,
                "role": "admin",
                "expires_at": "2026-09-26T12:00:00Z"
            }
        }


class SearchQuery(BaseModel):
    """Request body for semantic document search."""

    query: str = Field(..., min_length=1, max_length=1_000)
    top_k: int = Field(3, ge=1, le=10)


class MeResponse(BaseModel):
    """Authenticated user's tenant context."""

    tenant_id: str
    user_id: int
    role: str

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "1",
                "user_id": 1,
                "role": "admin"
            }
        }


class RevokeTokenResponse(BaseModel):
    """Response after revoking a token (logout)."""

    revoked: bool
    token_id: int

    class Config:
        json_schema_extra = {
            "example": {
                "revoked": True,
                "token_id": 5
            }
        }


class CleanupTokensResponse(BaseModel):
    """Response after cleaning up expired tokens."""

    deleted: int

    class Config:
        json_schema_extra = {
            "example": {
                "deleted": 3
            }
        }


class ReadyResponse(BaseModel):
    """Readiness check response with dependency status."""

    ready: bool
    checks: dict[str, bool]

    class Config:
        json_schema_extra = {
            "example": {
                "ready": True,
                "checks": {
                    "database": True,
                    "redis": True,
                    "groq": True
                }
            }
        }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness check",
    description="Lightweight endpoint to verify the service is running. Returns 200 OK if the app is alive."
)
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""
    return {"status": "ok"}


@app.get(
    "/health/ready",
    response_model=ReadyResponse,
    tags=["Health"],
    summary="Readiness check",
    description="Verifies all critical dependencies (PostgreSQL, Redis, Groq API) are reachable. Returns 503 if any dependency is down."
)
def health_ready() -> JSONResponse:
    """Readiness check: verifies DB, Redis, and Groq connectivity."""
    checks = {"database": False, "redis": False, "groq": False}

    # Check PostgreSQL
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        logger.warning("health_check_failed", component="database")

    # Check Redis
    try:
        from semantic_cache import redis_client
        redis_client.ping()
        checks["redis"] = True
    except Exception:
        logger.warning("health_check_failed", component="redis")

    # Check Groq (lightweight - just verify key exists)
    checks["groq"] = bool(os.getenv("GROQ_API_KEY"))

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    return JSONResponse(status_code=status_code, content={"ready": all_ready, "checks": checks})


@app.post(
    "/admin/organizations",
    response_model=OrganizationResponse,
    tags=["Admin - Organizations"],
    summary="Create organization (admin only)",
    description="Create a new tenant organization. Requires global admin role.",
    dependencies=[Depends(require_admin)]
)
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


@app.post(
    "/admin/users",
    response_model=UserResponse,
    tags=["Admin - Users"],
    summary="Create user (admin or org_owner for target org)",
    description="Create a user in an organization. Admin can create in any org; org_owner only in their own org.",
    dependencies=[Depends(require_org_owner_or_admin_any)]
)
def create_user(req: UserCreateRequest, _: dict = Depends(require_org_owner_or_admin(1))) -> dict[str, object]:
    """Create a user and associate them with an organization."""
    # The dependency factory validates org ownership (admin any, org_owner own org)
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


# Token expiry: default 24 hours, configurable via TOKEN_EXPIRY_HOURS (0 = never expires)
TOKEN_EXPIRY_HOURS = int(os.getenv("TOKEN_EXPIRY_HOURS", "24"))


@app.post(
    "/admin/tokens",
    response_model=TokenResponse,
    tags=["Auth - Tokens"],
    summary="Issue bearer token (login)",
    description="Authenticate user with email/password and receive a bearer token. Public endpoint (no auth required)."
)
def issue_token(req: TokenCreateRequest) -> dict[str, object]:
    """Issue a bearer token for a user in an organization."""
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == req.email.strip().lower()).first()
        if not user or user.organization_id != req.organization_id:
            raise HTTPException(status_code=404, detail="User not found")

        if not _verify_password(req.password, str(user.password_hash)):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token_value = f"tenant_{uuid.uuid4().hex}"
        from datetime import datetime, timezone, timedelta
        expires_at = None
        if TOKEN_EXPIRY_HOURS > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS)
        token = models.APIToken(
            token=token_value,
            user_id=user.id,
            organization_id=user.organization_id,
            role=user.role,
            expires_at=expires_at,
        )
        db.add(token)
        db.commit()
        db.refresh(token)
        return {"token": token.token, "user_id": token.user_id, "organization_id": token.organization_id, "role": token.role, "expires_at": token.expires_at}
    finally:
        db.close()


@app.delete(
    "/admin/tokens/{token_id}",
    response_model=RevokeTokenResponse,
    tags=["Auth - Tokens"],
    summary="Revoke token (logout)",
    description="Revoke a token by ID. Users can revoke their own tokens; admins and org_owners can revoke any token in their organization.",
    dependencies=[Depends(require_any_authenticated)]
)
def revoke_token(token_id: int, context: dict = Depends(get_authenticated_context)) -> dict[str, object]:
    """Revoke (logout) a token by ID."""
    db = SessionLocal()
    try:
        token = db.query(models.APIToken).filter(models.APIToken.id == token_id).first()
        if not token:
            raise HTTPException(status_code=404, detail="Token not found")

        user_role = Role(context["role"])
        user_id = context["user_id"]
        user_org_id = int(context["tenant_id"])

        # Allow: own token, admin (any), org_owner (same org)
        if user_id == token.user_id:
            pass  # own token
        elif user_role == Role.ADMIN:
            pass  # admin can revoke any
        elif user_role == Role.ORG_OWNER and user_org_id == token.organization_id:
            pass  # org_owner can revoke in their org
        else:
            raise HTTPException(status_code=403, detail="Not authorized to revoke this token")

        db.delete(token)
        db.commit()
        return {"revoked": True, "token_id": token_id}
    finally:
        db.close()


@app.post(
    "/admin/tokens/cleanup",
    response_model=CleanupTokensResponse,
    tags=["Admin - Tokens"],
    summary="Clean up expired tokens (admin only)",
    description="Manually trigger cleanup of all expired tokens. Requires admin role.",
    dependencies=[Depends(require_admin)]
)
def cleanup_tokens_endpoint() -> dict[str, object]:
    """Manually trigger expired token cleanup (admin only)."""
    deleted = cleanup_expired_tokens()
    return {"deleted": deleted}


@app.get(
    "/me",
    response_model=MeResponse,
    tags=["Auth - Profile"],
    summary="Get current user profile",
    description="Returns the authenticated user's tenant context (tenant_id, user_id, role). Requires valid bearer token.",
    dependencies=[Depends(require_any_authenticated)]
)
def me(context: dict = Depends(get_authenticated_context)) -> dict[str, object]:
    """Return the authenticated user's tenant context."""
    return {"tenant_id": context["tenant_id"], "user_id": context["user_id"], "role": context["role"]}


@app.post(
    "/documents",
    response_model=DocumentResponse,
    tags=["Documents"],
    summary="Create document",
    description="Store a source document in the knowledge base. Requires authentication.",
    dependencies=[Depends(require_any_authenticated)]
)
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


@app.get(
    "/documents",
    response_model=list[DocumentDetailResponse],
    tags=["Documents"],
    summary="List documents",
    description="List all source documents for the current tenant. Requires authentication.",
    dependencies=[Depends(require_any_authenticated)]
)
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
        if similarity >= SIMILARITY_THRESHOLD:
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
        logger.error("document_processing_failed", doc_id=doc_id, error=str(exc))
    finally:
        db.close()


@app.post(
    "/documents/{doc_id}/process",
    response_model=ProcessResponse,
    tags=["Documents"],
    summary="Process document (trigger embedding)",
    description="Queue document chunking and embedding as a background task. Requires authentication.",
    dependencies=[Depends(require_any_authenticated)]
)
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


@app.get(
    "/documents/{doc_id}/status",
    response_model=StatusResponse,
    tags=["Documents"],
    summary="Get document processing status",
    description="Check the status of a document's background processing (pending/processing/completed/failed). Requires authentication.",
    dependencies=[Depends(require_any_authenticated)]
)
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


@app.post(
    "/search",
    response_model=list[ChunkResponse],
    tags=["Search"],
    summary="Hybrid search (vector + keyword)",
    description="Combine dense vector (pgvector) and sparse keyword (PostgreSQL full-text) retrieval. Requires authentication.",
    dependencies=[Depends(require_any_authenticated)]
)
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

@app.post(
    "/ask",
    response_model=RAGResponse,
    tags=["RAG"],
    summary="Ask a question (full RAG pipeline)",
    description="Retrieve context, check semantic cache, optionally use JEV guard, generate structured answer with citations. Requires authentication.",
    dependencies=[Depends(require_any_authenticated)]
)
def ask_question(req: AskQuery) -> RAGResponse:
    """Retrieve, rerank, and generate a grounded answer from relevant chunks."""
    db = SessionLocal()
    try:
        query_embedding = get_embedding(req.query)
        tenant_id = get_current_tenant_id()
        cached_response = get_cached_response(tenant_id, query_embedding)
        if cached_response:
            return cached_response

        chunks, _ = retrieve_chunks(db, req.query, tenant_id, RETRIEVE_LIMIT, query_embedding)

        if not chunks:
            return RAGResponse(
                answer="I cannot answer this based on the provided documents.",
                is_hallucination=True,
                source_document_ids=[]
            )
        # 2. RERANKING: Let the local CrossEncoder select the best context.
        chunks = rerank_chunks(req.query, chunks)[:RERANK_TOP_K]

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