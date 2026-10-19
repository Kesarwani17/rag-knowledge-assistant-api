"""API contract tests that avoid PostgreSQL, embeddings, and external LLM calls."""

import importlib
import hashlib
import os
import sys
import types
from pathlib import Path

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from passlib.hash import bcrypt
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).parents[1] / "fastapi-rag"
sys.path.insert(0, str(PROJECT_DIR))


class FakeRAGResponse(BaseModel):
    """Minimal response model needed while importing the API module."""

    answer: str = ""
    is_hallucination: bool = False
    source_document_ids: list[int] = []


def load_api_without_external_services():
    """Import the FastAPI app with database and model dependencies stubbed."""
    # Disable mock tenant mode for tests to verify auth behavior
    os.environ["USE_MOCK_TENANT"] = "false"

    class FakeConnection:
        """Context manager replacing the import-time SQLAlchemy connection."""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def execute(self, statement):
            return None

        def commit(self):
            return None

    class FakeColumn:
        """Mini column object that supports equality comparisons used by filter calls."""

        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            return (self.name, other)

    class FakeOrganization:
        email = FakeColumn("email")

        def __init__(self, name=None, slug=None):
            self.id = None
            self.name = name
            self.slug = slug

    class FakeDocument:
        id = FakeColumn("id")
        tenant_id = FakeColumn("tenant_id")

        def __init__(self, title=None, content=None, tenant_id=None):
            self.id = None
            self.title = title
            self.content = content
            self.tenant_id = tenant_id
            self.status = "pending"

    class FakeDocumentChunk:
        document_id = FakeColumn("document_id")
        tenant_id = FakeColumn("tenant_id")

        def __init__(self, document_id=None, chunk_index=None, chunk_text=None,
                     embedding=None, tenant_id=None, search_vector=None):
            self.id = None
            self.document_id = document_id
            self.chunk_index = chunk_index
            self.chunk_text = chunk_text
            self.embedding = embedding
            self.tenant_id = tenant_id
            self.search_vector = search_vector

    class FakeUser:
        email = FakeColumn("email")
        token = FakeColumn("token")

        def __init__(self, email=None, password_hash=None, organization_id=None, role="member"):
            self.id = None
            self.email = email
            self.password_hash = password_hash
            self.organization_id = organization_id
            self.role = role

    class FakeAPIToken:
        token_hash = FakeColumn("token_hash")

        def __init__(self, token_hash=None, user_id=None, organization_id=None, role="member", expires_at=None):
            self.id = None
            self.token_hash = token_hash
            self.user_id = user_id
            self.organization_id = organization_id
            self.role = role
            self.expires_at = expires_at

    class FakeQuery:
        """Minimal query object used by the tenant auth routes."""

        def __init__(self, session, model):
            self.session = session
            self.model = model
            self._filters = []

        def filter(self, *args, **kwargs):
            self._filters.extend(args)
            self._filters.extend(kwargs.items())
            return self

        def first(self):
            return next(iter(self.all()), None)

        def all(self):
            records = self.session.records(self.model)
            return [record for record in records if self._matches(record)]

        def delete(self, synchronize_session=False):
            records = self.session.records(self.model)
            matching_records = [record for record in records if self._matches(record)]
            for record in matching_records:
                records.remove(record)
            return len(matching_records)

        def _matches(self, record):
            for predicate in self._filters:
                if not isinstance(predicate, tuple):
                    continue
                column, expected = predicate
                name = getattr(column, "name", column)
                if getattr(record, name, None) != expected:
                    return False
            return True

    class FakeSession:
        """In-memory session used by offline tests."""

        def __init__(self):
            self.organizations = []
            self.users = []
            self.tokens = []
            self.documents = []
            self.chunks = []

        def records(self, model):
            return {
                FakeOrganization: self.organizations,
                FakeUser: self.users,
                FakeAPIToken: self.tokens,
                FakeDocument: self.documents,
                FakeDocumentChunk: self.chunks,
            }.get(model, [])

        def query(self, model):
            return FakeQuery(self, model)

        def add(self, obj):
            if isinstance(obj, FakeOrganization):
                self.organizations.append(obj)
            elif isinstance(obj, FakeUser):
                self.users.append(obj)
            elif isinstance(obj, FakeAPIToken):
                self.tokens.append(obj)
            elif isinstance(obj, FakeDocument):
                self.documents.append(obj)
            elif isinstance(obj, FakeDocumentChunk):
                self.chunks.append(obj)

        def commit(self):
            for records in (self.organizations, self.users, self.tokens, self.documents, self.chunks):
                for item in records:
                    if item.id is None:
                        item.id = len(records)

        def refresh(self, obj):
            if isinstance(obj, FakeOrganization):
                if obj.id is None:
                    obj.id = len(self.organizations)
            elif isinstance(obj, FakeUser):
                if obj.id is None:
                    obj.id = len(self.users)
            elif isinstance(obj, FakeAPIToken):
                if obj.id is None:
                    obj.id = len(self.tokens)
            elif isinstance(obj, FakeDocument):
                if obj.id is None:
                    obj.id = len(self.documents)

        def delete(self, obj):
            self.records(type(obj)).remove(obj)

        def close(self):
            return None

    class FakeRateLimitRedis:
        def __init__(self):
            self.counters = {}
            self.expirations = {}

        def incr(self, key):
            self.counters[key] = self.counters.get(key, 0) + 1
            return self.counters[key]

        def expire(self, key, seconds):
            self.expirations[key] = seconds
            return True

    shared_session = FakeSession()

    fake_database = types.ModuleType("database")
    fake_database.Base = types.SimpleNamespace(
        metadata=types.SimpleNamespace(create_all=lambda bind: None)
    )
    fake_database.SessionLocal = lambda: shared_session
    fake_database.engine = types.SimpleNamespace(connect=FakeConnection)
    fake_database.init_db = lambda: None

    fake_models = types.ModuleType("models")
    fake_models.Organization = FakeOrganization
    fake_models.User = FakeUser
    fake_models.APIToken = FakeAPIToken
    fake_models.Document = FakeDocument
    fake_models.DocumentChunk = FakeDocumentChunk
    fake_llm = types.ModuleType("llm")
    fake_llm.RAGResponse = FakeRAGResponse
    fake_llm.generate_answer = lambda query, context_chunks: FakeRAGResponse()
    fake_reranker = types.ModuleType("reranker")
    fake_reranker.rerank_chunks = lambda query, chunks: chunks
    fake_semantic_cache = types.ModuleType("semantic_cache")
    fake_semantic_cache.cache_response = lambda tenant_id, query_vector, response: None
    fake_semantic_cache.get_cached_response = lambda tenant_id, query_vector: None
    fake_semantic_cache.bump_corpus_version = lambda tenant_id: None
    fake_semantic_cache.get_corpus_version = lambda tenant_id: 0
    fake_semantic_cache.redis_client = FakeRateLimitRedis()

    fake_structlog = types.ModuleType("structlog")
    fake_structlog.configure = lambda *args, **kwargs: None
    fake_structlog.get_logger = lambda: types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    fake_structlog.processors = types.SimpleNamespace(
        add_log_level=lambda *args, **kwargs: None,
        TimeStamper=lambda *args, **kwargs: None,
        JSONRenderer=lambda *args, **kwargs: None,
    )
    fake_structlog.make_filtering_bound_logger = lambda *args, **kwargs: None
    fake_structlog.wrapper_class = types.SimpleNamespace()

    fake_passlib = types.ModuleType("passlib")
    fake_passlib.hash = types.SimpleNamespace()
    fake_passlib.hash.bcrypt = types.SimpleNamespace(
        hash=lambda p: f"hashed::{p}",
        verify=lambda p, h: True,
    )

    # Add a fake admin token to the shared session so admin routes work
    admin_token = FakeAPIToken(
        token_hash=hashlib.sha256(b"admin_test_token").hexdigest(),
        user_id=1,
        organization_id=1,
        role="admin",
    )
    admin_token.id = 1
    shared_session.tokens.append(admin_token)
    # Also add a fake user for that token
    admin_user = FakeUser(email="admin@test.com", password_hash="hashed::secret", organization_id=1, role="admin")
    admin_user.id = 1
    shared_session.users.append(admin_user)

    previous_modules = {
        name: sys.modules.get(name)
        for name in (
            "database",
            "models",
            "embeddings",
            "llm",
            "reranker",
            "semantic_cache",
            "structlog",
            "passlib",
            "passlib.hash",
            "main",
        )
    }
    sys.modules["database"] = fake_database
    sys.modules["models"] = fake_models
    sys.modules["embeddings"] = types.ModuleType("embeddings")
    sys.modules["embeddings"].get_embedding = lambda text: []
    sys.modules["llm"] = fake_llm
    sys.modules["reranker"] = fake_reranker
    sys.modules["semantic_cache"] = fake_semantic_cache
    sys.modules["structlog"] = fake_structlog
    sys.modules["passlib"] = fake_passlib
    sys.modules["passlib.hash"] = fake_passlib.hash
    sys.modules.pop("main", None)

    try:
        return importlib.import_module("main")
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_health_endpoint_returns_ok():
    """The health endpoint should be available without infrastructure."""
    app_module = load_api_without_external_services()
    response = TestClient(app_module.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_document_requires_fields():
    """Document creation should reject a body missing required fields."""
    app_module = load_api_without_external_services()
    headers = {"Authorization": "Bearer admin_test_token"}
    response = TestClient(app_module.app).post("/documents", json={}, headers=headers)
    assert response.status_code == 422


def test_hash_password_bcrypt_round_trips():
    """Password hashes produced by the API verify with the configured bcrypt backend."""
    app_module = load_api_without_external_services()
    password_hash = app_module._hash_password("x")

    assert bcrypt.verify("x", password_hash)
    assert not bcrypt.verify("different", password_hash)


def test_passwords_over_bcrypt_byte_limit_return_422():
    """Password endpoints reject values that exceed bcrypt's 72-byte input limit."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    password = "é" * 37
    headers = {"Authorization": "Bearer admin_test_token"}

    user_response = client.post(
        "/admin/users",
        json={
            "email": "long-password@example.com",
            "password": password,
            "organization_id": 1,
            "role": "member",
        },
        headers=headers,
    )
    token_response = client.post(
        "/admin/tokens",
        json={
            "email": "admin@test.com",
            "password": password,
            "organization_id": 1,
        },
    )

    assert user_response.status_code == 422
    assert token_response.status_code == 422


def test_create_document_rejects_oversized_content():
    """Document input should have a bounded size at the API boundary."""
    app_module = load_api_without_external_services()
    headers = {"Authorization": "Bearer admin_test_token"}
    response = TestClient(app_module.app).post(
        "/documents",
        json={
            "title": "Large document",
            "content": "x" * 100_001,
            "tenant_id": "tenant-a",
        },
        headers=headers,
    )

    assert response.status_code == 422


def test_chunked_request_body_exceeding_limit_returns_413(monkeypatch):
    """The body limit applies while receiving chunked payloads without Content-Length."""
    monkeypatch.setenv("MAX_REQUEST_SIZE", "128")
    app_module = load_api_without_external_services()
    chunks = iter([b'{"title":"large","content":"', b"x" * 256, b'"}'])

    response = TestClient(app_module.app).post(
        "/documents",
        content=chunks,
        headers={
            "Authorization": "Bearer admin_test_token",
            "Content-Type": "application/json",
            "Transfer-Encoding": "chunked",
        },
    )

    assert response.status_code == 413


def test_search_rejects_oversized_query():
    """Search input should have a bounded size at the API boundary."""
    app_module = load_api_without_external_services()
    headers = {"Authorization": "Bearer admin_test_token"}
    response = TestClient(app_module.app).post(
        "/search",
        json={"query": "x" * 1_001},
        headers=headers,
    )

    assert response.status_code == 422


def test_process_trigger_queues_work_and_reports_processing():
    """Processing should be queued and expose its initial status immediately."""
    app_module = load_api_without_external_services()
    background_tasks = BackgroundTasks()

    class FakeColumn:
        """Column stand-in that supports the filter expression used by the API."""

        def __eq__(self, other):
            return True

    class FakeDocument:
        """Database document stand-in for status persistence assertions."""

        id = FakeColumn()
        tenant_id = FakeColumn()

        def __init__(self):
            self.status = "pending"

    document = FakeDocument()

    class FakeQuery:
        """Minimal SQLAlchemy query chain used by the status routes."""

        def filter(self, *expressions):
            return self

        def first(self):
            return document

    class FakeSession:
        """Minimal session that records commits and supports cleanup."""

        def query(self, model):
            return FakeQuery()

        def commit(self):
            return None

        def close(self):
            return None

    app_module.models.Document = FakeDocument
    app_module.SessionLocal = FakeSession

    tenant_context = {"tenant_id": "1", "user_id": 1, "role": "admin"}
    response = app_module.trigger_process(42, background_tasks, tenant_context)

    assert response["status"] == "processing"
    assert response["doc_id"] == 42
    assert background_tasks.tasks[0].func is app_module.process_heavy_document
    assert document.status == "processing"
    assert app_module.get_document_status(42, tenant_context) == {"status": "processing"}


def test_document_processing_bumps_tenant_corpus_version(monkeypatch):
    """A successful embedding write invalidates the processed tenant's cache."""
    app_module = load_api_without_external_services()
    session = app_module.SessionLocal()
    document = app_module.models.Document(
        title="Source",
        content="processed content",
        tenant_id="1",
    )
    session.add(document)
    session.commit()
    version_bumps = []
    monkeypatch.setattr(app_module, "chunk_text", lambda content: [content])
    monkeypatch.setattr(app_module, "get_embedding", lambda content: [0.0])
    monkeypatch.setattr(app_module, "bump_corpus_version", version_bumps.append)

    app_module.process_heavy_document(document.id, "1")

    assert document.status == "completed"
    assert len(session.chunks) == 1
    assert session.chunks[0].tenant_id == "1"
    assert version_bumps == ["1"]


def test_admin_can_create_organization_and_issue_token():
    """The admin bootstrap flow should create an organization and issue a tenant-scoped token."""
    app_module = load_api_without_external_services()

    headers = {"Authorization": "Bearer admin_test_token"}

    org_response = TestClient(app_module.app).post(
        "/admin/organizations",
        json={"name": "Acme Health"},
        headers=headers,
    )
    assert org_response.status_code == 200
    assert org_response.json()["name"] == "Acme Health"

    user_response = TestClient(app_module.app).post(
        "/admin/users",
        json={
            "email": "alice@acme.com",
            "password": "secret123",
            "organization_id": 1,
            "role": "admin",
        },
        headers=headers,
    )
    assert user_response.status_code == 200
    assert user_response.json()["email"] == "alice@acme.com"

    token_response = TestClient(app_module.app).post(
        "/admin/tokens",
        json={"email": "alice@acme.com", "password": "secret123", "organization_id": 1},
    )
    assert token_response.status_code == 200
    assert token_response.json()["organization_id"] == 1
    raw_token = token_response.json()["token"]
    assert raw_token
    stored_token = app_module.SessionLocal().tokens[-1]
    assert stored_token.token_hash == app_module.hash_token(raw_token)
    assert not hasattr(stored_token, "token")


def test_protected_route_requires_valid_bearer_token():
    """Protected tenant-scoped routes should reject missing or invalid bearer tokens."""
    app_module = load_api_without_external_services()

    # Test without token - should fail
    response = TestClient(app_module.app).get("/me")
    assert response.status_code == 401

    # Test with invalid token - should fail
    response = TestClient(app_module.app).get(
        "/me",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401

    # Test with valid admin token - should succeed
    response = TestClient(app_module.app).get(
        "/me",
        headers={"Authorization": "Bearer admin_test_token"},
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "1"
    assert response.json()["user_id"] == 1
    assert response.json()["role"] == "admin"


def add_test_token(app_module, token_value, organization_id, role="member"):
    """Add an authenticated tenant token to the API's in-memory database."""
    session = app_module.SessionLocal()
    organization = app_module.models.Organization(
        name=f"Organization {organization_id}",
        slug=f"organization-{organization_id}",
    )
    session.add(organization)
    user = app_module.models.User(
        email=f"user-{token_value}@example.com",
        password_hash="hashed::secret",
        organization_id=organization_id,
        role=role,
    )
    session.add(user)
    session.commit()
    token = app_module.models.APIToken(
        token_hash=app_module.hash_token(token_value),
        user_id=user.id,
        organization_id=organization_id,
        role=role,
    )
    session.add(token)
    session.commit()
    return {"Authorization": f"Bearer {token_value}"}


def test_documents_and_search_are_isolated_between_authenticated_tenants():
    """A body-supplied tenant is ignored, and other tenants cannot list or access it."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    tenant_a = add_test_token(app_module, "tenant-a-token", 1)
    tenant_b = add_test_token(app_module, "tenant-b-token", 2)

    created = client.post(
        "/documents",
        json={"title": "A private doc", "content": "tenant A source", "tenant_id": "2"},
        headers=tenant_a,
    )

    assert created.status_code == 200
    document_id = created.json()["id"]
    assert app_module.SessionLocal().documents[0].tenant_id == "1"
    assert [doc["id"] for doc in client.get("/documents", headers=tenant_a).json()] == [document_id]
    assert client.get("/documents", headers=tenant_b).json() == []

    app_module.retrieve_chunks = lambda db, query, tenant_id, limit, query_embedding=None: (
        ([{"text": "tenant A source", "document_id": document_id, "similarity_score": 1.0}]
         if tenant_id == "1" else []),
        [],
    )
    search = client.post("/search", json={"query": "private"}, headers=tenant_b)
    assert search.status_code == 200
    assert search.json() == []

    status_response = client.get(f"/documents/{document_id}/status", headers=tenant_b)
    process_response = client.post(f"/documents/{document_id}/process", headers=tenant_b)
    update_response = client.put(
        f"/documents/{document_id}",
        json={"title": "Changed", "content": "unauthorized update"},
        headers=tenant_b,
    )
    delete_response = client.delete(f"/documents/{document_id}", headers=tenant_b)
    assert status_response.status_code == 404
    assert process_response.status_code == 404
    assert update_response.status_code == 404
    assert delete_response.status_code == 404


def test_ask_cache_is_tenant_isolated_and_document_update_invalidates(monkeypatch):
    """Updating a tenant's source invalidates its old answer and permits reprocessing."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    tenant_a = add_test_token(app_module, "tenant-a-token", 1)
    tenant_b = add_test_token(app_module, "tenant-b-token", 2)
    versions = {"1": 0, "2": 0}
    cached_answers = {}
    corpus_answers = {"1": "A before update", "2": "B answer"}

    app_module.get_embedding = lambda text: [1.0, 0.0]
    app_module.USE_JEV = False
    app_module.retrieve_chunks = lambda db, query, tenant_id, limit, query_embedding=None: (
        [{"text": corpus_answers[tenant_id], "document_id": 1, "similarity_score": 1.0}],
        [1.0, 0.0],
    )
    app_module.rerank_chunks = lambda query, chunks: chunks
    app_module.generate_answer = lambda query, chunks: app_module.RAGResponse(
        answer=chunks[0]["text"], is_hallucination=False
    )
    app_module.get_cached_response = lambda tenant_id, vector: cached_answers.get(
        (tenant_id, versions[tenant_id])
    )
    app_module.get_corpus_version = lambda tenant_id: versions[tenant_id]
    app_module.cache_response = lambda tenant_id, vector, response, corpus_version=None: cached_answers.__setitem__(
        (tenant_id, versions[tenant_id]), response
    )
    app_module.bump_corpus_version = lambda tenant_id: versions.__setitem__(
        tenant_id, versions[tenant_id] + 1
    )

    created = client.post(
        "/documents",
        json={"title": "Original source", "content": "A before update"},
        headers=tenant_a,
    )
    document_id = created.json()["id"]

    answer_a = client.post("/ask", json={"query": "same question"}, headers=tenant_a)
    answer_b = client.post("/ask", json={"query": "same question"}, headers=tenant_b)
    assert answer_a.json()["answer"] == "A before update"
    assert answer_b.json()["answer"] == "B answer"

    corpus_answers["1"] = "A after update"
    updated = client.put(
        f"/documents/{document_id}",
        json={"title": "Updated source", "content": "A after update"},
        headers=tenant_a,
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "pending"

    updated_answer = client.post("/ask", json={"query": "same question"}, headers=tenant_a)
    assert updated_answer.json()["answer"] == "A after update"

    reprocess = client.post(f"/documents/{document_id}/process", headers=tenant_a)
    assert reprocess.status_code == 200


def test_delete_document_removes_chunks_from_search(monkeypatch):
    """Deleting a document removes its tenant-owned chunks from search results."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    tenant_a = add_test_token(app_module, "tenant-a-token", 1)
    document_response = client.post(
        "/documents",
        json={"title": "To delete", "content": "searchable content"},
        headers=tenant_a,
    )
    document_id = document_response.json()["id"]
    session = app_module.SessionLocal()
    chunk = app_module.models.DocumentChunk(
        document_id=document_id,
        chunk_index=0,
        chunk_text="searchable content",
        embedding=[1.0],
        tenant_id="1",
    )
    session.add(chunk)
    session.commit()
    app_module.retrieve_chunks = lambda db, query, tenant_id, limit, query_embedding=None: (
        [
            {"text": item.chunk_text, "document_id": item.document_id, "similarity_score": 1.0}
            for item in session.chunks
            if item.tenant_id == tenant_id
        ],
        [],
    )

    before = client.post("/search", json={"query": "searchable"}, headers=tenant_a)
    deleted = client.delete(f"/documents/{document_id}", headers=tenant_a)
    after = client.post("/search", json={"query": "searchable"}, headers=tenant_a)

    assert before.json()
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert session.chunks == []
    assert after.json() == []


def test_ask_does_not_cache_answer_when_corpus_changes_during_generation(monkeypatch):
    """An in-flight stale answer must not be written under the updated version."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    tenant_a = add_test_token(app_module, "tenant-a-token", 1)
    version = {"value": 0}
    cached_writes = []

    app_module.get_embedding = lambda text: [1.0, 0.0]
    app_module.USE_JEV = False
    app_module.get_corpus_version = lambda tenant_id: version["value"]
    app_module.get_cached_response = lambda tenant_id, vector: None
    app_module.retrieve_chunks = lambda db, query, tenant_id, limit, query_embedding=None: (
        [{"text": "old corpus", "document_id": 1, "similarity_score": 1.0}],
        [1.0, 0.0],
    )
    app_module.rerank_chunks = lambda query, chunks: chunks

    def generate_during_update(query, chunks):
        version["value"] += 1
        return app_module.RAGResponse(answer="stale", is_hallucination=False)

    app_module.generate_answer = generate_during_update
    app_module.cache_response = lambda *args, **kwargs: cached_writes.append((args, kwargs))

    response = client.post("/ask", json={"query": "same question"}, headers=tenant_a)

    assert response.status_code == 200
    assert response.json()["answer"] == "stale"
    assert cached_writes == []


def test_org_owner_cannot_create_user_in_another_org_but_admin_can():
    """User creation checks the requested organization rather than a hardcoded ID."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    owner = add_test_token(app_module, "org-two-owner", 2, "org_owner")
    admin = {"Authorization": "Bearer admin_test_token"}
    payload = {
        "email": "new-user@example.com",
        "password": "secret123",
        "organization_id": 1,
        "role": "member",
    }

    forbidden = client.post("/admin/users", json=payload, headers=owner)
    created = client.post("/admin/users", json=payload, headers=admin)
    assert forbidden.status_code == 403
    assert created.status_code == 200
    assert created.json()["organization_id"] == 1


def test_user_creation_role_authorization_rules():
    """Only admins may assign privileged roles; invalid roles fail validation."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    owner = add_test_token(app_module, "org-two-owner", 2, "org_owner")
    admin = {"Authorization": "Bearer admin_test_token"}

    def create_user(email, role, headers):
        return client.post(
            "/admin/users",
            json={
                "email": email,
                "password": "secret123",
                "organization_id": 2,
                "role": role,
            },
            headers=headers,
        )

    owner_admin = create_user("owner-admin@example.com", "admin", owner)
    owner_member = create_user("owner-member@example.com", "member", owner)
    unknown_role = create_user("unknown-role@example.com", "superuser", owner)
    admin_admin = create_user("admin-admin@example.com", "admin", admin)

    assert owner_admin.status_code == 403
    assert owner_member.status_code == 200
    assert owner_member.json()["role"] == "member"
    assert unknown_role.status_code == 422
    assert admin_admin.status_code == 200
    assert admin_admin.json()["role"] == "admin"


def test_document_rate_limit_returns_retry_after(monkeypatch):
    """Tenant document writes are rejected after the configured fixed-window cap."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    monkeypatch.setattr(app_module, "RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(app_module, "RATE_LIMIT_WINDOW_SECONDS", 60)
    headers = {"Authorization": "Bearer admin_test_token"}

    responses = [
        client.post(
            "/documents",
            json={"title": f"Doc {index}", "content": "content"},
            headers=headers,
        )
        for index in range(3)
    ]
    tenant_b = add_test_token(app_module, "tenant-b-token", 2)
    other_tenant_response = client.post(
        "/documents",
        json={"title": "Other tenant document", "content": "content"},
        headers=tenant_b,
    )

    assert [response.status_code for response in responses] == [200, 200, 429]
    assert int(responses[-1].headers["Retry-After"]) > 0
    assert other_tenant_response.status_code == 200


def test_search_and_ask_rate_limits(monkeypatch):
    """Search and ask use independent tenant-scoped fixed-window limits."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    monkeypatch.setattr(app_module, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(app_module, "RATE_LIMIT_WINDOW_SECONDS", 60)
    app_module.get_embedding = lambda query: [1.0, 0.0]
    app_module.retrieve_chunks = lambda db, query, tenant_id, limit, query_embedding=None: ([], [1.0, 0.0])
    headers = {"Authorization": "Bearer admin_test_token"}

    for path in ("/search", "/ask"):
        responses = [
            client.post(path, json={"query": "question"}, headers=headers)
            for _ in range(2)
        ]
        assert [response.status_code for response in responses] == [200, 429]
        assert responses[-1].headers.get("Retry-After")


def test_rate_limit_fails_open_when_redis_is_unavailable(monkeypatch):
    """Redis errors must not make authenticated API requests unavailable."""
    app_module = load_api_without_external_services()

    class UnavailableRedis:
        def incr(self, key):
            raise app_module.RedisError("connection unavailable")

    monkeypatch.setattr(app_module, "redis_client", UnavailableRedis())
    response = TestClient(app_module.app).post(
        "/documents",
        json={"title": "Available", "content": "content"},
        headers={"Authorization": "Bearer admin_test_token"},
    )

    assert response.status_code == 200


def test_login_unknown_user_and_wrong_password_have_identical_responses(monkeypatch):
    """Missing accounts use dummy bcrypt verification and share the login error."""
    app_module = load_api_without_external_services()
    client = TestClient(app_module.app)
    verified_hashes = []

    def verify_password(password, password_hash):
        verified_hashes.append(password_hash)
        return False

    monkeypatch.setattr(app_module, "_verify_password", verify_password)
    app_module.SessionLocal().users[0].password_hash = "correct-password-hash"

    unknown_user = client.post(
        "/admin/tokens",
        json={"email": "missing@example.com", "password": "incorrect", "organization_id": 1},
    )
    wrong_password = client.post(
        "/admin/tokens",
        json={"email": "admin@test.com", "password": "incorrect", "organization_id": 1},
    )

    assert unknown_user.status_code == wrong_password.status_code == 401
    assert unknown_user.json() == wrong_password.json() == {"detail": "Invalid credentials"}
    assert verified_hashes == [app_module.DUMMY_PASSWORD_HASH, "correct-password-hash"]