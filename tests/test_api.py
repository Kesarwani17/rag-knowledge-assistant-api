"""API contract tests that avoid PostgreSQL, embeddings, and external LLM calls."""

import importlib
import os
import sys
import types
from pathlib import Path

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
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
        token = FakeColumn("token")

        def __init__(self, token=None, user_id=None, organization_id=None, role="member", expires_at=None):
            self.id = None
            self.token = token
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

        def close(self):
            return None

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
    admin_token = FakeAPIToken(token="admin_test_token", user_id=1, organization_id=1, role="admin")
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
    assert token_response.json()["token"]


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
        token=token_value,
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
    assert status_response.status_code == 404
    assert process_response.status_code == 404


def test_ask_cache_is_tenant_isolated_and_document_creation_invalidates(monkeypatch):
    """Identical questions stay tenant-scoped and a corpus change expires old answers."""
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

    answer_a = client.post("/ask", json={"query": "same question"}, headers=tenant_a)
    answer_b = client.post("/ask", json={"query": "same question"}, headers=tenant_b)
    assert answer_a.json()["answer"] == "A before update"
    assert answer_b.json()["answer"] == "B answer"

    corpus_answers["1"] = "A after update"
    added = client.post(
        "/documents",
        json={"title": "Updated source", "content": "new tenant A source"},
        headers=tenant_a,
    )
    assert added.status_code == 200

    updated_answer = client.post("/ask", json={"query": "same question"}, headers=tenant_a)
    assert updated_answer.json()["answer"] == "A after update"


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