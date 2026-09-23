"""API contract tests that avoid PostgreSQL, embeddings, and external LLM calls."""

import importlib
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

        def __init__(self, token=None, user_id=None, organization_id=None, role="member"):
            self.id = None
            self.token = token
            self.user_id = user_id
            self.organization_id = organization_id
            self.role = role

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
            if self.model is FakeUser:
                for predicate in self._filters:
                    if isinstance(predicate, tuple):
                        left, right = predicate
                        left_name = getattr(left, "name", None)
                        if left_name == "email" or left == "email":
                            for user in self.session.users:
                                if user.email == right:
                                    return user
                return None
            if self.model is FakeAPIToken:
                for predicate in self._filters:
                    if isinstance(predicate, tuple):
                        left, right = predicate
                        left_name = getattr(left, "name", None)
                        if left_name == "token" or left == "token":
                            for token in self.session.tokens:
                                if token.token == right:
                                    return token
                return None
            return None

    class FakeSession:
        """In-memory session used by offline tests."""

        def __init__(self):
            self.organizations = []
            self.users = []
            self.tokens = []

        def query(self, model):
            return FakeQuery(self, model)

        def add(self, obj):
            if isinstance(obj, FakeOrganization):
                self.organizations.append(obj)
            elif isinstance(obj, FakeUser):
                self.users.append(obj)
            elif isinstance(obj, FakeAPIToken):
                self.tokens.append(obj)

        def commit(self):
            for item in self.organizations:
                if item.id is None:
                    item.id = len(self.organizations)
            for item in self.users:
                if item.id is None:
                    item.id = len(self.users)
            for item in self.tokens:
                if item.id is None:
                    item.id = len(self.tokens)

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
    fake_llm = types.ModuleType("llm")
    fake_llm.RAGResponse = FakeRAGResponse
    fake_llm.generate_answer = lambda query, context_chunks: FakeRAGResponse()
    fake_reranker = types.ModuleType("reranker")
    fake_reranker.rerank_chunks = lambda query, chunks: chunks
    fake_semantic_cache = types.ModuleType("semantic_cache")
    fake_semantic_cache.cache_response = lambda tenant_id, query_vector, response: None
    fake_semantic_cache.get_cached_response = lambda tenant_id, query_vector: None

    previous_modules = {
        name: sys.modules.get(name)
        for name in (
            "database",
            "models",
            "embeddings",
            "llm",
            "reranker",
            "semantic_cache",
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
    response = TestClient(app_module.app).post("/documents", json={})
    assert response.status_code == 422


def test_create_document_rejects_oversized_content():
    """Document input should have a bounded size at the API boundary."""
    app_module = load_api_without_external_services()
    response = TestClient(app_module.app).post(
        "/documents",
        json={
            "title": "Large document",
            "content": "x" * 100_001,
            "tenant_id": "tenant-a",
        },
    )

    assert response.status_code == 422


def test_search_rejects_oversized_query():
    """Search input should have a bounded size at the API boundary."""
    app_module = load_api_without_external_services()
    response = TestClient(app_module.app).post(
        "/search",
        json={"query": "x" * 1_001},
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

    response = app_module.trigger_process(42, background_tasks)

    assert response["status"] == "processing"
    assert response["doc_id"] == 42
    assert background_tasks.tasks[0].func is app_module.process_heavy_document
    assert document.status == "processing"
    assert app_module.get_document_status(42) == {"status": "processing"}


def test_admin_can_create_organization_and_issue_token():
    """The admin bootstrap flow should create an organization and issue a tenant-scoped token."""
    app_module = load_api_without_external_services()

    org_response = TestClient(app_module.app).post(
        "/admin/organizations",
        json={"name": "Acme Health"},
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

    response = TestClient(app_module.app).get("/me")
    assert response.status_code == 401

    response = TestClient(app_module.app).get(
        "/me",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401