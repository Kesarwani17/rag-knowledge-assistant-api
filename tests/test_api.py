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

    fake_database = types.ModuleType("database")
    fake_database.Base = types.SimpleNamespace(
        metadata=types.SimpleNamespace(create_all=lambda bind: None)
    )
    fake_database.SessionLocal = lambda: None
    fake_database.engine = types.SimpleNamespace(connect=FakeConnection)

    fake_models = types.ModuleType("models")
    fake_llm = types.ModuleType("llm")
    fake_llm.RAGResponse = FakeRAGResponse
    fake_llm.generate_answer = lambda query, context_chunks: FakeRAGResponse()
    fake_reranker = types.ModuleType("reranker")
    fake_reranker.rerank_chunks = lambda query, chunks: chunks
    fake_semantic_cache = types.ModuleType("semantic_cache")
    fake_semantic_cache.cache_response = lambda query_vector, response: None
    fake_semantic_cache.get_cached_response = lambda query_vector: None

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


def test_process_trigger_queues_work_and_reports_processing():
    """Processing should be queued and expose its initial status immediately."""
    app_module = load_api_without_external_services()
    background_tasks = BackgroundTasks()

    response = app_module.trigger_process(42, background_tasks)

    assert response["status"] == "processing"
    assert response["doc_id"] == 42
    assert background_tasks.tasks[0].func is app_module.process_heavy_document
    assert app_module.get_document_status(42) == {"status": "processing"}