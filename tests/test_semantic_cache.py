"""Offline tests for Redis semantic cache matching and serialization."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1] / "fastapi-rag"
sys.path.insert(0, str(PROJECT_DIR))

from llm import RAGResponse
import semantic_cache


class FakeRedis:
    """Small in-memory Redis substitute for cache behavior tests."""

    def __init__(self):
        self.values = {}

    def scan_iter(self, match):
        return iter(self.values)

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex):
        self.values[key] = value


def test_cache_returns_response_above_similarity_threshold(monkeypatch):
    """Semantically equivalent vectors should bypass generation."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(semantic_cache, "redis_client", fake_redis)
    response = RAGResponse(
        answer="cached",
        is_hallucination=False,
        source_document_ids=[7],
    )

    semantic_cache.cache_response([1.0, 0.0], response)

    cached = semantic_cache.get_cached_response([0.999, 0.001])
    assert cached == response


def test_cache_misses_below_similarity_threshold(monkeypatch):
    """A materially different vector should not reuse an answer."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(semantic_cache, "redis_client", fake_redis)
    response = RAGResponse(answer="cached", is_hallucination=False)

    semantic_cache.cache_response([1.0, 0.0], response)

    assert semantic_cache.get_cached_response([0.0, 1.0]) is None