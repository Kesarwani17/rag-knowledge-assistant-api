"""Offline tests for Redis semantic cache matching and serialization."""

import sys
from fnmatch import fnmatch
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1] / "fastapi-rag"
sys.path.insert(0, str(PROJECT_DIR))

from llm import RAGResponse
import semantic_cache


class FakeRedis:
    """Small in-memory Redis substitute for cache behavior tests."""

    def __init__(self):
        self.values = {}
        self.counters = {}

    def scan_iter(self, match):
        return (key for key in self.values if fnmatch(key, match))

    def get(self, key):
        return self.counters.get(key, self.values.get(key))

    def set(self, key, value, ex):
        self.values[key] = value

    def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]


def test_cache_returns_response_above_similarity_threshold(monkeypatch):
    """Semantically equivalent vectors should bypass generation."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(semantic_cache, "redis_client", fake_redis)
    response = RAGResponse(
        answer="cached",
        is_hallucination=False,
        source_document_ids=[7],
    )

    semantic_cache.cache_response("tenant-a", [1.0, 0.0], response)

    cached = semantic_cache.get_cached_response("tenant-a", [0.999, 0.001])
    assert cached == response


def test_cache_misses_below_similarity_threshold(monkeypatch):
    """A materially different vector should not reuse an answer."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(semantic_cache, "redis_client", fake_redis)
    response = RAGResponse(answer="cached", is_hallucination=False)

    semantic_cache.cache_response("tenant-a", [1.0, 0.0], response)

    assert semantic_cache.get_cached_response("tenant-a", [0.0, 1.0]) is None


def test_cache_does_not_cross_tenants(monkeypatch):
    """A matching vector in another tenant must not return a cached answer."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(semantic_cache, "redis_client", fake_redis)
    response = RAGResponse(answer="tenant-a", is_hallucination=False)

    semantic_cache.cache_response("tenant-a", [1.0, 0.0], response)

    assert semantic_cache.get_cached_response("tenant-b", [1.0, 0.0]) is None


def test_corpus_version_bump_invalidates_cached_responses(monkeypatch):
    """A changed tenant corpus must no longer reuse its earlier answer."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(semantic_cache, "redis_client", fake_redis)
    response = RAGResponse(answer="stale", is_hallucination=False)

    semantic_cache.cache_response("tenant-a", [1.0, 0.0], response)
    semantic_cache.bump_corpus_version("tenant-a")

    assert semantic_cache.get_cached_response("tenant-a", [1.0, 0.0]) is None
    assert semantic_cache.get_cached_response("tenant-b", [1.0, 0.0]) is None


def test_cache_response_uses_captured_corpus_version(monkeypatch):
    """A response generated before an update stays in the old namespace."""
    fake_redis = FakeRedis()
    monkeypatch.setattr(semantic_cache, "redis_client", fake_redis)
    response = RAGResponse(answer="old", is_hallucination=False)

    semantic_cache.cache_response("tenant-a", [1.0, 0.0], response, corpus_version=0)
    semantic_cache.bump_corpus_version("tenant-a")

    assert semantic_cache.get_cached_response("tenant-a", [1.0, 0.0]) is None


def test_corpus_version_bump_logs_and_continues_when_redis_is_down(monkeypatch, caplog):
    """Redis outages must not make document writes fail."""
    class UnavailableRedis:
        def incr(self, key):
            raise semantic_cache.redis.RedisError("connection unavailable")

    monkeypatch.setattr(semantic_cache, "redis_client", UnavailableRedis())

    semantic_cache.bump_corpus_version("tenant-a")

    assert "semantic_cache_version_bump_failed" in caplog.text


def test_cosine_similarity_rejects_mismatched_vectors():
    """Vectors with different dimensions must not be compared partially."""
    assert semantic_cache.cosine_similarity([1.0, 0.0], [1.0]) == 0.0