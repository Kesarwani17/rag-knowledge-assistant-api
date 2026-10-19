"""Redis-backed semantic response caching for grounded RAG answers."""

import json
import logging
import math
import os
from uuid import uuid4

import redis

from llm import RAGResponse

CACHE_PREFIX = "rag:semantic-cache:"
CACHE_VERSION_PREFIX = "rag:corpus-version:"
CACHE_TTL_SECONDS = 3600
SIMILARITY_THRESHOLD = 0.95
logger = logging.getLogger(__name__)

redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return cosine similarity for two equal-length vectors."""
    if len(first) != len(second):
        return 0.0
    dot_product = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0
    return dot_product / (first_norm * second_norm)


def get_corpus_version(tenant_id: str) -> int | None:
    """Return the tenant corpus version, or None when Redis is unavailable."""
    try:
        version = redis_client.get(f"{CACHE_VERSION_PREFIX}{tenant_id}")
        return int(version or 0)
    except (redis.RedisError, TypeError, ValueError):
        return None


def _cache_namespace(tenant_id: str, corpus_version: int | None = None) -> str:
    version = corpus_version if corpus_version is not None else get_corpus_version(tenant_id)
    if version is None:
        raise redis.RedisError("Unable to read tenant corpus version")
    return f"{CACHE_PREFIX}{tenant_id}:{version}:"


def bump_corpus_version(tenant_id: str) -> None:
    """Invalidate a tenant's cached answers without deleting old keys."""
    try:
        redis_client.incr(f"{CACHE_VERSION_PREFIX}{tenant_id}")
    except redis.RedisError:
        logger.warning("semantic_cache_version_bump_failed", extra={"tenant_id": tenant_id})


def get_cached_response(tenant_id: str, query_vector: list[float]) -> RAGResponse | None:
    """Return the closest cached response when similarity exceeds the threshold."""
    try:
        cache_namespace = _cache_namespace(tenant_id)
        for key in redis_client.scan_iter(match=f"{cache_namespace}*"):
            cached = redis_client.get(key)
            if not cached:
                continue
            payload = json.loads(cached)
            similarity = cosine_similarity(query_vector, payload["query_vector"])
            if similarity > SIMILARITY_THRESHOLD:
                return RAGResponse.model_validate(payload["response"])
    except (redis.RedisError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def cache_response(
    tenant_id: str,
    query_vector: list[float],
    response: RAGResponse,
    corpus_version: int | None = None,
) -> None:
    """Store a generated response and its query vector with a bounded TTL."""
    payload = {
        "query_vector": query_vector,
        "response": response.model_dump(),
    }
    try:
        redis_client.set(
            f"{_cache_namespace(tenant_id, corpus_version)}{uuid4().hex}",
            json.dumps(payload),
            ex=CACHE_TTL_SECONDS,
        )
    except (redis.RedisError, TypeError, ValueError):
        return