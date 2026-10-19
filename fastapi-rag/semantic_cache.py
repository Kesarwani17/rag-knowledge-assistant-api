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
CACHE_MAX_ENTRIES = 200
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


def _cache_namespace(tenant_id: str, corpus_version: int) -> str:
    return f"{CACHE_PREFIX}{tenant_id}:{corpus_version}:"


def _cache_index_key(tenant_id: str, corpus_version: int) -> str:
    return f"{_cache_namespace(tenant_id, corpus_version)}recent"


def bump_corpus_version(tenant_id: str) -> None:
    """Invalidate a tenant's cached answers without deleting old keys."""
    try:
        redis_client.incr(f"{CACHE_VERSION_PREFIX}{tenant_id}")
    except redis.RedisError:
        logger.warning("semantic_cache_version_bump_failed", extra={"tenant_id": tenant_id})


def get_cached_response(tenant_id: str, query_vector: list[float]) -> RAGResponse | None:
    """Return the closest cached response when similarity exceeds the threshold."""
    try:
        corpus_version = get_corpus_version(tenant_id)
        if corpus_version is None:
            return None
        for key in redis_client.lrange(
            _cache_index_key(tenant_id, corpus_version),
            0,
            CACHE_MAX_ENTRIES - 1,
        ):
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
        if corpus_version is None:
            corpus_version = get_corpus_version(tenant_id)
        if corpus_version is None:
            return

        cache_key = f"{_cache_namespace(tenant_id, corpus_version)}{uuid4().hex}"
        redis_client.set(
            cache_key,
            json.dumps(payload),
            ex=CACHE_TTL_SECONDS,
        )
        index_key = _cache_index_key(tenant_id, corpus_version)
        redis_client.lpush(index_key, cache_key)
        redis_client.ltrim(index_key, 0, CACHE_MAX_ENTRIES - 1)
        redis_client.expire(index_key, CACHE_TTL_SECONDS)
    except (redis.RedisError, TypeError, ValueError):
        return