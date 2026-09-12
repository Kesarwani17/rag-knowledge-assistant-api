"""Redis-backed semantic response caching for grounded RAG answers."""

import json
import math
import os
from uuid import uuid4

import redis

from llm import RAGResponse

CACHE_PREFIX = "rag:semantic-cache:"
CACHE_TTL_SECONDS = 3600
SIMILARITY_THRESHOLD = 0.95

redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return cosine similarity for two equal-length vectors."""
    dot_product = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0
    return dot_product / (first_norm * second_norm)


def get_cached_response(query_vector: list[float]) -> RAGResponse | None:
    """Return the closest cached response when similarity exceeds the threshold."""
    try:
        for key in redis_client.scan_iter(match=f"{CACHE_PREFIX}*"):
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


def cache_response(query_vector: list[float], response: RAGResponse) -> None:
    """Store a generated response and its query vector with a bounded TTL."""
    payload = {
        "query_vector": query_vector,
        "response": response.model_dump(),
    }
    try:
        redis_client.set(
            f"{CACHE_PREFIX}{uuid4().hex}",
            json.dumps(payload),
            ex=CACHE_TTL_SECONDS,
        )
    except (redis.RedisError, TypeError, ValueError):
        return