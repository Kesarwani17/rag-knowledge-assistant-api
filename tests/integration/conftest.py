"""Fixtures for integration tests against real PostgreSQL and Redis services."""

import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_DIR = Path(__file__).parents[2] / "fastapi-rag"
sys.path.insert(0, str(PROJECT_DIR))


@pytest.fixture
def integration_services(monkeypatch):
    """Provide a real DB/Redis-backed API app with deterministic embeddings."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/ragdb",
    )
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("USE_MOCK_TENANT", "false")

    from redis import Redis
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    admin_engine = create_engine(database_url)
    probe_redis = Redis.from_url(redis_url, decode_responses=True)
    try:
        with admin_engine.begin() as connection:
            connection.execute(text("SELECT 1"))
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"))
        probe_redis.ping()
    except Exception as exc:
        admin_engine.dispose()
        probe_redis.close()
        if os.getenv("CI", "").lower() == "true":
            raise
        pytest.skip(f"PostgreSQL and Redis integration services are unavailable: {exc}")
    probe_redis.close()

    suffix = uuid.uuid4().hex
    schema = f"rag_it_{suffix}"
    tenant_ids = [100_000_000 + int(suffix[:7], 16), 200_000_000 + int(suffix[7:14], 16)]
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    database_url_object = make_url(database_url)
    query = dict(database_url_object.query)
    query["options"] = f"-csearch_path={schema},public"
    isolated_database_url = database_url_object.set(query=query).render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", isolated_database_url)

    for module_name in ("main", "database", "models", "semantic_cache", "embeddings", "reranker"):
        sys.modules.pop(module_name, None)

    client = None
    database = None
    semantic_cache = None
    try:
        import database
        import main
        import models
        import semantic_cache
        from fastapi.testclient import TestClient

        database.init_db()
        session = database.SessionLocal()
        organizations = [
            models.Organization(
                id=tenant_ids[0],
                name=f"Integration A {suffix}",
                slug=f"integration-a-{suffix}",
            ),
            models.Organization(
                id=tenant_ids[1],
                name=f"Integration B {suffix}",
                slug=f"integration-b-{suffix}",
            ),
        ]
        session.add_all(organizations)
        session.commit()
        users = [
            models.User(
                email=f"integration-a-{suffix}@example.com",
                password_hash="unused-integration-password",
                organization_id=tenant_ids[0],
                role="member",
            ),
            models.User(
                email=f"integration-b-{suffix}@example.com",
                password_hash="unused-integration-password",
                organization_id=tenant_ids[1],
                role="member",
            ),
        ]
        session.add_all(users)
        session.commit()

        raw_tokens = [f"tenant_{uuid.uuid4().hex}" for _ in organizations]
        tokens = [
            models.APIToken(
                token_hash=main.hash_token(raw_token),
                user_id=user.id,
                organization_id=tenant_id,
                role="member",
            )
            for raw_token, user, tenant_id in zip(raw_tokens, users, tenant_ids)
        ]
        session.add_all(tokens)
        session.commit()
        session.close()

        vector = [1.0] + [0.0] * (models.EMBEDDING_DIM - 1)
        main.get_embedding = lambda text_value: vector
        main.generate_answer = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("The integration cache tests must not call the LLM")
        )
        main.USE_JEV = False
        client = TestClient(main.app)

        services = SimpleNamespace(
            main=main,
            models=models,
            database=database,
            semantic_cache=semantic_cache,
            session_factory=database.SessionLocal,
            client=client,
            tenant_a=str(tenant_ids[0]),
            tenant_b=str(tenant_ids[1]),
            headers_a={"Authorization": f"Bearer {raw_tokens[0]}"},
            headers_b={"Authorization": f"Bearer {raw_tokens[1]}"},
            vector=vector,
        )
        yield services
    finally:
        if client is not None:
            client.close()
        if semantic_cache is not None:
            keys = []
            for tenant_id in tenant_ids:
                keys.extend(semantic_cache.redis_client.scan_iter(match=f"rag:semantic-cache:{tenant_id}:*"))
                keys.extend(semantic_cache.redis_client.scan_iter(match=f"rag:corpus-version:{tenant_id}"))
                keys.extend(semantic_cache.redis_client.scan_iter(match=f"rag:rate-limit:{tenant_id}:*"))
            if keys:
                semantic_cache.redis_client.delete(*set(keys))
        if database is not None:
            database.engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin_engine.dispose()