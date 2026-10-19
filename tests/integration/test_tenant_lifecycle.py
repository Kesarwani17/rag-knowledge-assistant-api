"""Real PostgreSQL/pgvector and Redis integration tests for tenant workflows."""

import pytest

from llm import RAGResponse


pytestmark = pytest.mark.integration


def create_and_process_document(services, headers, title, content):
    created = services.client.post(
        "/documents",
        json={"title": title, "content": content},
        headers=headers,
    )
    assert created.status_code == 200
    document_id = created.json()["id"]
    processing = services.client.post(f"/documents/{document_id}/process", headers=headers)
    assert processing.status_code == 200
    assert services.client.get(f"/documents/{document_id}/status", headers=headers).json() == {
        "status": "completed"
    }
    return document_id


def test_real_sql_tenant_isolation_and_private_document_ids(integration_services):
    """PostgreSQL filters every read and mutation by the authenticated tenant."""
    services = integration_services
    document_id = create_and_process_document(
        services,
        services.headers_a,
        "Tenant A policy",
        "alpha-only integration policy",
    )

    tenant_a_documents = services.client.get("/documents", headers=services.headers_a)
    tenant_b_documents = services.client.get("/documents", headers=services.headers_b)
    tenant_b_search = services.client.post(
        "/search",
        json={"query": "alpha-only"},
        headers=services.headers_b,
    )
    assert [document["id"] for document in tenant_a_documents.json()] == [document_id]
    assert tenant_b_documents.json() == []
    assert tenant_b_search.json() == []

    assert services.client.get(
        f"/documents/{document_id}/status", headers=services.headers_b
    ).status_code == 404
    assert services.client.post(
        f"/documents/{document_id}/process", headers=services.headers_b
    ).status_code == 404
    assert services.client.put(
        f"/documents/{document_id}",
        json={"title": "Unauthorized", "content": "not allowed"},
        headers=services.headers_b,
    ).status_code == 404
    assert services.client.delete(
        f"/documents/{document_id}", headers=services.headers_b
    ).status_code == 404


def test_real_update_delete_and_redis_cache_invalidation(integration_services):
    """PUT clears old SQL chunks and Redis answers; DELETE removes new chunks."""
    services = integration_services
    document_id = create_and_process_document(
        services,
        services.headers_a,
        "Original policy",
        "old-content-token alpha",
    )
    original_search = services.client.post(
        "/search",
        json={"query": "old-content-token"},
        headers=services.headers_a,
    )
    assert original_search.json()

    stale_response = RAGResponse(answer="stale tenant answer", is_hallucination=False)
    services.semantic_cache.cache_response(services.tenant_a, services.vector, stale_response)
    assert services.client.post(
        "/ask", json={"query": "cached question"}, headers=services.headers_a
    ).json()["answer"] == "stale tenant answer"
    assert services.client.post(
        "/ask", json={"query": "cached question"}, headers=services.headers_b
    ).json()["answer"] != "stale tenant answer"

    updated = services.client.put(
        f"/documents/{document_id}",
        json={"title": "Updated policy", "content": "new-content-token beta"},
        headers=services.headers_a,
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "pending"
    assert services.semantic_cache.get_cached_response(services.tenant_a, services.vector) is None
    post_update_answer = services.client.post(
        "/ask", json={"query": "cached question"}, headers=services.headers_a
    )
    assert post_update_answer.status_code == 200
    assert post_update_answer.json()["answer"] != "stale tenant answer"
    assert services.client.post(
        "/search",
        json={"query": "old-content-token"},
        headers=services.headers_a,
    ).json() == []

    assert services.client.post(
        f"/documents/{document_id}/process", headers=services.headers_a
    ).status_code == 200
    updated_search = services.client.post(
        "/search",
        json={"query": "new-content-token"},
        headers=services.headers_a,
    )
    assert updated_search.json()

    deleted = services.client.delete(f"/documents/{document_id}", headers=services.headers_a)
    assert deleted.status_code == 200
    assert services.client.post(
        "/search",
        json={"query": "new-content-token"},
        headers=services.headers_a,
    ).json() == []