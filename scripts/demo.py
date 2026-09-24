"""End-to-end demo proving every RAG pipeline stage works.
Run AFTER seeding: python scripts/demo.py
"""
import time
import requests

API = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 120


def timed_ask(label: str, query: str) -> dict:
    print(f"\n{'='*64}\n>> {label}\n  Query: {query!r}\n{'-'*64}")
    start = time.perf_counter()
    res = requests.post(
        f"{API}/ask",
        json={"query": query},
        timeout=REQUEST_TIMEOUT,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    res.raise_for_status()
    data = res.json()
    answer = data['answer'][:160].encode('ascii', 'replace').decode('ascii')
    print(f"  Answer      : {answer}...")
    print(f"  Hallucination: {data['is_hallucination']}")
    print(f"  Sources      : {data['source_document_ids']}")
    print(f"  [TIME]  Latency    : {elapsed_ms:.0f} ms")
    return data


def main() -> None:
    # 0. Health
    print("Checking API health...")
    health = requests.get(f"{API}/health", timeout=REQUEST_TIMEOUT)
    health.raise_for_status()
    assert health.json()["status"] == "ok", "API not healthy"
    print("[OK] API is up\n")

    password_query = "How should I store passwords securely?"

    # 1. First ask: retrieval, reranking, and generation unless already cached.
    first_result = timed_ask(
        "STAGE 1: Full pipeline (first call; cache may already exist)",
        password_query,
    )

    # 2. Exact repeat is eligible for a semantic cache hit.
    second_result = timed_ask(
        "STAGE 2: Semantic cache HIT expected (exact repeat)",
        password_query,
    )
    assert second_result == first_result, "Repeated query returned a different cached response"

    # 3. HALLUCINATION GUARD — out-of-scope question
    guard_result = timed_ask(
        "STAGE 3: Hallucination guard (out-of-scope, expect is_hallucination=True)",
        "Who won the 2024 cricket world cup?",
    )
    assert guard_result["is_hallucination"] is True
    assert guard_result["source_document_ids"] == []

    # 4. SERVER-SIDE CITATIONS — sources injected from DB, not LLM
    citation_result = timed_ask(
        "STAGE 4: Server-side citations (check source_document_ids populated)",
        "How do I prevent CSRF attacks?",
    )
    assert citation_result["is_hallucination"] is False
    assert citation_result["source_document_ids"]

    print(f"\n{'='*64}\n[DONE] Demo complete. Stage 2 returned the same response as Stage 1.\n{'='*64}")


if __name__ == "__main__":
    main()