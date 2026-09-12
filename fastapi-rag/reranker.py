"""Local CrossEncoder reranking for retrieved RAG context."""

from sentence_transformers import CrossEncoder

# Downloads a tiny, fast model locally (~80MB) the first time it runs.
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Sort retrieved chunks by CrossEncoder relevance to the query."""
    if not chunks:
        return []

    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = reranker_model.predict(pairs)

    for index, score in enumerate(scores):
        chunks[index]["rerank_score"] = float(score)

    return sorted(chunks, key=lambda chunk: chunk["rerank_score"], reverse=True)