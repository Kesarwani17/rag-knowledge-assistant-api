"""Local CrossEncoder reranking for retrieved RAG context."""

from sentence_transformers import CrossEncoder

reranker_model: CrossEncoder | None = None


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Sort retrieved chunks by CrossEncoder relevance to the query."""
    if not chunks:
        return []

    global reranker_model
    if reranker_model is None:
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = reranker_model.predict(pairs)

    for index, score in enumerate(scores):
        chunks[index]["rerank_score"] = float(score)

    return sorted(chunks, key=lambda chunk: chunk["rerank_score"], reverse=True)