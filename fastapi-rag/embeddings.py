"""Local sentence-transformer embedding generation for RAG content."""

from sentence_transformers import SentenceTransformer

model: SentenceTransformer | None = None

def get_embedding(text: str) -> list[float]:
    """Encode text into a 384-dimensional local embedding vector."""
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(text)
    return embedding.tolist()