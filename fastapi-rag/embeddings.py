"""Local sentence-transformer embedding generation for RAG content."""

from sentence_transformers import SentenceTransformer

# This downloads a tiny, free model (~80MB) the first time you run it
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> list[float]:
    """Encode text into a 384-dimensional local embedding vector."""
    # Generate the embedding locally (no API key needed!)
    embedding = model.encode(text)
    return embedding.tolist()