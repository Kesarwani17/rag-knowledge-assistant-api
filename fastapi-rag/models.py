"""SQLAlchemy models for source documents and their vectorized chunks."""

from sqlalchemy import Column, Integer, String, Text, DateTime, func
from pgvector.sqlalchemy import Vector
from database import Base

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class Document(Base):
    """A source document submitted to the knowledge base."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    tenant_id = Column(String, nullable=False, index=True)


class DocumentChunk(Base):
    """A chunk of a source document with its embedding vector."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM))
    tenant_id = Column(String, nullable=False, index=True)