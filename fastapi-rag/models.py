"""SQLAlchemy models for source documents and their vectorized chunks."""

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR
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
    status = Column(String, nullable=False, default="pending", server_default="pending")


class Organization(Base):
    """Top-level tenant boundary for product-ready multi-tenancy."""

    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(120), nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    """Authenticated user that belongs to exactly one organization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    organization_id = Column(Integer, nullable=False, index=True)
    role = Column(String(64), nullable=False, default="member", server_default="member")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class APIToken(Base):
    """Bearer token issued to tenant users."""

    __tablename__ = "api_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    organization_id = Column(Integer, nullable=False, index=True)
    role = Column(String(64), nullable=False, default="member", server_default="member")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DocumentChunk(Base):
    """A chunk of a source document with its embedding vector."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM))
    tenant_id = Column(String, nullable=False, index=True)
    search_vector = Column(TSVECTOR)

    __table_args__ = (
        Index(
            "ix_document_chunks_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )