"""FastAPI application exposing document ingestion, retrieval, and RAG answers."""

import os

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import text, select

from llm import generate_answer, RAGResponse
from database import Base, SessionLocal, engine
import models
from chunking import chunk_text
from embeddings import get_embedding

load_dotenv()

# Enable pgvector extension
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAG Learning API")
document_statuses: dict[int, str] = {}


class DocumentIn(BaseModel):
    """Request body for storing a source document."""

    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)

class SearchQuery(BaseModel):
    """Request body for semantic document search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""
    return {"status": "ok"}


@app.post("/documents")
def create_document(doc: DocumentIn) -> dict[str, object]:
    """Store a source document and return its generated identifier."""
    db = SessionLocal()
    
    useMockTenant = os.getenv("USE_MOCK_TENANT")
    if useMockTenant is True:
        print("Using mock tenant ID for document creation.")
        tenant_id = os.getenv("MOCK_TENANT_ID")
        if not tenant_id:
            raise RuntimeError("MOCK_TENANT_ID must be set")
    else:
        tenant_id = doc.tenant_id

    try:
        new_doc = models.Document(title=doc.title, content=doc.content, tenant_id=tenant_id)
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        return {"id": new_doc.id, "title": new_doc.title, "status": "stored"}
    finally:
        db.close()


@app.get("/documents")
def list_documents() -> list[dict[str, object]]:
    """Return all stored source documents."""
    db = SessionLocal()
    try:
        docs = db.query(models.Document).all()
        return [{"id": d.id, "title": d.title, "content": d.content} for d in docs]
    finally:
        db.close()


def process_heavy_document(doc_id: int) -> None:
    """Chunk and embed a stored document in a background task."""
    db = SessionLocal()
    try:
        doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
        if not doc:
            return

        chunks = chunk_text(doc.content)

        for index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            db.add(
                models.DocumentChunk(
                    document_id=doc.id,
                    chunk_index=index,
                    chunk_text=chunk,
                    embedding=embedding,
                )
            )

        db.commit()
        document_statuses[doc_id] = "completed"
    finally:
        db.close()


@app.post("/documents/{doc_id}/process")
def trigger_process(doc_id: int, background_tasks: BackgroundTasks) -> dict[str, object]:
    """Queue document processing and return before embedding work begins."""
    document_statuses[doc_id] = "processing"
    background_tasks.add_task(process_heavy_document, doc_id)
    return {
        "status": "processing",
        "doc_id": doc_id,
        "message": "Check status later",
    }


@app.get("/documents/{doc_id}/status")
def get_document_status(doc_id: int) -> dict[str, str]:
    """Return the current background processing status for a document."""
    return {"status": document_statuses.get(doc_id, "processing")}


@app.post("/search")
def search_documents(req: SearchQuery) -> list[dict[str, object]]:
    """Return the closest document chunks for a semantic query."""
    db = SessionLocal()
    try:
        # 1. Turn the user's question into a vector
        query_embedding = get_embedding(req.query)

        # 2. Use the built-in pgvector column method for cosine distance
        stmt = (
            select(
                models.DocumentChunk,
                models.DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
            )
            .order_by("distance")
            .limit(req.top_k)
        )

        results = db.execute(stmt).all()

        # 3. Return the matched text chunks
        return [
            {
                "text": chunk.chunk_text,
                "document_id": chunk.document_id,
                "similarity_score": round(1.0 - float(dist), 4)
            }
            for chunk, dist in results
        ]
    finally:
        db.close()

class AskQuery(BaseModel):
    """Request body for a guarded question-answering request."""

    query: str = Field(..., min_length=1)

@app.post("/ask", response_model=RAGResponse)
def ask_question(req: AskQuery) -> RAGResponse:
    """Retrieve relevant chunks and generate a grounded structured answer."""
    db = SessionLocal()
    try:
        # 1. RETRIEVAL: Get the top 3 most relevant chunks
        query_embedding = get_embedding(req.query)
        stmt = (
            select(
                models.DocumentChunk,
                models.DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
            )
            .where(models.DocumentChunk.tenant_id == os.getenv("MOCK_TENANT_ID"))
            .order_by("distance")
            .limit(3)
        )
        results = db.execute(stmt).all()

        chunks = []

        for c, distance in results:
            similarity = 1.0 - float(distance)

            # Ignore weak/irrelevant matches
            if similarity >= 0.35:
                chunks.append({
                    "text": c.chunk_text,
                    "document_id": c.document_id,
                    "similarity_score": round(similarity, 4)
                })

        if not chunks:
            return RAGResponse(
                answer="I cannot answer this based on the provided documents.",
                is_hallucination=True,
                source_document_ids=[]
            )
        # 2. GENERATION: Send chunks to Groq LLM with Pydantic Guardrails
        llm_response = generate_answer(req.query, chunks)

        # --- SENIOR FIX: Force the correct IDs from the database ---
        # Don't rely on the LLM to remember IDs. We inject the exact ones we retrieved.
        llm_response.source_document_ids = list(set(c["document_id"] for c in chunks))

        return llm_response

    finally:
        db.close()