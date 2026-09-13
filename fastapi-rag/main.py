"""FastAPI application exposing document ingestion, retrieval, and RAG answers."""

import os

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import desc, func, select, text

from llm import generate_answer, RAGResponse
from database import Base, SessionLocal, engine, init_db
import models
from chunking import chunk_text
from embeddings import get_embedding
from reranker import rerank_chunks
from semantic_cache import cache_response, get_cached_response

load_dotenv()

init_db()

app = FastAPI(title="RAG Learning API")


def get_current_tenant_id() -> str:
    """Return the configured tenant identifier for the current deployment."""
    return os.getenv("MOCK_TENANT_ID", "default")


class DocumentIn(BaseModel):
    """Request body for storing a source document."""

    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)

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
    
    useMockTenant = os.getenv("USE_MOCK_TENANT", "false").lower() == "true"
    if useMockTenant:
        tenant_id = os.getenv("MOCK_TENANT_ID")
        if not tenant_id:
            raise RuntimeError("MOCK_TENANT_ID must be set")
    else:
        tenant_id = doc.tenant_id # Now this works!

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
    doc = None
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
                    tenant_id=doc.tenant_id,
                    search_vector=func.to_tsvector("english", chunk),
                )
            )

        db.commit()
        doc.status = "completed"
        db.commit()
    except Exception as exc:
        db.rollback()
        if doc is None:
            doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
        print(f"Error processing doc {doc_id}: {exc}")
    finally:
        db.close()


@app.post("/documents/{doc_id}/process")
def trigger_process(doc_id: int, background_tasks: BackgroundTasks) -> dict[str, object]:
    """Queue document processing and return before embedding work begins."""
    db = SessionLocal()
    try:
        doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        doc.status = "processing"
        db.commit()
        background_tasks.add_task(process_heavy_document, doc_id)
        return {
            "status": "processing",
            "doc_id": doc_id,
            "message": "Embedding started. Check /documents/{doc_id}/status for updates.",
        }
    finally:
        db.close()


@app.get("/documents/{doc_id}/status")
def get_document_status(doc_id: int) -> dict[str, str]:
    """Return the current background processing status for a document."""
    db = SessionLocal()
    try:
        doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": doc.status}
    finally:
        db.close()


@app.post("/search")
def search_documents(req: SearchQuery) -> list[dict[str, object]]:
    """Combine dense vector and sparse keyword retrieval for a query."""
    db = SessionLocal()
    try:
        tenant_id = get_current_tenant_id()
        # Dense retrieval captures semantic meaning.
        query_embedding = get_embedding(req.query)
        vector_stmt = (
            select(
                models.DocumentChunk,
                models.DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
            )
            .where(models.DocumentChunk.tenant_id == tenant_id)
            .order_by("distance")
            .limit(req.top_k)
        )
        vector_results = db.execute(vector_stmt).all()

        # Sparse retrieval preserves exact entities such as error codes and SKUs.
        keyword_stmt = (
            select(
                models.DocumentChunk,
                func.ts_rank(
                    models.DocumentChunk.search_vector,
                    func.plainto_tsquery("english", req.query),
                ).label("keyword_score"),
            )
            .where(
                models.DocumentChunk.tenant_id == tenant_id,
                models.DocumentChunk.search_vector.match(req.query),
            )
            .order_by(desc("keyword_score"))
            .limit(req.top_k)
        )
        keyword_results = db.execute(keyword_stmt).all()

        combined_results: dict[int, dict[str, object]] = {}
        for chunk, distance in vector_results:
            combined_results[chunk.id] = {
                "text": chunk.chunk_text,
                "document_id": chunk.document_id,
                "similarity_score": round(1.0 - float(distance), 4),
            }

        for chunk, keyword_score in keyword_results:
            combined_results.setdefault(
                chunk.id,
                {
                    "text": chunk.chunk_text,
                    "document_id": chunk.document_id,
                    "similarity_score": round(float(keyword_score), 4),
                },
            )

        return list(combined_results.values())[: req.top_k]
    finally:
        db.close()

class AskQuery(BaseModel):
    """Request body for a guarded question-answering request."""

    query: str = Field(..., min_length=1)

@app.post("/ask", response_model=RAGResponse)
def ask_question(req: AskQuery) -> RAGResponse:
    """Retrieve, rerank, and generate a grounded answer from relevant chunks."""
    db = SessionLocal()
    try:
        # 1. RETRIEVAL: Get a broad candidate set for local reranking
        query_embedding = get_embedding(req.query)
        tenant_id = get_current_tenant_id()
        cached_response = get_cached_response(tenant_id, query_embedding)
        if cached_response:
            return cached_response

        stmt = (
            select(
                models.DocumentChunk,
                models.DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
            )
            .where(models.DocumentChunk.tenant_id == tenant_id)
            .order_by("distance")
            .limit(15)
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
        # 2. RERANKING: Let the local CrossEncoder select the best context.
        chunks = rerank_chunks(req.query, chunks)[:3]

        # 3. GENERATION: Send only the top 3 chunks to Groq with guardrails
        llm_response = generate_answer(req.query, chunks)

        # --- SENIOR FIX: Force the correct IDs from the database ---
        # Don't rely on the LLM to remember IDs. We inject the exact ones we retrieved.
        llm_response.source_document_ids = list(set(c["document_id"] for c in chunks))
        cache_response(tenant_id, query_embedding, llm_response)

        return llm_response

    finally:
        db.close()