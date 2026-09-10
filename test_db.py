"""Manual database connectivity check for the local pgvector development stack."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv("fastapi-rag/.env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set")

engine = create_engine(DATABASE_URL)

def init_db() -> None:
    """Enable pgvector and verify that PostgreSQL accepts a connection."""
    with engine.connect() as conn:
        # This command activates the vector database capabilities
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        conn.commit()
    print("SUCCESS: Connected to PostgreSQL and pgvector extension is enabled!")

if __name__ == "__main__":
    init_db()