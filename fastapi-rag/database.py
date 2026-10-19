"""Database engine, session, and metadata setup for the RAG backend."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from token_security import hash_token

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set")

# Connection pool config via env (production-ready defaults)
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

engine = create_engine(
    DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_pre_ping=True,  # validates connections before use
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def init_db() -> None:
    """Create the fresh database schema and enable pgvector for the app."""
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    Base.metadata.create_all(bind=engine)
    migrate_api_token_storage()
    print("Database connected")
    print("pgvector extension enabled")


def migrate_api_token_storage() -> None:
    """Hash legacy bearer tokens and remove the plaintext column."""
    with engine.begin() as conn:
        columns = {column["name"] for column in inspect(conn).get_columns("api_tokens")}
        if "token_hash" not in columns:
            conn.execute(text("ALTER TABLE api_tokens ADD COLUMN token_hash VARCHAR(64)"))

        if "token" in columns:
            legacy_tokens = conn.execute(
                text("SELECT id, token FROM api_tokens WHERE token_hash IS NULL")
            ).all()
            for token_id, token_value in legacy_tokens:
                conn.execute(
                    text("UPDATE api_tokens SET token_hash = :token_hash WHERE id = :token_id"),
                    {"token_hash": hash_token(token_value), "token_id": token_id},
                )
            conn.execute(text("ALTER TABLE api_tokens DROP COLUMN token"))

        conn.execute(text("ALTER TABLE api_tokens ALTER COLUMN token_hash SET NOT NULL"))
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_api_tokens_token_hash "
                "ON api_tokens (token_hash)"
            )
        )


if __name__ == "__main__":
    init_db()