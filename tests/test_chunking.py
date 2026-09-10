"""Behavioral tests for overlapping RAG text chunks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "fastapi-rag"))

from chunking import chunk_text


def test_empty_text_returns_no_chunks():
    """Blank input should not create an embedding workload."""
    assert chunk_text("  ") == []


def test_short_text_returns_one_chunk():
    """Text shorter than the chunk size should remain intact."""
    assert chunk_text("short text") == ["short text"]


def test_long_text_returns_multiple_chunks():
    """Text longer than the chunk size should be split."""
    assert len(chunk_text("a" * 2200, chunk_size=1000, overlap=200)) == 3


def test_consecutive_chunks_share_overlap_content():
    """Adjacent chunks should share the configured overlap region."""
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)
    assert set(chunks[0][-3:]).issubset(chunks[1])