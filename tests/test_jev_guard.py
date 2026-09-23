"""Offline tests for the Jev answerability guard."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1] / "fastapi-rag"
sys.path.insert(0, str(PROJECT_DIR))

import jev_guard


class FakeResponse:
    """Small HTTP response substitute for Jev decisions."""

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def test_refuses_high_confidence_not_answerable(monkeypatch):
    """A high-confidence Jev refusal should be returned as not answerable."""
    monkeypatch.setattr(
        jev_guard.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"decision": "no", "probability": 0.97}),
    )

    assert jev_guard.is_answerable("What is the weather?", "Policy text") == (False, 0.97)


def test_answerable_yes(monkeypatch):
    """A Jev yes decision should preserve its probability."""
    monkeypatch.setattr(
        jev_guard.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"decision": "yes", "probability": 0.8}),
    )

    assert jev_guard.is_answerable("What is the policy?", "Policy text") == (True, 0.8)


def test_fail_open_on_error(monkeypatch):
    """HTTP failures should allow the request to continue with zero confidence."""
    def raise_error(*args, **kwargs):
        raise RuntimeError("Jev unavailable")

    monkeypatch.setattr(jev_guard.requests, "post", raise_error)

    assert jev_guard.is_answerable("Question", "Context") == (True, 0.0)