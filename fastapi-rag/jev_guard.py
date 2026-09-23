"""Jev is a System-1 model returning typed, calibrated decisions.

This guard sits between retrieval and generation so out-of-scope queries are
refused before the expensive LLM call. It uses fail-open semantics.
"""

import os

import requests


USE_JEV = os.getenv("USE_JEV", "false").lower() == "true"
JEV_URL = os.getenv("JEV_API_URL", "https://api.typesafe.ai/v1/decide")
CONFIDENCE_THRESHOLD = float(os.getenv("JEV_CONFIDENCE_THRESHOLD", "0.9"))


def is_answerable(query: str, context_text: str) -> tuple[bool, float]:
    """Ask Jev whether a query is answerable from the provided context.

    Args:
        query: The user's question.
        context_text: The retrieved context available to the answerer.

    Returns:
        A tuple containing the answerability decision and its probability.
        Failures return an answerable decision with zero confidence.
    """
    try:
        # Request schema is a placeholder pending TypeSafe docs; only this function changes when the real schema arrives.
        response = requests.post(
            JEV_URL,
            timeout=2.0,
            headers={"Authorization": f"Bearer {os.getenv('JEV_API_KEY', '')}"},
            json={
                "model": "jev",
                "question": "Is the user's question answerable using ONLY the provided context?",
                "choices": ["yes", "no"],
                "input": {"question": query, "context": context_text},
            },
        )
        data = response.json()
        decision = data["decision"]
        probability = float(data["probability"])
        return decision == "yes", probability
    except Exception:
        return True, 0.0