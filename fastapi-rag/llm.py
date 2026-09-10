"""Groq-backed structured answer generation for retrieved RAG context."""

import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Groq uses the exact same SDK as OpenAI, we just change the base_url!
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# --- RESUME FLEX: Structured Outputs & Hallucination Guards ---
class RAGResponse(BaseModel):
    """Validated answer payload returned by the guarded LLM call."""

    answer: str = Field(description="The answer to the user's question based STRICTLY on the context.")
    is_hallucination: bool = Field(description="True if the context does NOT contain the answer. False if it does.")
    source_document_ids: list[int] = Field(default_factory=list, description="List of document IDs used to generate the answer.")


def generate_answer(query: str, context_chunks: list[dict]) -> RAGResponse:
    """Generate a schema-validated answer using only retrieved context chunks."""
    # Format the retrieved chunks into a single text block
    context_text = "\n\n".join([
        f"[Doc ID {c['document_id']}]: {c['text']}" for c in context_chunks
    ])

    system_prompt = """
    You are a strict, production-grade RAG assistant.
    Answer the user's question using ONLY the provided context.
    You MUST respond in JSON format matching the provided schema.
    If the answer is not explicitly in the context, you MUST set 'is_hallucination' to True
    and set the 'answer' to "I cannot answer this based on the provided documents."
    """

    user_prompt = f"""
    Context:
    {context_text}

    User Question: {query}
    """

        # This forces the LLM to output valid JSON that matches our Pydantic model
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={ "type": "json_object" }, # Standard JSON mode
        temperature=0.0
    )

    # Manually parse the JSON string into our Pydantic model
    import json
    raw_json = completion.choices[0].message.content
    return RAGResponse.model_validate_json(raw_json)