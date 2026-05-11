"""LLM client wrapper. Handles all communication with Gemini."""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Make sure your .env file exists and has the key."
    )

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

# Base system prompt — controls bot behavior across the whole conversation
BASE_SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant. "
    "Keep replies clear and concise. "
    "When the user shares personal details (like their name), remember them."
)


def _build_system_prompt(context: str | None) -> str:
    """If context is provided, append it to the system prompt as grounding."""
    if not context:
        return BASE_SYSTEM_PROMPT 
    return (
    BASE_SYSTEM_PROMPT
    + "\n\nThe following context was retrieved from the user's documents. "
    + "If the user's question is about these documents, use this context "
    + "as your primary source. If the documents don't contain the answer, "
    + "say so honestly. For general questions unrelated to the documents "
    + "(e.g., math, definitions, coding help, casual conversation), "
    + "answer normally using your own knowledge.\n\n"
    + "=== CONTEXT START ===\n"
    + context
    + "\n=== CONTEXT END ==="
)


async def stream_response(history: list[dict], context: str | None = None):
    """
    Send the full chat history to Gemini and stream the reply.

    history: list of {"role": "user" | "assistant", "content": "..."}
    context: optional retrieved document chunks to ground the answer
    """
    gemini_contents = [
        types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part.from_text(text=msg["content"])],
        )
        for msg in history
    ]

    response_stream = await client.aio.models.generate_content_stream(
        model=MODEL_NAME,
        contents=gemini_contents,
        config=types.GenerateContentConfig(
            system_instruction=_build_system_prompt(context),
        ),
    )

    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text