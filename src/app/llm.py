"""LLM client wrapper. Handles all communication with Gemini."""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


API_KEY = os.getenv("GOOGLE_API_KEY")

# Only create the client if the key exists. Tests can import this module
# without a real key — they only need _build_system_prompt etc.
client = genai.Client(api_key=API_KEY) if API_KEY else None


MODEL_NAME = "gemini-2.5-flash-lite"

# Base system prompt — controls bot behavior across the whole conversation
BASE_SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant. "
    "Keep replies clear and concise. "
    "When the user shares personal details (like their name), remember them."
)


def _build_system_prompt(context: str | None) -> str:
    if not context:
        return BASE_SYSTEM_PROMPT

    return (
        BASE_SYSTEM_PROMPT
        + "\n\nYou have been given context to help answer the user's question. "
        + "The context may come from the user's documents (marked 'FROM YOUR DOCUMENTS') "
        + "and/or from a live web search (marked 'FROM THE WEB').\n\n"
        + "Instructions:\n"
        + "1. Use the provided context as your primary source.\n"
        + "2. If multiple sources are provided, synthesize them into one clear answer. "
        + "Mention briefly where key facts came from (e.g., 'According to your documents...' or 'Recent web sources indicate...').\n"
        + "3. If the sources conflict, point this out instead of picking one silently.\n"
        + "4. If the context doesn't contain the answer, say so honestly — don't guess.\n"
        + "5. For general/casual questions not related to the context, answer normally from your own knowledge.\n\n"
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

    if client is None:
        raise ValueError(
            "GOOGLE_API_KEY not found. Make sure your .env file exists and has the key."
        ) 

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