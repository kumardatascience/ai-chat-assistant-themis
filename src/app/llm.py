"""LLM client wrapper. Handles all communication with Gemini."""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load variables from .env file into the environment
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Make sure your .env file exists and has the key."
    )

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# Tells the model how to behave across the whole conversation
SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant. "
    "Keep replies clear and concise. "
    "When the user shares personal details (like their name), remember them."
)


async def stream_response(history: list[dict]):
    """
    Send the full chat history to Gemini and stream the reply.

    history: list of {"role": "user" | "assistant", "content": "..."}
    """
    # Convert our simple history format into Gemini's expected format
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
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text