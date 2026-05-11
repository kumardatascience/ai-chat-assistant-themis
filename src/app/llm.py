"""LLM client wrapper. Handles all communication with Gemini."""

import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env file into the environment
load_dotenv()

# Read the API key from the environment
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Make sure your .env file exists and has the key."
    )

# Create one client we'll reuse for every request
client = genai.Client(api_key=API_KEY)

# The model we'll use. Gemini 2.5 Flash = fast, cheap, smart enough for chat.
MODEL_NAME = "gemini-2.5-flash"


async def stream_response(user_message: str):
    """
    Send the user's message to Gemini and yield the response in chunks.

    'yield' instead of 'return' makes this a generator — it produces
    pieces of the response one at a time as Gemini streams them.
    """
    response_stream = await client.aio.models.generate_content_stream(
        model=MODEL_NAME,
        contents=user_message,
    )

    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text