"""Day 3 — Chainlit chatbot with Gemini + chat memory."""

import chainlit as cl
from llm import stream_response


@cl.on_chat_start
async def start():
    # Initialize an empty chat history for this user's session
    cl.user_session.set("history", [])

    await cl.Message(
        content="👋 Hi! I'm now powered by Gemini and I remember our conversation. Try me!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    # Get the existing history (or empty list if new session)
    history = cl.user_session.get("history")

    # Add the user's new message to the history
    history.append({"role": "user", "content": message.content})

    # Create an empty reply we'll stream into
    reply = cl.Message(content="")

    # Collect chunks as they stream in, so we can save the full reply at the end
    full_response = ""
    async for chunk in stream_response(history):
        full_response += chunk
        await reply.stream_token(chunk)