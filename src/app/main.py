"""Chainlit chatbot powered by Gemini with streaming."""

import chainlit as cl
from llm import stream_response


@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Hi! I'm now powered by Gemini. Ask me anything!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    # Create an empty message we'll fill up as chunks arrive
    reply = cl.Message(content="")

    # Stream chunks from Gemini and append each one to the message
    async for chunk in stream_response(message.content):
        await reply.stream_token(chunk)

    # Mark the message complete (sends final state to UI)
    await reply.send()