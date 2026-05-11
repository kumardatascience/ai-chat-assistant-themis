"""Day 5 — Chainlit chatbot with Gemini + chat memory + RAG."""

import chainlit as cl
from llm import stream_response
from rag import retrieve_context


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Hi! I'm powered by Gemini and I can answer questions about your documents. Ask away!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # 🔍 Retrieve relevant chunks from ChromaDB
    context = retrieve_context(message.content, top_k=3)

    # Stream Gemini's reply, grounded in the retrieved context
    reply = cl.Message(content="")
    full_response = ""
    async for chunk in stream_response(history, context=context):
        full_response += chunk
        await reply.stream_token(chunk)
    await reply.send()

    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)