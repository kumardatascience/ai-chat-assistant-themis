"""Day 9 — Chainlit chatbot with intelligent routing + reasoning steps in UI."""

import chainlit as cl
from router import ChatRouter


# Create one workflow instance to reuse across all chats
workflow = ChatRouter(timeout=60)


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Hi! I'm an intelligent assistant. I can answer from your documents, "
                "search the web, or just chat. Watch the reasoning steps as I work!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # Workflow emits its own Chainlit steps; just await the final result
    result = await workflow.run(question=message.content, history=history)

    # Stream the LLM reply
    reply = cl.Message(content="")
    full_response = ""
    async for chunk in result.stream:
        full_response += chunk
        await reply.stream_token(chunk)
    await reply.send()

    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)