"""Day 7 — Chainlit chatbot with intelligent query routing."""

import chainlit as cl
from router import ChatRouter


# Create one workflow instance to reuse for all chats
workflow = ChatRouter(timeout=60)


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Hi! I'm an intelligent assistant. I can answer from your documents, "
                "search the web, or just chat. I'll decide what's best for each question."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # Run the workflow — it routes, fetches context, and returns a stream
    result = await workflow.run(question=message.content, history=history)

    stream = result.stream
    decision = result.decision

    # Show which path was taken (we'll make this prettier on Day 9)
    route_labels = {
        "direct": "💬 Direct answer",
        "rag": "📚 Searching your documents",
        "web": "🌐 Searching the web",
        "multi": "📚🌐 Using documents + web",
    }
    await cl.Message(content=f"_{route_labels.get(decision, decision)}_").send()

    # Stream the LLM reply
    reply = cl.Message(content="")
    full_response = ""
    async for chunk in stream:
        full_response += chunk
        await reply.stream_token(chunk)
    await reply.send()

    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)