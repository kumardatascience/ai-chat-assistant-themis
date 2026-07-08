"""Chainlit chatbot with intelligent routing + live file upload support."""

import chainlit as cl
from router import ChatRouter
from rag import index_file_for_session, clear_session


workflow = ChatRouter(timeout=60)


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Hi! I'm an intelligent assistant. Ask me anything, or upload a "
                "PDF/text file and ask questions about it. Watch the reasoning steps as I work!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    session_id = cl.user_session.get("id") or "default"

    # 📎 Handle any uploaded files
    if message.elements:
        for element in message.elements:
            file_path = getattr(element, "path", None)
            if not file_path:
                continue

            async with cl.Step(
                name=f"📥 Indexing {element.name}",
                type="tool",
            ) as step:
                step.input = element.name
                try:
                    chunk_count = index_file_for_session(file_path, session_id)
                    step.output = f"Stored {chunk_count} chunks from **{element.name}**."
                except Exception as e:
                    step.output = f"⚠️ Failed to index {element.name}: {e}"

    history.append({"role": "user", "content": message.content})

    # Run the workflow with the session ID so RAG can find uploaded chunks
    result = await workflow.run(
        question=message.content,
        history=history,
        session_id=session_id,
    )

    # Stream the LLM reply
    reply = cl.Message(content="")
    full_response = ""
    async for chunk in result.stream:
        full_response += chunk
        await reply.stream_token(chunk)
    await reply.send()

    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)


@cl.on_chat_end
async def end():
    """Clean up the session's uploaded documents when the chat ends."""
    session_id = cl.user_session.get("id") or "default"
    clear_session(session_id)