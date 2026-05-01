import chainlit as cl 

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Hi! I'm your future AI assistant. Right now I just echo. Try saying something!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    reply = f"You said: '{message.content}'. Got it! 🎉"
    await cl.Message(content=reply).send()