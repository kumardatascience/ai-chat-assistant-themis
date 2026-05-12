"""Tiny workflow demo to understand the pattern. Not used by the chatbot."""

import asyncio
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    StartEvent,
    StopEvent,
)


# 1. Define our custom event types (just data containers)
class GreetingEvent(Event):
    message: str


class ShoutingEvent(Event):
    message: str


# 2. Define the workflow with its steps
class DemoWorkflow(Workflow):

    @step
    async def greet(self, ev: StartEvent) -> GreetingEvent:
        """Step 1: receives the start, emits a greeting."""
        print("👋 Step 1: greeting...")
        return GreetingEvent(message=f"Hello, {ev.name}!")

    @step
    async def shout(self, ev: GreetingEvent) -> ShoutingEvent:
        """Step 2: takes the greeting, shouts it."""
        print("📢 Step 2: shouting...")
        return ShoutingEvent(message=ev.message.upper())

    @step
    async def finish(self, ev: ShoutingEvent) -> StopEvent:
        """Step 3: ends the workflow with the final result."""
        print("🏁 Step 3: done.")
        return StopEvent(result=ev.message)


# 3. Run it
async def main():
    wf = DemoWorkflow()
    result = await wf.run(name="Mihir")
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    asyncio.run(main())