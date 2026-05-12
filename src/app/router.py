"""LlamaIndex Workflow that routes user queries to the right knowledge source.
Emits Chainlit steps so the user sees reasoning in real time."""

import asyncio
import chainlit as cl
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    StartEvent,
    StopEvent,
)

from llm import client, MODEL_NAME, stream_response
from rag import retrieve_context
from web_search import search_web


# --- Events ---

class RouteDecision(Event):
    decision: str
    question: str
    history: list[dict]


class ContextReady(Event):
    context: str | None
    question: str
    history: list[dict]
    decision: str


class ChatStopEvent(StopEvent):
    stream: object
    decision: str


# --- Router prompt ---

ROUTER_PROMPT = """You are a query router for an AI assistant. Classify the user's question into ONE of these categories:

- "direct": General knowledge, math, writing, coding, casual chat, definitions. Doesn't need external info.
- "rag": About uploaded documents (e.g., company policies, contracts, the user's PDFs).
- "web": Needs current/real-time information (news, weather, today's events, recent prices).
- "multi": Needs BOTH documents and current web info (e.g., "compare our pricing to current market rates").

Reply with EXACTLY one word: direct, rag, web, or multi. No punctuation, no explanation.

Question: {question}
"""


# --- The Workflow ---

class ChatRouter(Workflow):

    @step
    async def route_query(self, ev: StartEvent) -> RouteDecision:
        async with cl.Step(name="🚦 Routing the query", type="tool") as cl_step:
            cl_step.input = ev.question

            prompt = ROUTER_PROMPT.format(question=ev.question)
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            raw = (response.text or "").strip().lower().split()
            decision = raw[0] if raw else "direct"
            if decision not in {"direct", "rag", "web", "multi"}:
                decision = "direct"

            cl_step.output = f"Decision: **{decision}**"

        return RouteDecision(decision=decision, question=ev.question, history=ev.history)

    @step
    async def gather_context(self, ev: RouteDecision) -> ContextReady:
        context_parts = []

        if ev.decision in ("rag", "multi"):
            async with cl.Step(name="📚 Retrieving from documents", type="retrieval") as cl_step:
                cl_step.input = ev.question
                rag_chunks = retrieve_context(ev.question, top_k=3)
                if rag_chunks:
                    context_parts.append("--- FROM YOUR DOCUMENTS ---\n" + rag_chunks)
                    cl_step.output = f"Found {len(rag_chunks)} characters of relevant content."
                else:
                    cl_step.output = "No relevant chunks found."

        if ev.decision in ("web", "multi"):
            async with cl.Step(name="🌐 Searching the web", type="tool") as cl_step:
                cl_step.input = ev.question
                web_results = await asyncio.to_thread(search_web, ev.question, 3)
                if web_results:
                    context_parts.append("--- FROM THE WEB ---\n" + web_results)
                    cl_step.output = f"Found {len(web_results)} characters of web content."
                else:
                    cl_step.output = "No web results found."

        context = "\n\n".join(context_parts) if context_parts else None

        return ContextReady(
            context=context,
            question=ev.question,
            history=ev.history,
            decision=ev.decision,
        )

    @step
    async def generate_answer(self, ev: ContextReady) -> ChatStopEvent:
        generator = stream_response(ev.history, context=ev.context)
        return ChatStopEvent(stream=generator, decision=ev.decision)