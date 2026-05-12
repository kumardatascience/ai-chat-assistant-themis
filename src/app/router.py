"""LlamaIndex Workflow that routes user queries to the right knowledge source."""

import asyncio
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
    """Custom stop event so the framework sees a typed StopEvent return."""
    stream: object       # the async generator from stream_response
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
        question = ev.question
        history = ev.history

        prompt = ROUTER_PROMPT.format(question=question)

        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        raw = (response.text or "").strip().lower().split()
        decision = raw[0] if raw else "direct"
        if decision not in {"direct", "rag", "web", "multi"}:
            decision = "direct"

        print(f"🚦 Router decision: {decision}")
        return RouteDecision(decision=decision, question=question, history=history)

    @step
    async def gather_context(self, ev: RouteDecision) -> ContextReady:
        context_parts = []

        if ev.decision in ("rag", "multi"):
            print("📚 Fetching from documents...")
            rag_chunks = retrieve_context(ev.question, top_k=3)
            if rag_chunks:
                context_parts.append("--- FROM YOUR DOCUMENTS ---\n" + rag_chunks)

        if ev.decision in ("web", "multi"):
            print("🌐 Searching the web...")
            web_results = await asyncio.to_thread(search_web, ev.question, 3)
            if web_results:
                context_parts.append("--- FROM THE WEB ---\n" + web_results)

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