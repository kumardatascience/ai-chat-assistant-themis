"""Smoke tests — basic sanity checks. No LLM calls, no network."""

import pytest


def test_python_works():
    """The most basic test: does Python even run?"""
    assert 1 + 1 == 2


def test_string_operations():
    """Test the cleanup logic our router uses on Gemini's response."""
    raw = "  RAG  "
    cleaned = raw.strip().lower().split()
    decision = cleaned[0] if cleaned else "direct"
    assert decision == "rag"


def test_decision_fallback():
    """If Gemini returns garbage, we should fall back to 'direct'."""
    raw = "🤖 maybe both?"
    cleaned = raw.strip().lower().split()
    decision = cleaned[0] if cleaned else "direct"
    if decision not in {"direct", "rag", "web", "multi"}:
        decision = "direct"
    assert decision == "direct"


def test_empty_response_fallback():
    """If Gemini returns nothing, fall back to 'direct'."""
    raw = ""
    cleaned = raw.strip().lower().split()
    decision = cleaned[0] if cleaned else "direct"
    assert decision == "direct"

# --- Tests for real project code ---

import sys
from pathlib import Path

# Add src/app to Python's import path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))

from llm import _build_system_prompt, BASE_SYSTEM_PROMPT


def test_system_prompt_without_context():
    """No context → return base prompt unchanged."""
    result = _build_system_prompt(None)
    assert result == BASE_SYSTEM_PROMPT


def test_system_prompt_with_empty_context():
    """Empty string context → also return base prompt."""
    result = _build_system_prompt("")
    assert result == BASE_SYSTEM_PROMPT


def test_system_prompt_with_context_includes_markers():
    """Context provided → should include start/end markers."""
    fake_context = "Themis charges $5000 per project."
    result = _build_system_prompt(fake_context)
    assert "=== CONTEXT START ===" in result
    assert "=== CONTEXT END ===" in result
    assert fake_context in result


def test_system_prompt_with_context_contains_instructions():
    """The synthesis instructions should be present."""
    result = _build_system_prompt("some context")
    # Spot check: a few key instruction phrases
    assert "FROM YOUR DOCUMENTS" in result
    assert "FROM THE WEB" in result
    assert "conflict" in result.lower()    