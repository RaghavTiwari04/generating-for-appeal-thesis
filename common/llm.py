"""Shared LLM call helpers used by brief generator, message generator, and LLM scorer.

Centralises provider dispatch, JSON extraction from LLM responses, and
API key validation so each module doesn't reimplement these.
"""

from __future__ import annotations

import json
import re

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)


def extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, tolerating code fences and prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    return json.loads(text[start : end + 1])


def call_llm(prompt: str, *, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Call the configured LLM provider and return raw text response."""
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return _call_anthropic(prompt, max_tokens=max_tokens)
    elif provider == "openai":
        return _call_openai(prompt, max_tokens=max_tokens, temperature=temperature)
    raise ValueError(f"Unknown LLM provider: {provider}")


def _call_anthropic(prompt: str, *, max_tokens: int = 1024) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(prompt: str, *, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
