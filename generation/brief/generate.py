"""Brief generator — LLM call returning a structured `Brief`.

Provider selectable via `LLM_PROVIDER` env var (anthropic | openai). Both
provider clients are pinned to their respective SDKs. Prompt template is
versioned under `prompts/brief_v*.txt`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from common.config import settings
from common.logging import get_logger
from generation.brief.market_signals import gather, render_for_prompt
from generation.brief.schema import Brief, BriefRequest, validate_request

log = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "brief_v1.txt"
PROMPT_VERSION = "brief_v1"


def _render_template(req: BriefRequest) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    signals = render_for_prompt(gather(req.occasion))
    return (
        template.replace("{{occasion}}", req.occasion)
        .replace("{{relationship}}", req.relationship or "(none)")
        .replace("{{tone}}", req.tone)
        .replace("{{constraints_json}}", json.dumps(req.constraints, ensure_ascii=False))
        .replace("{{top_tropes}}", signals["top_tropes"])
        .replace("{{coverage_gaps}}", signals["coverage_gaps"])
        .replace("{{longevity_caution}}", signals["longevity_caution"])
    )


def _call_anthropic(prompt: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _extract_json(text: str) -> dict:
    # Tolerate stray prose around the JSON body
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    return json.loads(text[start : end + 1])


def generate_brief(request: dict | BriefRequest) -> Brief:
    req = request if isinstance(request, BriefRequest) else validate_request(request)
    prompt = _render_template(req)
    log.debug(f"Brief prompt ({PROMPT_VERSION}, occasion={req.occasion})")

    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        raw = _call_anthropic(prompt)
    elif provider == "openai":
        raw = _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    payload = _extract_json(raw)
    return Brief.model_validate(payload)


if __name__ == "__main__":
    import typer

    def cli(occasion: str, tone: str = "warm-sincere", relationship: str | None = None) -> None:
        brief = generate_brief({"occasion": occasion, "tone": tone, "relationship": relationship})
        print(brief.model_dump_json(indent=2))

    typer.run(cli)
