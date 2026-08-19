"""Brief generator — LLM call returning a structured `Brief`.

Provider selectable via `LLM_PROVIDER` env var (anthropic | openai). Both
provider clients are pinned to their respective SDKs. Prompt template is
versioned under `prompts/brief_v*.txt`.
"""

from __future__ import annotations

import json
from pathlib import Path

from common.llm import call_llm, extract_json
from common.logging import get_logger
from common.occasions import TONES
from generation.brief.market_signals import gather, render_for_prompt
from generation.brief.schema import Brief, BriefRequest, validate_request

log = get_logger(__name__)

# v2 stopped the brief telling the image model not to render text, and caps the
# headline at four words. v1 required `visual_prompt` to say "no text in image",
# which contradicted the lettering request appended at generation time and left
# every card falling back to the typographic overlay. Kept as a separate file
# because the version is recorded per card, so briefs from before and after
# stay distinguishable.
PROMPT_PATH = Path(__file__).parent / "prompts" / "brief_v2.txt"
PROMPT_VERSION = "brief_v2"


def _render_template(req: BriefRequest) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    signals = render_for_prompt(gather(req.occasion))
    return (
        template.replace("{{occasion}}", req.occasion)
        .replace("{{relationship}}", req.relationship or "(none)")
        # "(choose one)" rather than a default: naming a tone here would pin
        # every unconstrained brief to it, which is the bias that dropping the
        # required tone was meant to remove.
        .replace("{{tone}}", req.tone or "(choose one that suits the card)")
        .replace("{{constraints_json}}", json.dumps(req.constraints, ensure_ascii=False))
        .replace("{{top_tropes}}", signals["top_tropes"])
        .replace("{{bestseller_subjects}}", signals["bestseller_subjects"])
        .replace("{{coverage_gaps}}", signals["coverage_gaps"])
        .replace("{{longevity_caution}}", signals["longevity_caution"])
    )


def generate_brief(request: dict | BriefRequest) -> Brief:
    req = request if isinstance(request, BriefRequest) else validate_request(request)
    prompt = _render_template(req)
    log.debug(f"Brief prompt ({PROMPT_VERSION}, occasion={req.occasion})")

    raw = call_llm(prompt)
    payload = extract_json(raw)
    brief = Brief.model_validate(payload)
    # A pinned tone wins over whatever the model echoed back: the site's picker
    # is a promise to the customer, not a suggestion. Unpinned, an unrecognised
    # tone falls back rather than propagating a value the font palette and
    # message generator do not know.
    if req.tone:
        brief.tone = req.tone
    elif brief.tone not in TONES:
        # Do not substitute TONES[0]. It is `warm-sincere`, which is also the
        # value reported as never chosen when the generator picks freely, and
        # is the evidence that it does not collapse to a default. Writing it
        # here on a parse failure makes a code fault indistinguishable from a
        # model choice, and would be counted as the opposite of what happened.
        raise ValueError(
            f"Brief returned tone={brief.tone!r}, which is not one of {TONES}. "
            "Refusing to substitute a default: the substituted value is itself "
            "a reported result. Regenerate the brief instead."
        )
    return brief


if __name__ == "__main__":
    import typer

    def cli(occasion: str, tone: str = "warm-sincere", relationship: str | None = None) -> None:
        brief = generate_brief({"occasion": occasion, "tone": tone, "relationship": relationship})
        print(brief.model_dump_json(indent=2))

    typer.run(cli)
