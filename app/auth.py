"""Access control for a publicly reachable deployment.

The app is a research artefact, not a product: it holds a LoRA trained on
designs whose copyright belongs to the sellers who made them, and the pipeline
has no filter for recognisable intellectual property. Section 5.4 of the thesis
states that this is sufficient for research and would not be sufficient for a
product. A gate is what keeps the deployed demo on the research side of that
line, so it is required rather than optional whenever the app is exposed.

It also bounds spend. Every generation is an LLM call for the brief plus
`n_candidates` diffusion passes, so an ungated endpoint is an open invitation
to bill someone else's card.

Two mechanisms, deliberately simple, because the alternative to simple here is
nothing at all:

  a shared token, supplied by whoever runs the deployment through
      GC_ACCESS_TOKEN. Sent as an `X-Access-Token` header by normal requests.
      Server-sent events cannot carry headers, so the streaming endpoint also
      accepts `?token=`; that puts the token in a URL and therefore in server
      logs, which is why it is a demo token to be rotated rather than a
      credential to be reused anywhere else.

  a per-token rate limit on generation only, since that is the endpoint that
      costs money.

Set GC_ACCESS_TOKEN to enable both. With it unset the app runs open, which is
correct for the SSH-tunnel workflow of `07_serve_app.sh` where the tunnel is
already the gate, and wrong for anything reachable from the internet --
`require_gate_configured()` exists so a deployment can refuse to start that way.
"""

from __future__ import annotations

import hmac
import os
import time
from collections import deque

from fastapi import Header, HTTPException, Query

from common.logging import get_logger

log = get_logger(__name__)

TOKEN_ENV = "GC_ACCESS_TOKEN"
# Generations per token per window. A demo visitor needs a handful; a scraper
# wants thousands, and the gap between those two is the whole point.
RATE_LIMIT = int(os.environ.get("GC_RATE_LIMIT", "12"))
RATE_WINDOW_S = int(os.environ.get("GC_RATE_WINDOW_S", "3600"))

_recent: dict[str, deque[float]] = {}


def gate_enabled() -> bool:
    return bool(os.environ.get(TOKEN_ENV, "").strip())


def require_gate_configured() -> None:
    """Refuse to serve publicly without a token. Called by the deploy entrypoint."""
    if not gate_enabled():
        raise SystemExit(
            f"{TOKEN_ENV} is not set. Refusing to start a publicly reachable "
            "instance without an access gate: the endpoint spends API credit "
            "and GPU time per request, and the adapter is trained on designs "
            "this project has no licence to serve to the public. Set "
            f"{TOKEN_ENV} to a random string, or run behind the SSH tunnel "
            "instead."
        )


def _valid(candidate: str | None) -> bool:
    expected = os.environ.get(TOKEN_ENV, "")
    if not expected:
        return True  # open mode: the tunnel is the gate
    # Constant-time: a token is short and an attacker can retry freely.
    return bool(candidate) and hmac.compare_digest(candidate, expected)


def check_token(x_access_token: str | None = Header(default=None)) -> str:
    """FastAPI dependency for ordinary requests."""
    if not _valid(x_access_token):
        raise HTTPException(status_code=401, detail="Invalid or missing access token.")
    return x_access_token or "open"


def check_token_query(token: str | None = Query(default=None)) -> str:
    """Dependency for server-sent events, which cannot send headers."""
    if not _valid(token):
        raise HTTPException(status_code=401, detail="Invalid or missing access token.")
    return token or "open"


def check_rate_limit(token: str) -> None:
    """Raise 429 once a token has spent its allowance for the window."""
    if not gate_enabled():
        return
    now = time.time()
    seen = _recent.setdefault(token, deque())
    while seen and now - seen[0] > RATE_WINDOW_S:
        seen.popleft()
    if len(seen) >= RATE_LIMIT:
        wait = int(RATE_WINDOW_S - (now - seen[0]))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached: {RATE_LIMIT} cards per "
                   f"{RATE_WINDOW_S // 60} minutes. Try again in {wait // 60 + 1} min.",
        )
    seen.append(now)


def allowed_origins() -> list[str]:
    """Origins the frontend may be served from.

    Split deployment puts the page on one host and this API on another, so the
    browser sends a cross-origin request and CORS has to name the page's origin
    explicitly. `*` is the development default and is refused once a gate is
    configured, because a wildcard plus a token in a URL is how a token leaks.
    """
    raw = os.environ.get("GC_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    if gate_enabled():
        log.warning(
            "GC_ALLOWED_ORIGINS is unset while the gate is on; allowing no "
            "cross-origin requests. Set it to the frontend's origin."
        )
        return []
    return ["*"]
