"""FastAPI survey instrument for Prolific studies.

Participants land via:
    /survey?PROLIFIC_PID=<id>&STUDY_ID=<id>&SESSION_ID=<id>

Flow:
    1. GET  /survey          → redirect to first card
    2. GET  /card/{session}  → render rating form for next card
    3. POST /rate/{session}  → save rating, redirect to next card or completion
    4. GET  /done/{session}  → show completion URL (Prolific redirects)

Card images are served from MinIO via pre-signed URL (5 min expiry).

Run locally:
    uvicorn survey.instrument.app:app --reload --port 8080

Deploy on university server or a cheap VPS (Fly.io / Render free tier).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, Form, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from common.config import settings
from common.db import connection
from common.logging import get_logger
from survey.instrument.sampler import CardAssignment, sample_main, sample_system_eval

log = get_logger(__name__)
app = FastAPI(title="Greeting Cards Survey")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory session store (fine for low-concurrency Prolific study;
# swap to Redis if concurrent participants > 50).
_SESSIONS: dict[str, dict] = {}

def _completion_url(study_id: str) -> str:
    """Build Prolific completion URL from env var per study."""
    import os
    key_map = {
        "pilot_v1":        "PROLIFIC_COMPLETION_CODE_PILOT",
        "main_v1":         "PROLIFIC_COMPLETION_CODE_MAIN",
        "system_eval_v1":  "PROLIFIC_COMPLETION_CODE_SYSEVAL",
    }
    env_key = key_map.get(study_id, "PROLIFIC_COMPLETION_CODE_PILOT")
    code = os.environ.get(env_key, "PLACEHOLDER")
    return f"https://app.prolific.co/submissions/complete?cc={code}"
STUDY_TYPE_MAP: dict[str, str] = {
    "main_v1": "main",
    "system_eval_v1": "system_eval",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/survey", response_class=HTMLResponse)
async def start_survey(
    request: Request,
    PROLIFIC_PID: str = Query(...),
    STUDY_ID: str = Query(...),
    SESSION_ID: str = Query(...),
) -> Response:
    study_type = STUDY_TYPE_MAP.get(STUDY_ID, "main")

    if study_type == "system_eval":
        cards = sample_system_eval(PROLIFIC_PID, STUDY_ID)
    else:
        cards = sample_main(PROLIFIC_PID, STUDY_ID)

    session_token = str(uuid.uuid4())
    _SESSIONS[session_token] = {
        "prolific_pid": PROLIFIC_PID,
        "study_id": STUDY_ID,
        "prolific_session_id": SESSION_ID,
        "cards": [c.__dict__ for c in cards],
        "current_index": 0,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
        "attention_checks": {},
    }
    return RedirectResponse(url=f"/card/{session_token}")


@app.get("/card/{session_token}", response_class=HTMLResponse)
async def show_card(request: Request, session_token: str) -> Response:
    session = _SESSIONS.get(session_token)
    if not session:
        return HTMLResponse("<h1>Session expired. Please return to Prolific and restart.</h1>", status_code=410)

    idx = session["current_index"]
    cards = session["cards"]
    if idx >= len(cards):
        return RedirectResponse(url=f"/done/{session_token}")

    card = cards[idx]
    image_url = _presign(card["cover_path"])
    occasion_display = card["occasion"].replace("/", " — ").replace("_", " ").title()
    is_attention = (idx % 10 == 9)  # every 10th item is an attention check

    return TEMPLATES.TemplateResponse(
        request,
        "card.html",
        {
            "session_token": session_token,
            "card_number": idx + 1,
            "total_cards": len(cards),
            "image_url": image_url,
            "occasion_display": occasion_display,
            "headline": card.get("headline") or "",
            "inside_message": card.get("inside_message") or "",
            "is_attention_check": is_attention,
            "card_key": card["card_key"],
        },
    )


@app.post("/rate/{session_token}", response_class=HTMLResponse)
async def submit_rating(
    session_token: str,
    card_key: Annotated[str, Form()],
    purchase_intent: Annotated[int, Form()],
    occasion_fit: Annotated[int, Form()],
    aesthetic: Annotated[int, Form()],
    emotional_resonance: Annotated[int, Form()],
    distinctiveness: Annotated[int, Form()],
    max_price_gbp: Annotated[float, Form()] = 0.0,
    free_text: Annotated[str, Form()] = "",
    response_time_ms: Annotated[int, Form()] = 0,
    attention_answer: Annotated[str, Form()] = "",
) -> Response:
    session = _SESSIONS.get(session_token)
    if not session:
        return HTMLResponse("<h1>Session expired.</h1>", status_code=410)

    idx = session["current_index"]
    is_attention = (idx % 10 == 9)
    attention_pass = True
    if is_attention:
        attention_pass = attention_answer.strip().lower() == "strongly disagree"
        session["attention_checks"][str(idx)] = attention_pass

    card = session["cards"][idx]
    listing_id = card["card_key"] if not card.get("is_generated") else None
    generated_card_id = card["card_key"] if card.get("is_generated") else None

    _insert_rating(
        participant_id=session["prolific_pid"],
        study_id=session["study_id"],
        listing_id=listing_id,
        generated_card_id=generated_card_id,
        occasion_shown=card["occasion"],
        purchase_intent=purchase_intent,
        occasion_fit=occasion_fit,
        aesthetic=aesthetic,
        emotional_resonance=emotional_resonance,
        distinctiveness=distinctiveness,
        max_price_gbp=max_price_gbp,
        free_text=free_text.strip()[:500],
        response_time_ms=response_time_ms,
        attention_check_pass=attention_pass if is_attention else None,
    )

    session["current_index"] += 1
    return RedirectResponse(url=f"/card/{session_token}", status_code=303)


@app.get("/done/{session_token}", response_class=HTMLResponse)
async def done(request: Request, session_token: str) -> Response:
    session = _SESSIONS.pop(session_token, {})
    fails = sum(1 for v in session.get("attention_checks", {}).values() if not v)
    study_id = session.get("study_id", "pilot_v1")
    return TEMPLATES.TemplateResponse(
        request,
        "done.html",
        {
            "completion_url": _completion_url(study_id),
            "attention_failures": fails,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _presign(storage_path: str) -> str:
    from minio import Minio
    from datetime import timedelta

    if not storage_path or not storage_path.startswith("s3://"):
        return "/static/placeholder.png"
    rest = storage_path[5:]
    bucket, _, key = rest.partition("/")
    endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")
    secure = settings.minio_endpoint.startswith("https://")
    client = Minio(endpoint, access_key=settings.minio_root_user,
                   secret_key=settings.minio_root_password, secure=secure)
    url = client.presigned_get_object(bucket, key, expires=timedelta(minutes=10))
    return url


_INSERT_RATING = """
INSERT INTO survey_ratings (
    participant_id, study_id, listing_id, generated_card_id,
    occasion_shown, purchase_intent, occasion_fit, aesthetic,
    emotional_resonance, distinctiveness, max_price_gbp,
    free_text, response_time_ms, attention_check_pass, rated_at
) VALUES (
    %(participant_id)s, %(study_id)s, %(listing_id)s, %(generated_card_id)s,
    %(occasion_shown)s, %(purchase_intent)s, %(occasion_fit)s, %(aesthetic)s,
    %(emotional_resonance)s, %(distinctiveness)s, %(max_price_gbp)s,
    %(free_text)s, %(response_time_ms)s, %(attention_check_pass)s, NOW()
);
"""


def _insert_rating(**kwargs) -> None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(_INSERT_RATING, kwargs)
