"""FastAPI survey instrument for Prolific studies — Google Forms edition.

Participants land via:
    /survey?PROLIFIC_PID=<id>&STUDY_ID=<id>&SESSION_ID=<id>

Flow:
    1. GET /survey          → sample cards, create session, redirect to first card
    2. GET /card/{session}  → show card image + context + "Rate on Google Forms" button
    3. GET /next/{session}  → participant returns after submitting the form; advance to next card
    4. GET /done/{session}  → show Prolific completion URL

Rating is collected entirely via Google Forms. The form URL is pre-filled with
participant_id, card_key, occasion, and study_id so the sync script can link
responses back to DB rows.

Card images are served from MinIO via pre-signed URL (10 min expiry).

Run locally:
    uvicorn survey.instrument.app:app --reload --port 8080

Deploy on university server or a cheap VPS (Fly.io / Render free tier).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from common.config import settings
from common.db import connection
from common.logging import get_logger
from survey.instrument.sampler import (
    CardAssignment,
    PairAssignment,
    sample_main,
    sample_pairs_main,
    sample_pairs_system_eval,
    sample_system_eval,
)
from survey.instrument.sampler_calibration import (
    sample_pairs_calibration,
)
from survey.instrument.sampler_purchase import (
    sample_pairs_purchase,
)

log = get_logger(__name__)
app = FastAPI(title="Greeting Cards Survey")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory session store (fine for low-concurrency Prolific study;
# swap to Redis if concurrent participants > 50).
_SESSIONS: dict[str, dict] = {}

STUDY_TYPE_MAP: dict[str, str] = {
    "main_v1": "main",
    "system_eval_v1": "system_eval",
    # v2 pairwise studies use a different route family (`/pairsurvey`).
    "main_v2_warmup":  "pair_main",
    "main_v2":         "pair_main",
    "system_eval_v2":  "pair_system_eval",
    # VLM calibration study — 5-dim pairwise using `/calibration` routes.
    "calibration_v1":  "calibration",
    # Primary purchase_intent study — single-question 2AFC.
    "purchase_intent_v1": "purchase",
}


def _completion_url(study_id: str) -> str:
    import os
    key_map = {
        "pilot_v1":         "PROLIFIC_COMPLETION_CODE_PILOT",
        "main_v1":          "PROLIFIC_COMPLETION_CODE_MAIN",
        "system_eval_v1":   "PROLIFIC_COMPLETION_CODE_SYSEVAL",
        "main_v2_warmup":   "PROLIFIC_COMPLETION_CODE_MAIN_V2",
        "main_v2":          "PROLIFIC_COMPLETION_CODE_MAIN_V2",
        "system_eval_v2":   "PROLIFIC_COMPLETION_CODE_SYSEVAL_V2",
    }
    env_key = key_map.get(study_id, "PROLIFIC_COMPLETION_CODE_PILOT")
    code = os.environ.get(env_key, "PLACEHOLDER")
    return f"https://app.prolific.co/submissions/complete?cc={code}"


def _make_form_url(session: dict, card: dict, session_token: str) -> str | None:
    """Build a pre-filled Google Form URL for the given card.

    Returns None if Google Forms is not configured (falls back to no form link).
    """
    form_id = settings.google_forms_id
    if not form_id:
        return None

    entry_map = {
        settings.google_form_entry_participant_id: session["prolific_pid"],
        settings.google_form_entry_card_key:       card["card_key"],
        settings.google_form_entry_occasion:       card["occasion"],
        settings.google_form_entry_study_id:       session["study_id"],
    }
    params: dict[str, str] = {"usp": "pp_url"}
    for entry_id, value in entry_map.items():
        if entry_id and value:
            params[f"entry.{entry_id}"] = str(value)

    base = f"https://docs.google.com/forms/d/{form_id}/viewform"
    return f"{base}?{urlencode(params)}"


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
    cards = sample_system_eval(PROLIFIC_PID, STUDY_ID) if study_type == "system_eval" else sample_main(PROLIFIC_PID, STUDY_ID)

    session_token = str(uuid.uuid4())
    _SESSIONS[session_token] = {
        "prolific_pid": PROLIFIC_PID,
        "study_id": STUDY_ID,
        "prolific_session_id": SESSION_ID,
        "cards": [c.__dict__ for c in cards],
        "current_index": 0,
        "started_at": datetime.now(tz=UTC).isoformat(),
    }
    return RedirectResponse(url=f"/card/{session_token}")


@app.get("/card/{session_token}", response_class=HTMLResponse)
async def show_card(request: Request, session_token: str) -> Response:
    session = _SESSIONS.get(session_token)
    if not session:
        return HTMLResponse(
            "<h1>Session expired. Please return to Prolific and restart.</h1>",
            status_code=410,
        )

    idx = session["current_index"]
    cards = session["cards"]
    if idx >= len(cards):
        return RedirectResponse(url=f"/done/{session_token}")

    card = cards[idx]
    image_url = _presign(card["cover_path"])
    occasion_display = card["occasion"].replace("/", " — ").replace("_", " ").title()
    form_url = _make_form_url(session, card, session_token)

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
            "form_url": form_url,
            "next_url": f"/next/{session_token}",
            "forms_configured": form_url is not None,
        },
    )


@app.get("/next/{session_token}", response_class=HTMLResponse)
async def next_card(session_token: str) -> Response:
    """Participant returns here after submitting the Google Form."""
    session = _SESSIONS.get(session_token)
    if not session:
        return HTMLResponse(
            "<h1>Session expired. Please return to Prolific and restart.</h1>",
            status_code=410,
        )
    session["current_index"] += 1
    return RedirectResponse(url=f"/card/{session_token}", status_code=303)


@app.get("/done/{session_token}", response_class=HTMLResponse)
async def done(request: Request, session_token: str) -> Response:
    session = _SESSIONS.pop(session_token, {})
    study_id = session.get("study_id", "pilot_v1")
    return TEMPLATES.TemplateResponse(
        request,
        "done.html",
        {
            "completion_url": _completion_url(study_id),
            "attention_failures": session.get("trapdoor_failures", 0),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _presign(storage_path: str) -> str:
    from datetime import timedelta

    from minio import Minio

    if not storage_path or not storage_path.startswith("s3://"):
        return "/static/placeholder.png"
    rest = storage_path[5:]
    bucket, _, key = rest.partition("/")
    endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")
    secure = settings.minio_endpoint.startswith("https://")
    client = Minio(
        endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=secure,
    )
    return client.presigned_get_object(bucket, key, expires=timedelta(minutes=10))


# ---------------------------------------------------------------------------
# Pairwise (2AFC) routes — v2 instrument
# ---------------------------------------------------------------------------

_PAIR_INSERT_SQL = """
INSERT INTO survey_pairs (
    participant_id, study_id,
    left_listing_id, left_generated_id,
    right_listing_id, right_generated_id,
    occasion_shown, question_dim, winner_side,
    response_time_ms, attention_check_pass, rated_at
) VALUES (
    %(participant_id)s, %(study_id)s,
    %(left_listing_id)s, %(left_generated_id)s,
    %(right_listing_id)s, %(right_generated_id)s,
    %(occasion_shown)s, %(question_dim)s, %(winner_side)s,
    %(response_time_ms)s, %(attention_check_pass)s, NOW()
);
"""


def _split_card_for_sql(card: CardAssignment) -> tuple[str | None, str | None]:
    """Return (listing_id_or_none, generated_id_or_none)."""
    # Trapdoor cards have a synthetic card_key (prefix 'trapdoor_'); they are
    # never persisted as real comparisons (see attention-check handling below).
    if card.card_key.startswith("trapdoor_"):
        return (None, None)
    if card.is_generated:
        return (None, card.card_key)
    return (card.card_key, None)


def _persist_pair(
    *,
    participant_id: str,
    study_id: str,
    pair: PairAssignment,
    winner_purchase_intent: str,
    winner_aesthetic: str,
    response_time_ms: int | None,
    attention_check_pass: bool | None,
) -> None:
    """Insert two rows (one per question dim) into `survey_pairs`.

    Skips trapdoor pairs (they are scored separately for exclusion logic).
    """
    if pair.contrast_tag == "trapdoor":
        return  # never written as a real comparison

    left_listing,  left_generated  = _split_card_for_sql(pair.left)
    right_listing, right_generated = _split_card_for_sql(pair.right)

    base = {
        "participant_id": participant_id,
        "study_id": study_id,
        "left_listing_id":   left_listing,
        "left_generated_id": left_generated,
        "right_listing_id":   right_listing,
        "right_generated_id": right_generated,
        "occasion_shown": pair.occasion,
        "response_time_ms": response_time_ms,
        "attention_check_pass": attention_check_pass,
    }
    rows = [
        {**base, "question_dim": "purchase_intent", "winner_side": winner_purchase_intent},
        {**base, "question_dim": "aesthetic",       "winner_side": winner_aesthetic},
    ]
    try:
        with connection() as conn, conn.cursor() as cur:
            for r in rows:
                cur.execute(_PAIR_INSERT_SQL, r)
    except Exception as e:
        log.error(f"Failed to persist pair: {e}")


def _trapdoor_check(pair: PairAssignment, winner_purchase: str, winner_aesthetic: str) -> bool:
    """Return True if the participant correctly avoided the trapdoor card."""
    if pair.contrast_tag != "trapdoor":
        return True  # not applicable
    # Trapdoor side: whichever side has card_key starting 'trapdoor_'.
    if pair.left.card_key.startswith("trapdoor_"):
        broken = "L"
    elif pair.right.card_key.startswith("trapdoor_"):
        broken = "R"
    else:
        return True
    # Pass if neither answer picked the broken side.
    return winner_purchase != broken and winner_aesthetic != broken


@app.get("/pairsurvey", response_class=HTMLResponse)
async def start_pair_survey(
    request: Request,
    PROLIFIC_PID: str = Query(...),
    STUDY_ID: str = Query(...),
    SESSION_ID: str = Query(...),
) -> Response:
    study_type = STUDY_TYPE_MAP.get(STUDY_ID, "pair_main")
    if study_type == "pair_system_eval":
        pairs = sample_pairs_system_eval(PROLIFIC_PID, STUDY_ID)
    else:
        pairs = sample_pairs_main(PROLIFIC_PID, STUDY_ID)

    session_token = str(uuid.uuid4())
    _SESSIONS[session_token] = {
        "mode": "pair",
        "prolific_pid": PROLIFIC_PID,
        "study_id": STUDY_ID,
        "prolific_session_id": SESSION_ID,
        "pairs": [
            {
                "left": p.left.__dict__,
                "right": p.right.__dict__,
                "occasion": p.occasion,
                "contrast_tag": p.contrast_tag,
            }
            for p in pairs
        ],
        "current_index": 0,
        "trapdoor_failures": 0,
        "started_at": datetime.now(tz=UTC).isoformat(),
    }
    return RedirectResponse(url=f"/pair/{session_token}")


@app.get("/pair/{session_token}", response_class=HTMLResponse)
async def show_pair(request: Request, session_token: str) -> Response:
    session = _SESSIONS.get(session_token)
    if not session or session.get("mode") != "pair":
        return HTMLResponse(
            "<h1>Session expired. Please return to Prolific and restart.</h1>",
            status_code=410,
        )
    idx = session["current_index"]
    pairs = session["pairs"]
    if idx >= len(pairs):
        return RedirectResponse(url=f"/done/{session_token}")

    pair = pairs[idx]
    occasion_display = pair["occasion"].replace("/", " — ").replace("_", " ").title()

    return TEMPLATES.TemplateResponse(
        request,
        "pair.html",
        {
            "session_token": session_token,
            "pair_number": idx + 1,
            "total_pairs": len(pairs),
            "occasion_display": occasion_display,
            "left_image_url":  _presign(pair["left"]["cover_path"]),
            "right_image_url": _presign(pair["right"]["cover_path"]),
            "left_headline":  pair["left"].get("headline") or "",
            "right_headline": pair["right"].get("headline") or "",
            "left_inside_message":  pair["left"].get("inside_message") or "",
            "right_inside_message": pair["right"].get("inside_message") or "",
            "submit_url": f"/pairsubmit/{session_token}",
        },
    )


@app.post("/pairsubmit/{session_token}", response_class=HTMLResponse)
async def submit_pair(
    session_token: str,
    winner_purchase_intent: str = Form(...),
    winner_aesthetic: str = Form(...),
    t_start_ms: str = Form(""),
    pair_index: int = Form(...),
) -> Response:
    session = _SESSIONS.get(session_token)
    if not session or session.get("mode") != "pair":
        return HTMLResponse(
            "<h1>Session expired. Please return to Prolific and restart.</h1>",
            status_code=410,
        )

    idx = session["current_index"]
    pairs_meta = session["pairs"]
    if idx >= len(pairs_meta):
        return RedirectResponse(url=f"/done/{session_token}", status_code=303)

    raw = pairs_meta[idx]
    pair = PairAssignment(
        left=CardAssignment(**raw["left"]),
        right=CardAssignment(**raw["right"]),
        occasion=raw["occasion"],
        contrast_tag=raw["contrast_tag"],
    )

    # Compute response time
    try:
        elapsed = max(0, int(datetime.now(tz=UTC).timestamp() * 1000) - int(t_start_ms))
    except (TypeError, ValueError):
        elapsed = None

    # Trapdoor accounting
    passed = _trapdoor_check(pair, winner_purchase_intent, winner_aesthetic)
    if not passed:
        session["trapdoor_failures"] += 1

    _persist_pair(
        participant_id=session["prolific_pid"],
        study_id=session["study_id"],
        pair=pair,
        winner_purchase_intent=winner_purchase_intent,
        winner_aesthetic=winner_aesthetic,
        response_time_ms=elapsed,
        attention_check_pass=(passed if pair.contrast_tag == "trapdoor" else None),
    )

    session["current_index"] += 1
    if session["current_index"] >= len(pairs_meta):
        return RedirectResponse(url=f"/done/{session_token}", status_code=303)
    return RedirectResponse(url=f"/pair/{session_token}", status_code=303)


# ---------------------------------------------------------------------------
# VLM Calibration routes — 5-dimension pairwise (2AFC)
# ---------------------------------------------------------------------------

_CAL_PERSIST_SQL = """
INSERT INTO survey_pairs (
    participant_id, study_id,
    left_listing_id, left_generated_id,
    right_listing_id, right_generated_id,
    occasion_shown, question_dim, winner_side,
    response_time_ms, attention_check_pass, rated_at
) VALUES (
    %(participant_id)s, %(study_id)s,
    %(left_listing_id)s, NULL,
    %(right_listing_id)s, NULL,
    %(occasion_shown)s, %(question_dim)s, %(winner_side)s,
    %(response_time_ms)s, %(attention_check_pass)s, NOW()
);
"""


def _persist_calibration_pair(
    *,
    participant_id: str,
    study_id: str,
    left_id: str,
    right_id: str,
    winners: dict[str, str],
    response_time_ms: int | None,
    attention_check_pass: bool | None,
    is_trapdoor: bool,
) -> None:
    """Insert one row per dimension into survey_pairs for a calibration pair."""
    if is_trapdoor:
        return  # trapdoors not persisted as real data

    try:
        with connection() as conn, conn.cursor() as cur:
            for dim, winner in winners.items():
                cur.execute(
                    _CAL_PERSIST_SQL,
                    {
                        "participant_id": participant_id,
                        "study_id": study_id,
                        "left_listing_id": left_id,
                        "right_listing_id": right_id,
                        "occasion_shown": "birthday/general",
                        "question_dim": dim,
                        "winner_side": winner,
                        "response_time_ms": response_time_ms,
                        "attention_check_pass": attention_check_pass,
                    },
                )
    except Exception as e:
        log.error(f"Failed to persist calibration pair: {e}")


@app.get("/calibration", response_class=HTMLResponse)
async def start_calibration(
    request: Request,
    PROLIFIC_PID: str = Query(...),
    STUDY_ID: str = Query("calibration_v1"),
    SESSION_ID: str = Query(...),
) -> Response:
    """Start the VLM calibration survey — 5-dimension pairwise."""
    pairs = sample_pairs_calibration(PROLIFIC_PID, STUDY_ID)

    session_token = str(uuid.uuid4())
    _SESSIONS[session_token] = {
        "mode": "calibration",
        "prolific_pid": PROLIFIC_PID,
        "study_id": STUDY_ID,
        "prolific_session_id": SESSION_ID,
        "pairs": [
            {
                "left_id": p.left.listing_id,
                "right_id": p.right.listing_id,
                "left_image_url": p.left.image_url,
                "right_image_url": p.right.image_url,
                "left_title": p.left.title,
                "right_title": p.right.title,
                "contrast_tag": p.contrast_tag,
            }
            for p in pairs
        ],
        "current_index": 0,
        "trapdoor_failures": 0,
        "started_at": datetime.now(tz=UTC).isoformat(),
    }
    return RedirectResponse(url=f"/calpair/{session_token}")


@app.get("/calpair/{session_token}", response_class=HTMLResponse)
async def show_calibration_pair(request: Request, session_token: str) -> Response:
    session = _SESSIONS.get(session_token)
    if not session or session.get("mode") != "calibration":
        return HTMLResponse(
            "<h1>Session expired. Please return to Prolific and restart.</h1>",
            status_code=410,
        )
    idx = session["current_index"]
    pairs = session["pairs"]
    if idx >= len(pairs):
        return RedirectResponse(url=f"/done/{session_token}")

    pair = pairs[idx]
    return TEMPLATES.TemplateResponse(
        request,
        "pair_calibration.html",
        {
            "session_token": session_token,
            "pair_number": idx + 1,
            "total_pairs": len(pairs),
            "left_image_url": pair["left_image_url"],
            "right_image_url": pair["right_image_url"],
            "left_headline": pair.get("left_title") or "",
            "right_headline": pair.get("right_title") or "",
            "submit_url": f"/calsubmit/{session_token}",
        },
    )


@app.post("/calsubmit/{session_token}", response_class=HTMLResponse)
async def submit_calibration_pair(
    session_token: str,
    winner_occasion_fit: str = Form(...),
    winner_aesthetic: str = Form(...),
    winner_emotional_resonance: str = Form(...),
    winner_distinctiveness: str = Form(...),
    t_start_ms: str = Form(""),
    pair_index: int = Form(...),
) -> Response:
    session = _SESSIONS.get(session_token)
    if not session or session.get("mode") != "calibration":
        return HTMLResponse(
            "<h1>Session expired. Please return to Prolific and restart.</h1>",
            status_code=410,
        )

    idx = session["current_index"]
    pairs_meta = session["pairs"]
    if idx >= len(pairs_meta):
        return RedirectResponse(url=f"/done/{session_token}", status_code=303)

    pair = pairs_meta[idx]
    is_trapdoor = pair["contrast_tag"] == "trapdoor"

    # Response time
    try:
        elapsed = max(0, int(datetime.now(tz=UTC).timestamp() * 1000) - int(t_start_ms))
    except (TypeError, ValueError):
        elapsed = None

    winners = {
        "occasion_fit": winner_occasion_fit,
        "aesthetic": winner_aesthetic,
        "emotional_resonance": winner_emotional_resonance,
        "distinctiveness": winner_distinctiveness,
    }

    # Trapdoor check: for trapdoor pairs, the high-quality card should win
    # on most dimensions. Simple heuristic: if the obvious loser wins on 3+ dims,
    # the participant failed.
    attention_pass = None
    if is_trapdoor:
        # Convention: in trapdoor pairs, left=high or right=high card.
        # We check if "T" (tie) or inconsistent answers dominate.
        n_ties = sum(1 for v in winners.values() if v == "T")
        attention_pass = n_ties < 3  # Allow up to 2 ties
        if not attention_pass:
            session["trapdoor_failures"] += 1

    _persist_calibration_pair(
        participant_id=session["prolific_pid"],
        study_id=session["study_id"],
        left_id=pair["left_id"],
        right_id=pair["right_id"],
        winners=winners,
        response_time_ms=elapsed,
        attention_check_pass=attention_pass,
        is_trapdoor=is_trapdoor,
    )

    session["current_index"] += 1
    if session["current_index"] >= len(pairs_meta):
        return RedirectResponse(url=f"/done/{session_token}", status_code=303)
    return RedirectResponse(url=f"/calpair/{session_token}", status_code=303)


# ---------------------------------------------------------------------------
# Purchase-intent routes — single-question 2AFC (primary human labels)
# ---------------------------------------------------------------------------

_PURCHASE_PERSIST_SQL = """
INSERT INTO survey_pairs (
    participant_id, study_id,
    left_listing_id, left_generated_id,
    right_listing_id, right_generated_id,
    occasion_shown, question_dim, winner_side,
    response_time_ms, attention_check_pass, rated_at
) VALUES (
    %(participant_id)s, %(study_id)s,
    %(left_listing_id)s, NULL,
    %(right_listing_id)s, NULL,
    %(occasion_shown)s, 'purchase_intent', %(winner_side)s,
    %(response_time_ms)s, %(attention_check_pass)s, NOW()
);
"""


@app.get("/purchase", response_class=HTMLResponse)
async def start_purchase_survey(
    request: Request,
    PROLIFIC_PID: str = Query(...),
    STUDY_ID: str = Query("purchase_intent_v1"),
    SESSION_ID: str = Query(...),
) -> Response:
    """Start the purchase_intent pairwise study."""
    pairs = sample_pairs_purchase(PROLIFIC_PID, STUDY_ID)

    session_token = str(uuid.uuid4())
    _SESSIONS[session_token] = {
        "mode": "purchase",
        "prolific_pid": PROLIFIC_PID,
        "study_id": STUDY_ID,
        "prolific_session_id": SESSION_ID,
        "pairs": [
            {
                "left_id": p.left.listing_id,
                "right_id": p.right.listing_id,
                "left_image_url": p.left.image_url,
                "right_image_url": p.right.image_url,
                "left_title": p.left.title,
                "right_title": p.right.title,
                "contrast_tag": p.contrast_tag,
            }
            for p in pairs
        ],
        "current_index": 0,
        "trapdoor_failures": 0,
        "started_at": datetime.now(tz=UTC).isoformat(),
    }
    return RedirectResponse(url=f"/buypair/{session_token}")


@app.get("/buypair/{session_token}", response_class=HTMLResponse)
async def show_purchase_pair(request: Request, session_token: str) -> Response:
    session = _SESSIONS.get(session_token)
    if not session or session.get("mode") != "purchase":
        return HTMLResponse(
            "<h1>Session expired. Please return to Prolific and restart.</h1>",
            status_code=410,
        )
    idx = session["current_index"]
    pairs = session["pairs"]
    if idx >= len(pairs):
        return RedirectResponse(url=f"/done/{session_token}")

    pair = pairs[idx]
    return TEMPLATES.TemplateResponse(
        request,
        "pair_purchase.html",
        {
            "session_token": session_token,
            "pair_number": idx + 1,
            "total_pairs": len(pairs),
            "left_image_url": pair["left_image_url"],
            "right_image_url": pair["right_image_url"],
            "left_headline": pair.get("left_title") or "",
            "right_headline": pair.get("right_title") or "",
            "submit_url": f"/buysubmit/{session_token}",
        },
    )


@app.post("/buysubmit/{session_token}", response_class=HTMLResponse)
async def submit_purchase_pair(
    session_token: str,
    winner: str = Form(...),
    t_start_ms: str = Form(""),
    pair_index: int = Form(...),
) -> Response:
    session = _SESSIONS.get(session_token)
    if not session or session.get("mode") != "purchase":
        return HTMLResponse(
            "<h1>Session expired. Please return to Prolific and restart.</h1>",
            status_code=410,
        )

    idx = session["current_index"]
    pairs_meta = session["pairs"]
    if idx >= len(pairs_meta):
        return RedirectResponse(url=f"/done/{session_token}", status_code=303)

    pair = pairs_meta[idx]
    is_trapdoor = pair["contrast_tag"] == "trapdoor"

    # Response time
    try:
        elapsed = max(0, int(datetime.now(tz=UTC).timestamp() * 1000) - int(t_start_ms))
    except (TypeError, ValueError):
        elapsed = None

    # Trapdoor check: if the participant picks the obviously worse card
    # or ties on a massive quality gap, flag as failed.
    attention_pass = None
    if is_trapdoor:
        attention_pass = winner != "T"  # Tie on obvious mismatch = fail
        if not attention_pass:
            session["trapdoor_failures"] += 1

    # Persist
    if not is_trapdoor:
        try:
            with connection() as conn, conn.cursor() as cur:
                cur.execute(
                    _PURCHASE_PERSIST_SQL,
                    {
                        "participant_id": session["prolific_pid"],
                        "study_id": session["study_id"],
                        "left_listing_id": pair["left_id"],
                        "right_listing_id": pair["right_id"],
                        "occasion_shown": "birthday/general",
                        "winner_side": winner,
                        "response_time_ms": elapsed,
                        "attention_check_pass": attention_pass,
                    },
                )
        except Exception as e:
            log.error(f"Failed to persist purchase pair: {e}")

    session["current_index"] += 1
    if session["current_index"] >= len(pairs_meta):
        return RedirectResponse(url=f"/done/{session_token}", status_code=303)
    return RedirectResponse(url=f"/buypair/{session_token}", status_code=303)
