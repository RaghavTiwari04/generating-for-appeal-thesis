"""Anonymous logging of which card a visitor picks.

The predictor learns from judge labels, and the thesis has no human preference
data to check them against. A visitor choosing between four candidates is
exactly the comparison the reported best-of-N null leaves open, so the demo
records the choice.

Recording people's behaviour is human-subjects research, and this project's
ethics position is currently that no human participants were involved. So the
default is off. `GC_LOG_CHOICES` turns it on, and nothing is written until it
is set. The visitor's experience is identical either way: the endpoint answers
204 whether or not it stored anything, so there is no observable difference
between the two states and no branch in the frontend.

What is stored is in `migrations/0006_demo_choice_events.sql`. What is not:

    no IP address, no user agent, no referrer, no access token
    no free-text input, since a visitor may type a real person's name
    no image data
    no identifier that outlives the browser tab

`session_id` is minted client-side per tab. Two visits from one person are two
sessions and cannot be linked, which is the point: the research question is
about choices in aggregate, not about anybody in particular.

Scores and ranks are read from the server's own job record rather than taken
from the request, so a client cannot influence what is recorded about the
model's ranking.
"""

from __future__ import annotations

import json
import os

from common.logging import get_logger

log = get_logger(__name__)

FLAG_ENV = "GC_LOG_CHOICES"

EVENT_TYPES = frozenset({
    "choice",
    "download_front",
    "download_print",
    "regenerate",
    "message_edited",
})

_INSERT = """
INSERT INTO demo_choice_events (
    session_id, job_id, event_type, occasion, tone, relationship,
    n_candidates, scorer, candidates, chosen_display_id, chosen_rank,
    shown_position, agreed_top1, time_to_choice_ms
) VALUES (
    %(session_id)s, %(job_id)s, %(event_type)s, %(occasion)s, %(tone)s,
    %(relationship)s, %(n_candidates)s, %(scorer)s, %(candidates)s,
    %(chosen_display_id)s, %(chosen_rank)s, %(shown_position)s,
    %(agreed_top1)s, %(time_to_choice_ms)s
)
"""


def logging_enabled() -> bool:
    return os.environ.get(FLAG_ENV, "").strip().lower() in {"1", "true", "yes"}


def build_event(
    job,
    *,
    session_id: str,
    event_type: str,
    time_to_choice_ms: int | None = None,
) -> dict:
    """Assemble one event from the server's own record of the job.

    The client supplies only its session, the job and which event this is.
    Everything about the model's ranking is read from `job`, so a client
    cannot report a choice as agreeing with the predictor when it did not.
    """
    params = job.request_params or {}
    ranked = {r["display_id"]: r for r in job.results}
    order = list(job.display_order or [])

    chosen = job.chosen_display_id
    chosen_row = ranked.get(chosen) if chosen else None
    top = job.results[0]["display_id"] if job.results else None

    return {
        "session_id": session_id,
        "job_id": job.job_id,
        "event_type": event_type,
        # Enumerated request fields only. `constraints` is deliberately absent:
        # it is the one field a visitor can type into.
        "occasion": params.get("occasion"),
        "tone": params.get("tone"),
        "relationship": params.get("relationship"),
        "n_candidates": params.get("n_candidates"),
        "scorer": params.get("scorer"),
        "candidates": json.dumps([
            {
                "display_id": r["display_id"],
                "rank": r["rank"],
                "scores": r["scores"],
                "shown_position": order.index(r["display_id"]) if r["display_id"] in order else None,
            }
            for r in job.results
        ]),
        "chosen_display_id": chosen,
        "chosen_rank": chosen_row["rank"] if chosen_row else None,
        "shown_position": order.index(chosen) if chosen in order else None,
        "agreed_top1": (chosen == top) if (chosen and top) else None,
        "time_to_choice_ms": time_to_choice_ms,
    }


def record_event(event: dict) -> None:
    """Write one event, or do nothing.

    Never raises. A visitor mid-demo should not see an error because the
    database is unreachable, and losing a row costs the study one data point
    where a 500 would cost it the visitor.
    """
    if not logging_enabled():
        return
    if event.get("event_type") not in EVENT_TYPES:
        log.warning(f"Refusing to record unknown event type {event.get('event_type')!r}")
        return
    try:
        from common.db import connection

        with connection() as conn, conn.cursor() as cur:
            cur.execute(_INSERT, event)
    except Exception as e:
        log.warning(f"Choice event not recorded ({e})")
