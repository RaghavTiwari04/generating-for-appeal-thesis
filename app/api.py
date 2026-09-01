"""Greeting card generation web API.

Routes:
  GET  /                            → the development UI
  GET  /api/health                  → liveness, and whether the gate is on
  GET  /api/occasions               → canonical occasions, relationships, tones
  POST /api/generate                → start a job, return job_id
  GET  /api/generate/{job_id}       → SSE: progress, then the shuffled slate
  POST /api/generate/{job_id}/reveal → commit a choice, receive the ranking
  POST /api/choice                  → record an anonymous interaction
  GET  /api/history                 → recent jobs, development UI only

The generation pipeline is CPU/GPU-bound so each job runs in a thread-pool
executor. SSE pushes progress updates every ~2s so the UI can show a live
progress bar without polling.

The public demo asks visitors which of four cards they would send, and
compares that against the predictor. For the answer to mean anything the
ranking has to be hidden until they commit, so the job keeps two views of its
results: `results`, ranked and complete, which stays on the server, and
`public_results`, shuffled and stripped of rank and scores, which is what the
stream sends. `/reveal` trades a committed choice for the ranking. A client
that never calls it never learns what the model preferred.

Withholding is deliberate about ids too. Anything derived from position, an
array index or a `tmp-2` style identifier, leaks the ordering as surely as a
score does, so display ids are freshly generated and carry no rank.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.auth import (
    allowed_origins,
    check_rate_limit,
    check_token,
    check_token_query,
    gate_enabled,
)
from app.choices import EVENT_TYPES, build_event, logging_enabled, record_event
from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS, RELATIONSHIPS, TONES

log = get_logger(__name__)

app = FastAPI(title="Greeting Card Generator", version="1.0")
# Split deployment serves the page from one origin and this API from another,
# so the allowed origin is configuration rather than a constant. A wildcard is
# the development default and is withdrawn once a token is in play.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# In-memory job store (swap for Redis in multi-worker prod)
# ---------------------------------------------------------------------------
@dataclass
class Job:
    """One generation, and both views of its outcome.

    `results` is ranked and carries scores. It never leaves the server except
    through `/reveal`, and then only after a choice has been committed.

    `public_results` is what the stream sends: the same cards, shuffled, with
    rank and scores removed.
    """

    job_id: str
    status: str          # pending | running | done | error
    progress: list[str]
    results: list[dict]
    error: str | None = None
    created_at: float = 0.0
    public_results: list[dict] = field(default_factory=list)
    display_order: list[str] = field(default_factory=list)
    request_params: dict = field(default_factory=dict)
    revealed: bool = False
    chosen_display_id: str | None = None


_JOBS: dict[str, Job] = {}
_JOB_LIMIT = 50  # keep only the last N jobs in memory

# Anything else reaches the orchestrator and fails deep in a worker thread,
# where the visitor sees a generic error instead of a bad request.
_SCORERS = frozenset({"ridge", "mlp", "llm"})


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    occasion: str
    relationship: str | None = None
    # Optional, so the picker can offer "surprise me". Left unset, the brief
    # chooses a register to suit the concept it draws from the occasion's
    # bestsellers rather than defaulting to one.
    tone: str | None = None
    n_candidates: int = 8
    top_k: int = 3
    constraints: dict = {}
    scorer: str = "ridge"  # "ridge" | "mlp" | "llm"


class GenerateResponse(BaseModel):
    job_id: str


class RevealRequest(BaseModel):
    chosen_display_id: str


class ChoiceEvent(BaseModel):
    session_id: str
    job_id: str
    event_type: str
    time_to_choice_ms: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _evict_old_jobs() -> None:
    if len(_JOBS) > _JOB_LIMIT:
        oldest = sorted(_JOBS, key=lambda j: _JOBS[j].created_at)
        for jid in oldest[: len(_JOBS) - _JOB_LIMIT]:
            del _JOBS[jid]


def _run_pipeline(job: Job, request: dict) -> None:
    """Blocking function — runs in thread-pool executor."""
    try:
        job.status = "running"
        job.progress.append("Generating creative brief…")

        from pipeline.orchestrator import OrchestratorConfig, generate

        cfg = OrchestratorConfig(
            n_candidates=request.get("n_candidates", 8),
            top_k=request.get("top_k", 3),
            scorer=request.get("scorer", "ridge"),
        )

        # Monkey-patch orchestrator's generate_brief to emit progress
        original_generate = None
        try:
            import pipeline.orchestrator as _orch
            original_generate = _orch.generate_brief

            def _tracked_brief(req):
                job.progress.append("Brief ready — generating images…")
                return original_generate(req)

            _orch.generate_brief = _tracked_brief
        except Exception:
            pass

        job.progress.append("Generating cover images (this takes ~30s)…")
        ranked = generate(
            {
                "occasion": request["occasion"],
                "tone": request.get("tone"),
                "relationship": request.get("relationship"),
                "constraints": request.get("constraints", {}),
            },
            cfg,
        )
        job.progress.append("Scoring and ranking candidates…")

        results = []
        for i, cand in enumerate(ranked):
            # Convert image to base64 data-URL for inline display
            buf = io.BytesIO()
            cand.image.save(buf, format="JPEG", quality=90)
            import base64
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            results.append({
                "rank": i + 1,
                # Freshly generated, and unrelated to rank. The previous
                # fallback was f"tmp-{i}", which put the ranking into the id
                # itself: a visitor asked to choose blind could have read the
                # answer out of the markup.
                "display_id": str(uuid.uuid4()),
                "card_id": cand.card_id,
                "headline": cand.headline,
                "inside_message": cand.inside_message,
                "occasion": cand.occasion,
                "scores": cand.scores or {},
                "image_data_url": f"data:image/jpeg;base64,{img_b64}",
                "brief": cand.brief,
            })

        if original_generate:
            import pipeline.orchestrator as _orch
            _orch.generate_brief = original_generate

        job.results = results

        # The public view. Shuffled, and without rank, card_id or scores:
        # array order, an ordinal and a sortable score are three ways of
        # saying the same thing, and any of them would tell a visitor which
        # card the model liked before they had picked one.
        public = [
            {
                "display_id": r["display_id"],
                "headline": r["headline"],
                "inside_message": r["inside_message"],
                "occasion": r["occasion"],
                "image_data_url": r["image_data_url"],
                "brief": r["brief"],
            }
            for r in results
        ]
        random.shuffle(public)
        job.public_results = public
        job.display_order = [c["display_id"] for c in public]

        job.status = "done"
        job.progress.append(f"Done, {len(results)} cards generated.")

    except Exception as e:
        log.exception(f"Pipeline failed for job {job.job_id}")
        job.status = "error"
        job.error = str(e)
        job.progress.append(f"Error: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return Response("<h1>Frontend not found — run make frontend</h1>", media_type="text/html")
    return Response(index_file.read_text(encoding="utf-8"), media_type="text/html")


@app.get("/api/health")
async def health():
    """Unauthenticated: the frontend polls this to show whether the GPU host
    is up, which it often will not be, since running one costs money by the
    hour."""
    return {
        "status": "ok",
        "gated": gate_enabled(),
        # The site shows its data-collection notice only when this is true, so
        # the notice and the recording can never disagree.
        "logging": logging_enabled(),
    }


@app.get("/api/occasions")
async def list_occasions(token: str = Depends(check_token)):
    return {
        "occasions": list(ACTIVE_OCCASIONS),
        "relationships": list(RELATIONSHIPS),
        "tones": list(TONES),
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def start_generate(
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(check_token),
):
    # Generation is the endpoint that spends money: one LLM call for the brief
    # and n_candidates diffusion passes. It is the only one rate limited.
    check_rate_limit(token)
    if req.occasion not in ACTIVE_OCCASIONS:
        raise HTTPException(400, f"Unknown occasion: {req.occasion!r}")
    # None is allowed and means "let the brief decide"; a named tone still has
    # to be one the prompt understands.
    if req.tone is not None and req.tone not in TONES:
        raise HTTPException(400, f"Unknown tone: {req.tone!r}")
    # A visitor cannot ask for an unbounded batch: each candidate is a full
    # diffusion pass, and the reported evaluation used eight.
    if not 1 <= req.n_candidates <= 8:
        raise HTTPException(400, "n_candidates must be between 1 and 8.")
    if not 1 <= req.top_k <= req.n_candidates:
        raise HTTPException(400, "top_k must be between 1 and n_candidates.")
    if req.scorer not in _SCORERS:
        raise HTTPException(400, f"Unknown scorer: {req.scorer!r}")

    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        status="pending",
        progress=["Request received…"],
        results=[],
        created_at=time.time(),
    )
    # Kept so that a later choice event is described by what the server was
    # asked for, rather than by what a client claims it asked for.
    job.request_params = req.model_dump()
    _JOBS[job_id] = job
    _evict_old_jobs()

    asyncio.get_running_loop().run_in_executor(
        _EXECUTOR, _run_pipeline, job, req.model_dump()
    )
    return GenerateResponse(job_id=job_id)


# A comment line in the SSE grammar. EventSource drops anything beginning with
# a colon before it reaches a handler, so this is invisible to the page and
# exists purely to keep bytes moving through whatever sits in front of us.
_HEARTBEAT = ": keepalive\n\n"
_HEARTBEAT_SECONDS = 10


@app.get("/api/generate/{job_id}")
async def stream_job(job_id: str, token: str = Depends(check_token_query)):
    """Server-Sent Events stream for generation progress + final results.

    Takes its token from the query string because EventSource cannot set
    headers. See `app.auth` for why that is acceptable here and nowhere else.
    """
    if job_id not in _JOBS:
        raise HTTPException(404, "Job not found")

    async def _sse() -> AsyncIterator[str]:
        job = _JOBS[job_id]
        sent = 0
        quiet = 0
        while True:
            # Flush any new progress messages
            while sent < len(job.progress):
                msg = json.dumps({"type": "progress", "message": job.progress[sent]})
                yield f"data: {msg}\n\n"
                sent += 1
                quiet = 0

            if job.status == "done":
                # public_results only. Sending job.results here would undo the
                # whole arrangement.
                payload = json.dumps({"type": "done", "results": job.public_results})
                yield f"data: {payload}\n\n"
                return

            if job.status == "error":
                payload = json.dumps({"type": "error", "message": job.error})
                yield f"data: {payload}\n\n"
                return

            await asyncio.sleep(1.0)

            # Progress messages are coarse, and the gap between "generating
            # images" and the scoring line spans every diffusion pass with
            # nothing to report: minutes of a connection with no bytes on it.
            # Proxies read that as a dead origin and hang up. RunPod fronts
            # pods with Cloudflare, which gives up after 100 seconds, so
            # without this the demo fails on the proxy while the GPU is still
            # working perfectly.
            quiet += 1
            if quiet >= _HEARTBEAT_SECONDS:
                yield _HEARTBEAT
                quiet = 0

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/api/generate/{job_id}/reveal")
async def reveal_job(
    job_id: str,
    req: RevealRequest,
    token: str = Depends(check_token),
):
    """Trade a committed choice for the model's ranking.

    This is the only route that returns ranks and scores for a job, and it
    will not do so until a choice has been named. The first call fixes the
    choice; later calls return the same answer, so a reload after choosing
    shows what the visitor already saw rather than letting them choose twice.

    Costs nothing to serve, so it is not rate limited.
    """
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, "That job has not finished generating.")
    if req.chosen_display_id not in job.display_order:
        raise HTTPException(400, "That card is not part of this job.")

    if not job.revealed:
        job.chosen_display_id = req.chosen_display_id
        job.revealed = True

    chosen = next(
        (r for r in job.results if r["display_id"] == job.chosen_display_id), None
    )
    return {
        "candidates": [
            {"display_id": r["display_id"], "rank": r["rank"], "scores": r["scores"]}
            for r in job.results
        ],
        "model_top_display_id": job.results[0]["display_id"] if job.results else None,
        "chosen_display_id": job.chosen_display_id,
        "chosen_rank": chosen["rank"] if chosen else None,
    }


@app.post("/api/choice", status_code=204)
async def record_choice(ev: ChoiceEvent, token: str = Depends(check_token)):
    """Record one anonymous interaction, or quietly do nothing.

    Answers 204 in every case: logging disabled, job long since evicted,
    database unreachable. The visitor's experience must not depend on whether
    the study is currently collecting, and the frontend should not have to
    know either.

    The client sends its session, the job and which event this is. Everything
    about the ranking is read from the server's own record of that job.
    """
    if ev.event_type not in EVENT_TYPES:
        raise HTTPException(400, f"Unknown event type: {ev.event_type!r}")

    job = _JOBS.get(ev.job_id)
    if job is not None and logging_enabled():
        record_event(
            build_event(
                job,
                session_id=ev.session_id,
                event_type=ev.event_type,
                time_to_choice_ms=ev.time_to_choice_ms,
            )
        )
    return FastAPIResponse(status_code=204)


@app.get("/api/history")
async def get_history(limit: int = 12, token: str = Depends(check_token)):
    """Recent jobs, for the development UI.

    Returns the top-ranked card and its score, so it is not something the
    public demo may call: it would hand over the ranking the choose step
    exists to withhold.
    """
    recent = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)
    done = [j for j in recent if j.status == "done"][:limit]
    return {
        "cards": [
            {
                "job_id": j.job_id,
                "occasion": (j.results[0]["occasion"] if j.results else ""),
                "headline": (j.results[0]["headline"] if j.results else ""),
                "image_data_url": (j.results[0]["image_data_url"] if j.results else ""),
                "score": (j.results[0]["scores"].get("saleability_calibrated", 0)
                          if j.results else 0),
            }
            for j in done
        ]
    }
