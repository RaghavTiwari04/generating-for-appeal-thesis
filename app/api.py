"""Greeting card generation web API.

Routes:
  GET  /                          → serve frontend SPA
  GET  /api/occasions             → list canonical occasions + tones
  POST /api/generate              → kick off card generation, return job_id
  GET  /api/generate/{job_id}     → SSE stream: progress events then results
  GET  /api/card/{card_id}/image  → serve cover PNG
  GET  /api/card/{card_id}        → full card JSON (scores, message, etc.)
  GET  /api/history               → last N generated cards

The generation pipeline is CPU/GPU-bound so each job runs in a thread-pool
executor. SSE pushes progress updates every ~2s so the UI can show a live
progress bar without polling.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common.logging import get_logger
from common.occasions import ACTIVE_OCCASIONS, RELATIONSHIPS, TONES

log = get_logger(__name__)

app = FastAPI(title="Greeting Card Generator", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# In-memory job store (swap for Redis in multi-worker prod)
# ---------------------------------------------------------------------------
@dataclass
class Job:
    job_id: str
    status: str          # pending | running | done | error
    progress: list[str]
    results: list[dict]
    error: str | None = None
    created_at: float = 0.0


_JOBS: dict[str, Job] = {}
_JOB_LIMIT = 50  # keep only the last N jobs in memory


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
            scorer=request.get("scorer", "predictor"),
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
                "card_id": cand.card_id or f"tmp-{i}",
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
        job.status = "done"
        job.progress.append(f"Done — {len(results)} cards generated.")

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


@app.get("/api/occasions")
async def list_occasions():
    return {
        "occasions": list(ACTIVE_OCCASIONS),
        "relationships": list(RELATIONSHIPS),
        "tones": list(TONES),
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def start_generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    if req.occasion not in ACTIVE_OCCASIONS:
        raise HTTPException(400, f"Unknown occasion: {req.occasion!r}")
    # None is allowed and means "let the brief decide"; a named tone still has
    # to be one the prompt understands.
    if req.tone is not None and req.tone not in TONES:
        raise HTTPException(400, f"Unknown tone: {req.tone!r}")

    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        status="pending",
        progress=["Request received…"],
        results=[],
        created_at=time.time(),
    )
    _JOBS[job_id] = job
    _evict_old_jobs()

    asyncio.get_running_loop().run_in_executor(
        _EXECUTOR, _run_pipeline, job, req.model_dump()
    )
    return GenerateResponse(job_id=job_id)


@app.get("/api/generate/{job_id}")
async def stream_job(job_id: str):
    """Server-Sent Events stream for generation progress + final results."""
    if job_id not in _JOBS:
        raise HTTPException(404, "Job not found")

    async def _sse() -> AsyncIterator[str]:
        job = _JOBS[job_id]
        sent = 0
        while True:
            # Flush any new progress messages
            while sent < len(job.progress):
                msg = json.dumps({"type": "progress", "message": job.progress[sent]})
                yield f"data: {msg}\n\n"
                sent += 1

            if job.status == "done":
                payload = json.dumps({"type": "done", "results": job.results})
                yield f"data: {payload}\n\n"
                return

            if job.status == "error":
                payload = json.dumps({"type": "error", "message": job.error})
                yield f"data: {payload}\n\n"
                return

            await asyncio.sleep(1.0)

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.get("/api/history")
async def get_history(limit: int = 12):
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
