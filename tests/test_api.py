"""Tests for the generation web API (no pipeline execution — all mocked)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import _JOBS, Job, app

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def clear_jobs():
    _JOBS.clear()
    yield
    _JOBS.clear()


def _dummy_result():
    import base64
    import io

    from PIL import Image
    img = Image.new("RGB", (64, 64), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "rank": 1,
        "card_id": "test-card-001",
        "headline": "Happy Birthday!",
        "inside_message": "Wishing you all the best.",
        "occasion": "birthday/general",
        "scores": {"saleability_calibrated": 0.82, "aesthetic": 0.75},
        "image_data_url": f"data:image/jpeg;base64,{b64}",
        "brief": {},
    }


class TestOccasionsEndpoint:
    def test_returns_occasions(self):
        r = client.get("/api/occasions")
        assert r.status_code == 200
        d = r.json()
        assert "occasions" in d
        assert "birthday/general" in d["occasions"]
        assert "tones" in d
        assert "relationships" in d

    def test_occasions_nonempty(self):
        r = client.get("/api/occasions")
        d = r.json()
        assert len(d["occasions"]) >= 1
        assert len(d["tones"]) >= 3


class TestGenerateEndpoint:
    def test_invalid_occasion_returns_400(self):
        r = client.post("/api/generate", json={
            "occasion": "not_a_real_occasion",
            "tone": "warm-sincere",
        })
        assert r.status_code == 400

    def test_invalid_tone_returns_400(self):
        r = client.post("/api/generate", json={
            "occasion": "birthday/general",
            "tone": "not_a_tone",
        })
        assert r.status_code == 400

    def test_valid_request_returns_job_id(self):
        # Patch run_in_executor on the running event loop so no real thread spawns.
        # TestClient runs the ASGI app in a thread with its own event loop.
        import concurrent.futures

        def _fake_run_in_executor(executor, fn, *args, **kwargs):
            """Run synchronously, mark job done immediately."""
            job = args[0]
            job.status = "done"
            job.results = [_dummy_result()]
            # Return a real future so wrap_future is happy
            fut = concurrent.futures.Future()
            fut.set_result(None)
            return fut

        with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=_fake_run_in_executor):
            r = client.post("/api/generate", json={
                "occasion": "birthday/general",
                "tone": "warm-sincere",
                "n_candidates": 2,
                "top_k": 1,
            })
        assert r.status_code == 200
        d = r.json()
        assert "job_id" in d
        assert len(d["job_id"]) == 36  # UUID

    def test_unknown_job_returns_404(self):
        r = client.get("/api/generate/nonexistent-job-id-123")
        assert r.status_code == 404


class TestHistoryEndpoint:
    def test_empty_history(self):
        r = client.get("/api/history")
        assert r.status_code == 200
        assert r.json()["cards"] == []

    def test_done_jobs_appear_in_history(self):
        job = Job(
            job_id="hist-test-001",
            status="done",
            progress=["done"],
            results=[_dummy_result()],
            created_at=time.time(),
        )
        _JOBS["hist-test-001"] = job
        r = client.get("/api/history?limit=5")
        assert r.status_code == 200
        cards = r.json()["cards"]
        assert len(cards) == 1
        assert cards[0]["headline"] == "Happy Birthday!"

    def test_running_jobs_not_in_history(self):
        job = Job(
            job_id="running-001",
            status="running",
            progress=[],
            results=[],
            created_at=time.time(),
        )
        _JOBS["running-001"] = job
        r = client.get("/api/history")
        assert all(c.get("job_id") != "running-001" for c in r.json()["cards"])


class TestIndexRoute:
    def test_index_returns_html(self):
        r = client.get("/")
        assert r.status_code == 200
        # Either serves the SPA or the fallback message
        assert "text/html" in r.headers["content-type"]
