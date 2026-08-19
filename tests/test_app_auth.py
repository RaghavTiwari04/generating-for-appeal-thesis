"""The access gate is what keeps a deployed instance a research demo.

Without it the endpoint spends an LLM call and eight diffusion passes for
anyone who finds the URL, and serves a LoRA trained on designs this project has
no licence to publish. These tests exist so that stays true after refactors.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("GC_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("GC_ALLOWED_ORIGINS", "https://example.invalid")
    monkeypatch.setenv("GC_RATE_LIMIT", "3")
    import app.auth as auth
    import app.api as api
    importlib.reload(auth)
    importlib.reload(api)
    auth._recent.clear()
    return TestClient(api.app)


GEN = {"occasion": "birthday/general", "n_candidates": 2, "top_k": 1}
HDR = {"X-Access-Token": "test-token"}


class TestGate:
    def test_health_is_open(self, client):
        """The frontend must be able to see whether the GPU host is up."""
        r = client.get("/api/health")
        assert r.status_code == 200 and r.json()["gated"] is True

    def test_generate_without_a_token_is_refused(self, client):
        assert client.post("/api/generate", json=GEN).status_code == 401

    def test_generate_with_a_wrong_token_is_refused(self, client):
        r = client.post("/api/generate", json=GEN,
                        headers={"X-Access-Token": "not-it"})
        assert r.status_code == 401

    def test_occasions_and_history_are_gated(self, client):
        assert client.get("/api/occasions").status_code == 401
        assert client.get("/api/history").status_code == 401
        assert client.get("/api/occasions", headers=HDR).status_code == 200

    def test_stream_takes_its_token_from_the_query(self, client):
        """EventSource cannot set headers, so the stream reads ?token=."""
        assert client.get("/api/generate/nope").status_code == 401
        # Correct token, unknown job: past the gate, so 404 not 401.
        assert client.get("/api/generate/nope?token=test-token").status_code == 404


class TestLimits:
    def test_batch_size_is_bounded(self, client):
        r = client.post("/api/generate", json={**GEN, "n_candidates": 500},
                        headers=HDR)
        assert r.status_code == 400

    def test_top_k_cannot_exceed_the_batch(self, client):
        r = client.post("/api/generate",
                        json={"occasion": "birthday/general",
                              "n_candidates": 2, "top_k": 5},
                        headers=HDR)
        assert r.status_code == 400

    def test_rate_limit_stops_a_scraper(self, client, monkeypatch):
        """Generation is the endpoint that costs money, so it is the one capped.

        `_run_pipeline` is replaced rather than the event loop: patching
        asyncio's loop accessor breaks Starlette's own use of it and the test
        hangs instead of failing.
        """
        import app.api as api
        monkeypatch.setattr(api, "_run_pipeline", lambda job, request: None)
        codes = [client.post("/api/generate", json=GEN, headers=HDR).status_code
                 for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert codes[3] == 429 and codes[4] == 429
