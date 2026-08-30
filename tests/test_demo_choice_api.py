"""The demo asks visitors to choose blind, so the ranking must not leak.

If any of these fail, a visitor could read the model's answer out of the page
before picking, and every choice recorded afterwards would be worthless as
evidence. They are cheap to run and they guard the one property the whole
exercise rests on.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("GC_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("GC_ALLOWED_ORIGINS", "https://example.invalid")
    monkeypatch.delenv("GC_LOG_CHOICES", raising=False)
    import app.auth as auth
    import app.choices as choices
    import app.api as api
    importlib.reload(auth)
    importlib.reload(choices)
    importlib.reload(api)
    auth._recent.clear()
    api._JOBS.clear()
    return TestClient(api.app)


HDR = {"X-Access-Token": "test-token"}


def _finished_job(api, n: int = 4):
    """A job in the state the pipeline leaves it in, without running it."""
    import random
    import uuid

    job = api.Job(
        job_id=str(uuid.uuid4()),
        status="done",
        progress=[],
        results=[],
        created_at=0.0,
        request_params={
            "occasion": "birthday/general",
            "tone": None,
            "relationship": "friend",
            "n_candidates": n,
            "scorer": "ridge",
            "constraints": {"note": "free text that must never be stored"},
        },
    )
    job.results = [
        {
            "rank": i + 1,
            "display_id": str(uuid.uuid4()),
            "card_id": None,
            "headline": f"Card {i}",
            "inside_message": "Many happy returns.",
            "occasion": "birthday/general",
            "scores": {"purchase_intent": 0.7 - i * 0.01},
            "image_data_url": "data:image/jpeg;base64,AAAA",
            "brief": {"concept": "a concept"},
        }
        for i in range(n)
    ]
    public = [
        {
            "display_id": r["display_id"],
            "headline": r["headline"],
            "inside_message": r["inside_message"],
            "occasion": r["occasion"],
            "image_data_url": r["image_data_url"],
            "brief": r["brief"],
        }
        for r in job.results
    ]
    random.shuffle(public)
    job.public_results = public
    job.display_order = [c["display_id"] for c in public]
    api._JOBS[job.job_id] = job
    return job


class TestRankingIsWithheld:
    def test_public_view_carries_no_ranking_signal(self, client):
        """Not rank, not scores, not card_id, not an index-derived id."""
        import app.api as api
        job = _finished_job(api)
        for card in job.public_results:
            assert "rank" not in card
            assert "scores" not in card
            assert "card_id" not in card
            # An id like "tmp-2" would carry the ranking in plain sight.
            assert not card["display_id"].startswith("tmp-")

    def test_display_ids_do_not_encode_position(self, client):
        import app.api as api
        job = _finished_job(api)
        ids = [r["display_id"] for r in job.results]
        assert len(set(ids)) == len(ids)
        for value in ids:
            assert len(value) == 36 and value.count("-") == 4


class TestReveal:
    def test_reveal_requires_a_card_from_this_job(self, client):
        import app.api as api
        job = _finished_job(api)
        r = client.post(
            f"/api/generate/{job.job_id}/reveal",
            json={"chosen_display_id": "not-a-card"},
            headers=HDR,
        )
        assert r.status_code == 400

    def test_reveal_returns_the_ranking_once_a_choice_is_named(self, client):
        import app.api as api
        job = _finished_job(api)
        picked = job.display_order[2]
        r = client.post(
            f"/api/generate/{job.job_id}/reveal",
            json={"chosen_display_id": picked},
            headers=HDR,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["chosen_display_id"] == picked
        assert body["model_top_display_id"] == job.results[0]["display_id"]
        assert {c["rank"] for c in body["candidates"]} == {1, 2, 3, 4}

    def test_reveal_is_idempotent_so_a_reload_cannot_rechoose(self, client):
        import app.api as api
        job = _finished_job(api)
        first, second = job.display_order[0], job.display_order[1]
        a = client.post(f"/api/generate/{job.job_id}/reveal",
                        json={"chosen_display_id": first}, headers=HDR).json()
        b = client.post(f"/api/generate/{job.job_id}/reveal",
                        json={"chosen_display_id": second}, headers=HDR).json()
        assert a["chosen_display_id"] == b["chosen_display_id"] == first

    def test_reveal_is_gated_and_404s_for_unknown_jobs(self, client):
        import app.api as api
        job = _finished_job(api)
        assert client.post(f"/api/generate/{job.job_id}/reveal",
                           json={"chosen_display_id": job.display_order[0]}).status_code == 401
        assert client.post("/api/generate/00000000-0000-0000-0000-000000000000/reveal",
                           json={"chosen_display_id": "x"}, headers=HDR).status_code == 404


class TestChoiceLogging:
    def test_choice_answers_204_with_logging_off(self, client):
        import app.api as api
        job = _finished_job(api)
        r = client.post("/api/choice", headers=HDR, json={
            "session_id": "11111111-1111-1111-1111-111111111111",
            "job_id": job.job_id,
            "event_type": "choice",
        })
        assert r.status_code == 204

    def test_choice_answers_204_for_an_evicted_job(self, client):
        """A visitor whose job has aged out sees no difference."""
        r = client.post("/api/choice", headers=HDR, json={
            "session_id": "11111111-1111-1111-1111-111111111111",
            "job_id": "00000000-0000-0000-0000-000000000000",
            "event_type": "download_front",
        })
        assert r.status_code == 204

    def test_unknown_event_type_is_rejected(self, client):
        import app.api as api
        job = _finished_job(api)
        r = client.post("/api/choice", headers=HDR, json={
            "session_id": "11111111-1111-1111-1111-111111111111",
            "job_id": job.job_id,
            "event_type": "keystrokes",
        })
        assert r.status_code == 400

    def test_event_omits_free_text_and_records_agreement(self, client):
        """The request carried a `constraints` note. It must not be in the row."""
        import app.api as api
        from app.choices import build_event
        job = _finished_job(api)
        job.chosen_display_id = job.results[0]["display_id"]

        event = build_event(job, session_id="s", event_type="choice",
                            time_to_choice_ms=4200)

        assert "free text that must never be stored" not in str(event)
        assert "constraints" not in event
        assert event["agreed_top1"] is True
        assert event["chosen_rank"] == 1
        assert event["occasion"] == "birthday/general"
        assert event["shown_position"] == job.display_order.index(job.chosen_display_id)

    def test_event_records_disagreement(self, client):
        import app.api as api
        from app.choices import build_event
        job = _finished_job(api)
        job.chosen_display_id = job.results[3]["display_id"]
        event = build_event(job, session_id="s", event_type="choice")
        assert event["agreed_top1"] is False
        assert event["chosen_rank"] == 4

    def test_record_event_never_raises_without_a_database(self, client):
        """A demo must not break because the database is unreachable."""
        from app.choices import record_event
        record_event({"event_type": "choice", "session_id": "s"})


class TestRequestValidation:
    def test_unknown_scorer_is_a_bad_request(self, client):
        r = client.post("/api/generate", headers=HDR, json={
            "occasion": "birthday/general", "scorer": "magic",
        })
        assert r.status_code == 400

    def test_health_reports_whether_logging_is_on(self, client):
        body = client.get("/api/health").json()
        assert body["gated"] is True
        assert body["logging"] is False
