"""The progress stream has to keep sending, even when there is no progress.

Progress messages are coarse. Between "generating cover images" and the
scoring line there is one silent stretch covering every diffusion pass, which
is minutes. A proxy reading a connection with no bytes on it concludes the
origin has died and hangs up: RunPod fronts pods with Cloudflare, which gives
up after 100 seconds, so a demo would fail there while the GPU was still
working perfectly.

These check the keepalive is emitted while a job is running and that it stops
once there is nothing left to wait for.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest


@pytest.fixture()
def api(monkeypatch):
    monkeypatch.setenv("GC_ACCESS_TOKEN", "test-token")
    monkeypatch.delenv("GC_LOG_CHOICES", raising=False)
    import app.auth as auth
    import app.api as api_mod
    importlib.reload(auth)
    importlib.reload(api_mod)
    api_mod._JOBS.clear()
    return api_mod


def _running_job(api, job_id="job-1"):
    job = api.Job(
        job_id=job_id,
        status="running",
        progress=["Generating cover images…"],
        results=[],
        request_params={},
    )
    api._JOBS[job_id] = job
    return job


async def _collect(api, job, job_id, ticks):
    """Drive the stream for `ticks` simulated seconds, collecting what it sends.

    asyncio.sleep is replaced so the test does not actually wait, and the
    counter is what decides when to stop rather than wall clock time.
    """
    frames: list[str] = []
    elapsed = {"n": 0}

    real_sleep = asyncio.sleep

    async def fake_sleep(_seconds):
        elapsed["n"] += 1
        if elapsed["n"] >= ticks:
            job.status = "done"
            job.public_results = []
        await real_sleep(0)

    asyncio.sleep = fake_sleep
    try:
        response = await api.stream_job(job_id, token="test-token")
        async for chunk in response.body_iterator:
            frames.append(chunk)
    finally:
        asyncio.sleep = real_sleep
    return frames


class TestHeartbeat:
    def test_keepalive_is_sent_while_a_job_is_quiet(self, api):
        job = _running_job(api)
        frames = asyncio.run(_collect(api, job, "job-1", ticks=35))

        beats = [f for f in frames if f == api._HEARTBEAT]
        # 35 quiet seconds at one beat every ten leaves three.
        assert len(beats) == 3, frames

    def test_the_gap_between_beats_stays_under_the_proxy_limit(self, api):
        # Cloudflare, which is what RunPod puts in front of a pod, gives up at
        # 100 seconds of silence. Anything at or above that is not a margin.
        assert api._HEARTBEAT_SECONDS < 100
        assert api._HEARTBEAT_SECONDS <= 30

    def test_the_keepalive_is_invisible_to_the_page(self, api):
        # EventSource discards lines that open with a colon, so this must not
        # be parseable as an event or the client would see junk messages.
        assert api._HEARTBEAT.startswith(":")
        assert not api._HEARTBEAT.startswith("data:")
        assert api._HEARTBEAT.endswith("\n\n")

    def test_progress_resets_the_timer_rather_than_adding_to_it(self, api):
        """A job that keeps reporting should not also emit keepalives."""
        job = _running_job(api)

        real_sleep = asyncio.sleep
        elapsed = {"n": 0}

        async def chatty_sleep(_seconds):
            elapsed["n"] += 1
            # Something to say every other second, so the quiet counter never
            # reaches the threshold.
            job.progress.append(f"step {elapsed['n']}")
            if elapsed["n"] >= 30:
                job.status = "done"
                job.public_results = []
            await real_sleep(0)

        async def run():
            asyncio.sleep = chatty_sleep
            try:
                response = await api.stream_job("job-1", token="test-token")
                return [c async for c in response.body_iterator]
            finally:
                asyncio.sleep = real_sleep

        frames = asyncio.run(run())
        assert api._HEARTBEAT not in frames
