"""FastAPI survey instrument tests using TestClient (no DB, no MinIO)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Patch DB + sampler before importing app so no real connections happen
@pytest.fixture(scope="module")
def client():
    from survey.instrument.sampler import CardAssignment

    dummy_card = CardAssignment(
        card_key="test-card-001",
        is_generated=False,
        condition_tag=None,
        occasion="birthday/general",
        cover_path="s3://bucket/images/test.png",
        headline="Happy Birthday!",
        inside_message="Wishing you all the best.",
    )

    with patch("survey.instrument.app.sample_main", return_value=[dummy_card]), \
         patch("survey.instrument.app.sample_system_eval", return_value=[dummy_card]), \
         patch("survey.instrument.app._insert_rating", return_value=None), \
         patch("survey.instrument.app._presign", return_value="/static/placeholder.png"):
        from survey.instrument.app import app as _app
        yield TestClient(_app, raise_server_exceptions=True)


class TestSurveyRoutes:
    def test_survey_start_redirects(self, client: TestClient) -> None:
        resp = client.get(
            "/survey",
            params={"PROLIFIC_PID": "p001", "STUDY_ID": "main_v1", "SESSION_ID": "s001"},
            follow_redirects=False,
        )
        assert resp.status_code in (302, 307)
        assert "/card/" in resp.headers["location"]

    def test_card_page_renders(self, client: TestClient) -> None:
        # Start session first
        resp = client.get(
            "/survey",
            params={"PROLIFIC_PID": "p002", "STUDY_ID": "main_v1", "SESSION_ID": "s002"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "birthday" in body.lower() or "card" in body.lower()

    def test_card_page_has_form_fields(self, client: TestClient) -> None:
        resp = client.get(
            "/survey",
            params={"PROLIFIC_PID": "p003", "STUDY_ID": "main_v1", "SESSION_ID": "s003"},
            follow_redirects=True,
        )
        assert "purchase_intent" in resp.text
        assert "aesthetic" in resp.text
        assert "distinctiveness" in resp.text

    def test_rate_submission_redirects_to_next(self, client: TestClient) -> None:
        # Create session
        start = client.get(
            "/survey",
            params={"PROLIFIC_PID": "p004", "STUDY_ID": "main_v1", "SESSION_ID": "s004"},
            follow_redirects=False,
        )
        token = start.headers["location"].split("/card/")[-1]

        # Submit rating
        resp = client.post(
            f"/rate/{token}",
            data={
                "card_key": "test-card-001",
                "purchase_intent": "5",
                "occasion_fit": "4",
                "aesthetic": "6",
                "emotional_resonance": "5",
                "distinctiveness": "3",
                "max_price_gbp": "4.5",
                "free_text": "Lovely card",
                "response_time_ms": "8000",
            },
            follow_redirects=False,
        )
        assert resp.status_code in (302, 303, 307)

    def test_done_page_renders(self, client: TestClient) -> None:
        # Create + exhaust session (1 card)
        start = client.get(
            "/survey",
            params={"PROLIFIC_PID": "p005", "STUDY_ID": "main_v1", "SESSION_ID": "s005"},
            follow_redirects=False,
        )
        token = start.headers["location"].split("/card/")[-1]

        client.post(
            f"/rate/{token}",
            data={
                "card_key": "test-card-001",
                "purchase_intent": "4",
                "occasion_fit": "4",
                "aesthetic": "4",
                "emotional_resonance": "4",
                "distinctiveness": "4",
                "max_price_gbp": "3.0",
                "response_time_ms": "7000",
            },
            follow_redirects=False,
        )
        done = client.get(f"/done/{token}")
        assert done.status_code == 200
        assert "prolific" in done.text.lower() or "thank" in done.text.lower()

    def test_expired_session_returns_410(self, client: TestClient) -> None:
        resp = client.get("/card/nonexistent-session-token")
        assert resp.status_code == 410

    def test_system_eval_study_uses_correct_sampler(self, client: TestClient) -> None:
        with patch("survey.instrument.app.sample_system_eval") as mock_se, \
             patch("survey.instrument.app.sample_main") as mock_main:
            mock_se.return_value = []
            mock_main.return_value = []
            client.get(
                "/survey",
                params={
                    "PROLIFIC_PID": "p006",
                    "STUDY_ID": "system_eval_v1",
                    "SESSION_ID": "s006",
                },
                follow_redirects=False,
            )
            mock_se.assert_called_once()
            mock_main.assert_not_called()


class TestResponseQuality:
    """Test the quality-check module directly (no DB needed)."""

    def _make_df(self, n_participants: int = 5, n_cards: int = 10) -> "pd.DataFrame":
        import pandas as pd
        import numpy as np

        rng = np.random.default_rng(42)
        rows = []
        for pid in [f"p{i}" for i in range(n_participants)]:
            for _ in range(n_cards):
                rows.append({
                    "participant_id": pid,
                    "purchase_intent": int(rng.integers(1, 8)),
                    "occasion_fit": int(rng.integers(1, 8)),
                    "aesthetic": int(rng.integers(1, 8)),
                    "emotional_resonance": int(rng.integers(1, 8)),
                    "distinctiveness": int(rng.integers(1, 8)),
                    "attention_check_pass": True,
                    "response_time_ms": int(rng.integers(4000, 30000)),
                })
        return pd.DataFrame(rows)

    def test_clean_data_no_exclusions(self) -> None:
        from survey.analysis.response_quality import quality_report
        df = self._make_df()
        report = quality_report(df, "test")
        assert report.n_excluded == 0

    def test_fast_responder_excluded(self) -> None:
        import pandas as pd
        from survey.analysis.response_quality import quality_report

        df = self._make_df()
        # Make participant p0 answer too fast for > 20% of cards
        df.loc[df["participant_id"] == "p0", "response_time_ms"] = 500
        report = quality_report(df, "test")
        assert "p0" in report.excluded_ids()

    def test_attention_failure_excluded(self) -> None:
        import pandas as pd
        from survey.analysis.response_quality import quality_report

        df = self._make_df()
        # p1 fails 2 attention checks
        mask = df["participant_id"] == "p1"
        df.loc[mask, "attention_check_pass"] = [
            False, False, True, True, True, True, True, True, True, True
        ]
        report = quality_report(df, "test")
        assert "p1" in report.excluded_ids()

    def test_straight_liner_excluded(self) -> None:
        import pandas as pd
        from survey.analysis.response_quality import quality_report

        df = self._make_df()
        df.loc[df["participant_id"] == "p2", "aesthetic"] = 4  # all 4s
        report = quality_report(df, "test")
        assert "p2" in report.excluded_ids()

    def test_apply_exclusions_removes_rows(self) -> None:
        import pandas as pd
        from survey.analysis.response_quality import apply_exclusions, quality_report

        df = self._make_df()
        df.loc[df["participant_id"] == "p3", "response_time_ms"] = 200
        report = quality_report(df, "test")
        clean = apply_exclusions(df, report)
        assert "p3" not in clean["participant_id"].values
