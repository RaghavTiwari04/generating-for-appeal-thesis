"""Prolific API v1 client.

Thin wrapper around Prolific's REST API for programmatic study management:
- Create / publish studies
- Poll completion status
- Export participant submissions
- Apply exclusion lists (for disjoint cohorts across studies)

Auth: Bearer token from PROLIFIC_API_TOKEN env var.
Docs: https://docs.prolific.com/docs/api-docs/public/

Usage:
    client = ProlificClient()
    study = client.create_study(StudySpec(...))
    client.publish(study["id"])
    subs = client.wait_for_completion(study["id"], target_n=50)
    df = client.export_submissions(study["id"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)

BASE = "https://api.prolific.com/api/v1"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


@dataclass
class StudySpec:
    """Minimal study spec for the greeting-cards surveys."""
    name: str
    internal_name: str
    description: str
    external_study_url: str           # survey instrument URL with Prolific params
    prolific_id_option: str = "url_parameters"
    completion_option: str = "url"
    completion_codes: list[dict] = field(default_factory=lambda: [
        {"code": "PLACEHOLDER", "code_type": "COMPLETED", "actions": [{"action": "AUTOMATICALLY_APPROVE"}]}
    ])
    total_available_places: int = 50
    estimated_completion_time: int = 20   # minutes
    reward: int = 300                     # pence; £3.00 = 20 min @ £9/hr
    device_compatibility: list[str] = field(default_factory=lambda: ["desktop"])
    peripheral_requirements: list[str] = field(default_factory=list)
    # Eligibility filters
    filters: list[dict] = field(default_factory=lambda: [
        {"filter_id": "current-country-of-residence", "selected_values": ["GB"]},
        {"filter_id": "age", "value": {"minimum": 18}},
    ])
    # Exclusions: list of study IDs whose participants must not participate
    exclude_participants_from_studies: list[str] = field(default_factory=list)


class ProlificClient:
    def __init__(self, token: str | None = None):
        tok = token or settings.prolific_api_token
        if not tok:
            raise ValueError(
                "PROLIFIC_API_TOKEN not set. Add it to .env or pass token= explicitly."
            )
        self._client = httpx.Client(
            base_url=BASE,
            headers={"Authorization": f"Token {tok}", "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )

    def _get(self, path: str, **params) -> Any:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> Any:
        resp = self._client.post(path, json=body)
        resp.raise_for_status()
        return resp.json()

    def _patch(self, path: str, body: dict) -> Any:
        resp = self._client.patch(path, json=body)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Study lifecycle
    # ------------------------------------------------------------------

    def create_study(self, spec: StudySpec) -> dict:
        """Create a draft study. Returns the full study object."""
        body = {
            "name": spec.name,
            "internal_name": spec.internal_name,
            "description": spec.description,
            "external_study_url": spec.external_study_url,
            "prolific_id_option": spec.prolific_id_option,
            "completion_option": spec.completion_option,
            "completion_codes": spec.completion_codes,
            "total_available_places": spec.total_available_places,
            "estimated_completion_time": spec.estimated_completion_time,
            "reward": spec.reward,
            "device_compatibility": spec.device_compatibility,
            "peripheral_requirements": spec.peripheral_requirements,
            "filters": spec.filters,
        }
        if spec.exclude_participants_from_studies:
            body["exclude_participants_from_studies"] = spec.exclude_participants_from_studies
        study = self._post("/studies/", body)
        log.info(f"Study created: {study['id']} — {spec.name}")
        return study

    def publish(self, study_id: str) -> dict:
        result = self._post(f"/studies/{study_id}/transition/", {"action": "PUBLISH"})
        log.info(f"Study published: {study_id}")
        return result

    def pause(self, study_id: str) -> dict:
        return self._post(f"/studies/{study_id}/transition/", {"action": "PAUSE"})

    def stop(self, study_id: str) -> dict:
        return self._post(f"/studies/{study_id}/transition/", {"action": "STOP"})

    def get_study(self, study_id: str) -> dict:
        return self._get(f"/studies/{study_id}/")

    # ------------------------------------------------------------------
    # Submissions
    # ------------------------------------------------------------------

    def list_submissions(self, study_id: str) -> list[dict]:
        data = self._get(f"/studies/{study_id}/submissions/")
        return data.get("results", [])

    def export_submissions(self, study_id: str) -> "pd.DataFrame":
        import pandas as pd
        subs = self.list_submissions(study_id)
        return pd.DataFrame(subs)

    def n_completed(self, study_id: str) -> int:
        subs = self.list_submissions(study_id)
        return sum(1 for s in subs if s.get("status") == "APPROVED")

    def wait_for_completion(
        self,
        study_id: str,
        *,
        target_n: int,
        poll_interval_sec: int = 120,
        timeout_min: int = 180,
    ) -> list[dict]:
        """Block until `target_n` approved submissions or timeout."""
        deadline = time.monotonic() + timeout_min * 60
        while time.monotonic() < deadline:
            n = self.n_completed(study_id)
            log.info(f"Study {study_id}: {n}/{target_n} completed")
            if n >= target_n:
                return self.list_submissions(study_id)
            time.sleep(poll_interval_sec)
        raise TimeoutError(
            f"Study {study_id} did not reach {target_n} completions within {timeout_min} min"
        )

    # ------------------------------------------------------------------
    # Participant management
    # ------------------------------------------------------------------

    def get_participant_ids(self, study_id: str, status: str = "APPROVED") -> list[str]:
        subs = self.list_submissions(study_id)
        return [s["participant_id"] for s in subs if s.get("status") == status]

    def bonus_payment(self, study_id: str, participant_id: str, amount_pence: int) -> dict:
        return self._post(
            f"/studies/{study_id}/bonuses/pay/",
            {"csv_bonuses": f"{participant_id},{amount_pence / 100:.2f}"},
        )

    # ------------------------------------------------------------------
    # Exclusion list helpers
    # ------------------------------------------------------------------

    def build_exclusion_list(self, *study_ids: str) -> list[str]:
        """Return list of study IDs to pass as exclude_participants_from_studies."""
        return list(study_ids)


# ------------------------------------------------------------------
# Pre-built study specs for the thesis
# ------------------------------------------------------------------

def pilot_spec(instrument_url: str) -> StudySpec:
    return StudySpec(
        name="Greeting Cards — Pilot Rating Study",
        internal_name="gc_pilot_v1",
        description=(
            "You will be shown a series of greeting card designs and asked to rate "
            "each one on five dimensions (e.g. how visually appealing it is, how well "
            "it fits the occasion). Each session takes approximately 20 minutes."
        ),
        external_study_url=(
            f"{instrument_url}/survey?"
            "PROLIFIC_PID={{%PROLIFIC_PID%}}"
            "&STUDY_ID=pilot_v1"
            "&SESSION_ID={{%SESSION_ID%}}"
        ),
        total_available_places=55,   # 10% buffer for exclusions
        estimated_completion_time=20,
        reward=300,
    )


def main_rating_spec(instrument_url: str, exclude_from: list[str]) -> StudySpec:
    return StudySpec(
        name="Greeting Cards — Main Rating Study",
        internal_name="gc_main_v1",
        description=(
            "You will rate 30 greeting card designs across several occasions. "
            "Each session takes approximately 20 minutes. "
            "Do NOT participate if you took part in the pilot study."
        ),
        external_study_url=(
            f"{instrument_url}/survey?"
            "PROLIFIC_PID={{%PROLIFIC_PID%}}"
            "&STUDY_ID=main_v1"
            "&SESSION_ID={{%SESSION_ID%}}"
        ),
        total_available_places=330,
        estimated_completion_time=20,
        reward=300,
        exclude_participants_from_studies=exclude_from,
    )


def system_eval_spec(instrument_url: str, exclude_from: list[str]) -> StudySpec:
    return StudySpec(
        name="Greeting Cards — Design Evaluation",
        internal_name="gc_syseval_v1",
        description=(
            "You will compare 32 greeting card designs and rate their suitability "
            "for different occasions. Takes approximately 25 minutes. "
            "Do NOT participate if you took part in previous greeting card studies."
        ),
        external_study_url=(
            f"{instrument_url}/survey?"
            "PROLIFIC_PID={{%PROLIFIC_PID%}}"
            "&STUDY_ID=system_eval_v1"
            "&SESSION_ID={{%SESSION_ID%}}"
        ),
        total_available_places=220,
        estimated_completion_time=25,
        reward=375,
        exclude_participants_from_studies=exclude_from,
    )
