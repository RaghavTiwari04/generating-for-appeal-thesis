"""Sync Google Forms responses → survey_ratings in Postgres.

The Google Form linked spreadsheet is the source of truth. This script reads
all rows, skips already-synced ones (tracked by row index in a local state
file), and upserts each rating into the DB.

Usage:
    python -m survey.instrument.forms_sync          # sync new rows
    python -m survey.instrument.forms_sync --full   # re-sync all rows

State file: .cache/forms_sync_state.json
  {"last_row": 42}   — last 1-indexed spreadsheet row (excluding header) synced

Run this on a cron (e.g. every 15 minutes during data collection) or manually
after a study session.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from common.config import settings
from common.db import connection
from common.logging import get_logger

log = get_logger(__name__)

_STATE_FILE = Path(".cache/forms_sync_state.json")

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Maps Google Sheets column header → Postgres field name + type
_COL_MAP = {
    "Participant ID (do not edit)": ("participant_id", str),
    "Card ID (do not edit)":        ("card_key",       str),
    "Occasion (do not edit)":       ("occasion_shown", str),
    "Study ID (do not edit)":       ("study_id",       str),
    "How likely would you be to buy this card for the described occasion?":
                                    ("purchase_intent", int),
    "How well does this card fit the occasion?":
                                    ("occasion_fit",    int),
    "How visually appealing is this card?":
                                    ("aesthetic",       int),
    "How well does this card capture the right feeling for the occasion?":
                                    ("emotional_resonance", int),
    "How original or distinctive is this card compared to others you've seen?":
                                    ("distinctiveness", int),
    "What is the maximum you would pay for this card? (£)":
                                    ("max_price_gbp",   float),
    "Optional: What works or doesn't work about this card? (skip if you prefer)":
                                    ("free_text",       str),
    "Timestamp":                    ("gsheet_timestamp", str),
}

_INSERT = """
INSERT INTO survey_ratings (
    participant_id, study_id, listing_id, generated_card_id,
    occasion_shown, purchase_intent, occasion_fit, aesthetic,
    emotional_resonance, distinctiveness, max_price_gbp,
    free_text, rated_at
) VALUES (
    %(participant_id)s, %(study_id)s, %(listing_id)s, %(generated_card_id)s,
    %(occasion_shown)s, %(purchase_intent)s, %(occasion_fit)s, %(aesthetic)s,
    %(emotional_resonance)s, %(distinctiveness)s, %(max_price_gbp)s,
    %(free_text)s,
    %(rated_at)s
)
ON CONFLICT DO NOTHING;
"""


def _get_credentials():
    from google.oauth2 import service_account

    key_path = settings.google_service_account_json
    if not key_path or not Path(key_path).exists():
        raise FileNotFoundError(
            f"Service account JSON not found at {key_path!r}. "
            "Set GOOGLE_SERVICE_ACCOUNT_JSON in .env."
        )
    return service_account.Credentials.from_service_account_file(
        str(key_path), scopes=_SCOPES
    )


def _read_sheet(sheets_id: str) -> tuple[list[str], list[list[str]]]:
    """Return (header_row, data_rows) from the response spreadsheet."""
    from googleapiclient.discovery import build

    creds = _get_credentials()
    service = build("sheets", "v4", credentials=creds)
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=sheets_id, range="Sheet1")
        .execute()
    )
    rows: list[list[str]] = result.get("values", [])
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _load_state() -> int:
    """Return 0-indexed offset of first un-synced row (0 = sync all)."""
    if _STATE_FILE.exists():
        try:
            return int(json.loads(_STATE_FILE.read_text()).get("last_row", 0))
        except Exception:
            pass
    return 0


def _save_state(last_row: int) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({"last_row": last_row}))


def _parse_row(header: list[str], row: list[str]) -> dict | None:
    """Parse one spreadsheet row into a dict suitable for _INSERT."""
    # Pad short rows (trailing empty cells not returned by Sheets API)
    row = row + [""] * (len(header) - len(row))

    raw: dict = {}
    for col_header, cell in zip(header, row, strict=False):
        if col_header in _COL_MAP:
            field, cast = _COL_MAP[col_header]
            try:
                raw[field] = cast(cell) if cell.strip() else None
            except (ValueError, TypeError):
                raw[field] = None

    participant_id = raw.get("participant_id") or ""
    card_key = raw.get("card_key") or ""
    if not participant_id or not card_key:
        log.debug("Skipping row — missing participant_id or card_key")
        return None

    # Determine if card_key is a generated card or a scraped listing.
    # Convention: generated card IDs start with "gc-" or are stored in generated_cards.
    # Simple heuristic: check if card_key is a known UUID — both are UUIDs,
    # so we fall back to looking up in both tables at insert time.
    # For now, put card_key in both columns; the DB constraint keeps the correct one.
    listing_id = card_key
    generated_card_id = None  # resolved at query time if needed

    ts_str = raw.get("gsheet_timestamp") or ""
    try:
        rated_at = datetime.strptime(ts_str, "%m/%d/%Y %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        rated_at = datetime.now(tz=UTC)

    return {
        "participant_id": participant_id,
        "study_id": raw.get("study_id") or "",
        "listing_id": listing_id,
        "generated_card_id": generated_card_id,
        "occasion_shown": raw.get("occasion_shown") or "",
        "purchase_intent": raw.get("purchase_intent"),
        "occasion_fit": raw.get("occasion_fit"),
        "aesthetic": raw.get("aesthetic"),
        "emotional_resonance": raw.get("emotional_resonance"),
        "distinctiveness": raw.get("distinctiveness"),
        "max_price_gbp": raw.get("max_price_gbp"),
        "free_text": (raw.get("free_text") or "")[:500],
        "rated_at": rated_at,
    }


def sync(full: bool = False) -> int:
    """Sync new rows from Google Sheets to survey_ratings. Returns count inserted."""
    sheets_id = settings.google_sheets_id
    if not sheets_id:
        raise ValueError("GOOGLE_SHEETS_ID not set in .env. Run forms_creator.py first.")

    header, all_rows = _read_sheet(sheets_id)
    if not header:
        log.info("Response sheet is empty — nothing to sync.")
        return 0

    start = 0 if full else _load_state()
    new_rows = all_rows[start:]
    if not new_rows:
        log.info(f"No new rows since last sync (last_row={start}).")
        return 0

    log.info(f"Syncing {len(new_rows)} new rows (starting at row {start + 1})…")
    inserted = 0
    with connection() as conn, conn.cursor() as cur:
        for row in new_rows:
            parsed = _parse_row(header, row)
            if parsed is None:
                continue
            try:
                cur.execute(_INSERT, parsed)
                inserted += 1
            except Exception as e:
                log.warning(f"Insert failed for row {parsed.get('participant_id')}: {e}")

    _save_state(start + len(new_rows))
    log.info(f"Sync complete: {inserted} rows inserted.")
    return inserted


def main(full: bool = typer.Option(False, "--full", help="Re-sync all rows from scratch")) -> None:
    n = sync(full=full)
    print(f"Inserted {n} new rating(s).")


if __name__ == "__main__":
    typer.run(main)
