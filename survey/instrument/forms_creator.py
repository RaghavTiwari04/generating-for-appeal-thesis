"""Create the Google Form for card ratings and print pre-fill entry IDs.

Run once before launching the survey:
    python -m survey.instrument.forms_creator

Prerequisites:
    1. Google Cloud project with Forms API + Sheets API + Drive API enabled.
    2. Service account with Editor role on the project; JSON key downloaded.
    3. Set GOOGLE_SERVICE_ACCOUNT_JSON in .env pointing to that key file.

What this script does:
    - Creates a new Google Form with all rating questions.
    - Fetches the question IDs needed for pre-filled URLs.
    - Prints the GOOGLE_FORMS_ID, GOOGLE_SHEETS_ID, and all ENTRY_* values
      to paste into .env.
    - Sets the form confirmation message to remind participants to return
      to the study server.

Re-running is safe: if GOOGLE_FORMS_ID is already set in .env, the script
skips creation and only re-prints the entry IDs.
"""

from __future__ import annotations

from pathlib import Path

from common.config import settings
from common.logging import get_logger

log = get_logger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]

_FORM_TITLE = "Greeting Card Rating Study"
_FORM_DESCRIPTION = (
    "Thank you for participating. You will rate one greeting card at a time. "
    "After submitting this form, return to the study page to view the next card."
)

_QUESTIONS = [
    # (field_name, title, type, extra)
    ("participant_id",      "Participant ID (do not edit)",        "SHORT_ANSWER", {}),
    ("card_key",            "Card ID (do not edit)",               "SHORT_ANSWER", {}),
    ("occasion",            "Occasion (do not edit)",              "SHORT_ANSWER", {}),
    ("study_id",            "Study ID (do not edit)",              "SHORT_ANSWER", {}),
    ("purchase_intent",     "How likely would you be to buy this card for the described occasion?",
                                                                   "SCALE",        {"low": "Very unlikely", "high": "Very likely"}),
    ("occasion_fit",        "How well does this card fit the occasion?",
                                                                   "SCALE",        {"low": "Poor fit",      "high": "Perfect fit"}),
    ("aesthetic",           "How visually appealing is this card?",
                                                                   "SCALE",        {"low": "Not appealing", "high": "Very appealing"}),
    ("emotional_resonance", "How well does this card capture the right feeling for the occasion?",
                                                                   "SCALE",        {"low": "Not at all",    "high": "Perfectly"}),
    ("distinctiveness",     "How original or distinctive is this card compared to others you've seen?",
                                                                   "SCALE",        {"low": "Very generic",  "high": "Very distinctive"}),
    ("max_price_gbp",       "What is the maximum you would pay for this card? (£)",
                                                                   "SCALE",        {"low": "£1",            "high": "£15",
                                                                                    "low_value": 1, "high_value": 15}),
    ("free_text",           "Optional: What works or doesn't work about this card? (skip if you prefer)",
                                                                   "PARAGRAPH",    {}),
]


def _build_item(field_name: str, title: str, qtype: str, extra: dict) -> dict:
    if qtype == "SHORT_ANSWER":
        return {
            "title": title,
            "questionItem": {
                "question": {
                    "required": False,
                    "textQuestion": {"paragraph": False},
                }
            },
        }
    if qtype == "PARAGRAPH":
        return {
            "title": title,
            "questionItem": {
                "question": {
                    "required": False,
                    "textQuestion": {"paragraph": True},
                }
            },
        }
    if qtype == "SCALE":
        low_val = extra.get("low_value", 1)
        high_val = extra.get("high_value", 7)
        return {
            "title": title,
            "questionItem": {
                "question": {
                    "required": True,
                    "scaleQuestion": {
                        "low": low_val,
                        "high": high_val,
                        "lowLabel": extra.get("low", ""),
                        "highLabel": extra.get("high", ""),
                    },
                }
            },
        }
    raise ValueError(f"Unknown question type: {qtype}")


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


def _create_form(service) -> str:
    """Create blank form, return form ID."""
    body = {
        "info": {
            "title": _FORM_TITLE,
            "documentTitle": _FORM_TITLE,
        }
    }
    result = service.forms().create(body=body).execute()
    form_id = result["formId"]
    log.info(f"Form created: {form_id}")
    return form_id


def _add_questions(service, form_id: str) -> None:
    """Batch-update the form to add all questions and set description."""
    requests = [
        {
            "updateFormInfo": {
                "info": {"description": _FORM_DESCRIPTION},
                "updateMask": "description",
            }
        }
    ]
    for i, (field_name, title, qtype, extra) in enumerate(_QUESTIONS):
        requests.append(
            {
                "createItem": {
                    "item": _build_item(field_name, title, qtype, extra),
                    "location": {"index": i},
                }
            }
        )

    service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()
    log.info("Questions added to form")


def _set_confirmation_message(service, form_id: str, return_url_hint: str) -> None:
    """Set the post-submission confirmation message."""
    msg = (
        "Thank you! Your rating has been recorded.\n\n"
        f"Please return to the study page to continue to the next card:\n{return_url_hint}"
    )
    service.forms().batchUpdate(
        formId=form_id,
        body={
            "requests": [
                {
                    "updateSettings": {
                        "settings": {
                            "quizSettings": {"isQuiz": False},
                        },
                        "updateMask": "quizSettings.isQuiz",
                    }
                }
            ]
        },
    ).execute()
    # Confirmation text is set via updateFormInfo
    service.forms().batchUpdate(
        formId=form_id,
        body={
            "requests": [
                {
                    "updateFormInfo": {
                        "info": {"confirmationMessage": msg},
                        "updateMask": "confirmationMessage",
                    }
                }
            ]
        },
    ).execute()


def _get_entry_ids(service, form_id: str) -> dict[str, str]:
    """Return {field_name: question_id} for all questions in the form."""
    form = service.forms().get(formId=form_id).execute()
    field_names = [q[0] for q in _QUESTIONS]
    entry_ids: dict[str, str] = {}
    for item, field_name in zip(form.get("items", []), field_names, strict=False):
        qi = item.get("questionItem", {})
        qid = qi.get("question", {}).get("questionId", "")
        if qid:
            entry_ids[field_name] = qid
    return entry_ids


def _get_linked_sheet_id(drive_service, form_id: str) -> str | None:
    """Find the Google Sheets spreadsheet that stores form responses."""
    result = drive_service.files().list(
        q="name contains 'Responses' and mimeType='application/vnd.google-apps.spreadsheet'",
        spaces="drive",
        fields="files(id,name)",
    ).execute()
    # The response sheet is linked by Drive; look for a file named "{FORM_TITLE} (Responses)"
    target = f"{_FORM_TITLE} (Responses)"
    for f in result.get("files", []):
        if f["name"] == target:
            return f["id"]
    return None


def main() -> None:
    from googleapiclient.discovery import build

    creds = _get_credentials()
    forms_service = build("forms", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    form_id = settings.google_forms_id
    if form_id:
        log.info(f"GOOGLE_FORMS_ID already set ({form_id}), skipping creation.")
    else:
        form_id = _create_form(forms_service)
        _add_questions(forms_service, form_id)
        _set_confirmation_message(
            forms_service, form_id,
            return_url_hint="<YOUR_SERVER_URL>/next/<session_token>"
        )

    entry_ids = _get_entry_ids(forms_service, form_id)
    sheets_id = settings.google_sheets_id or _get_linked_sheet_id(drive_service, form_id)

    print("\n" + "=" * 60)
    print("Paste these into your .env file:")
    print("=" * 60)
    print(f"GOOGLE_FORMS_ID={form_id}")
    print(f"GOOGLE_SHEETS_ID={sheets_id or '<check Google Drive — sheet may take a minute to appear>'}")
    for field_name, qid in entry_ids.items():
        env_key = f"GOOGLE_FORM_ENTRY_{field_name.upper()}"
        print(f"{env_key}={qid}")
    print("=" * 60)
    print(f"\nForm URL: https://docs.google.com/forms/d/{form_id}/edit")


if __name__ == "__main__":
    main()
