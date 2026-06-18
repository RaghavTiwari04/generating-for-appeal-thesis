# IRB Amendment — Greeting Cards Survey v2 (Likert → 2AFC Pairwise)

**Submission type:** Amendment to existing approved protocol
**Principal Investigator:** Raghav Gagan (MSc thesis)
**Supervisor:** *to be inserted*
**Existing approval reference:** *to be inserted*
**Amendment date:** 2026-05-25

## 1. Summary of change

The approved protocol used a 7-point Likert rating instrument on individual
greeting cards. This amendment replaces that instrument with a **two-alternative
forced-choice (2AFC) pairwise comparison** instrument, and reduces the planned
sample size accordingly. Scope is restricted to birthday cards only
(unchanged in spirit — already a subset of the approved scope, now formalised
in code via `common/occasions.ACTIVE_OCCASIONS`).

## 2. Rationale

- **Lower participant burden.** Sessions shorten from ~20 min to ~10–12 min.
- **Higher data quality per minute.** Pairwise comparisons avoid Likert-scale
  anchoring drift, have lower individual-rater bias, and produce more
  comparisons per unit time. (Reference: standard psychometric finding;
  see Stewart, Brown & Chater, *Psych Rev*, 2005.)
- **Lower total budget.** Combined survey spend drops from ~£1,470 to ~£350,
  preserving research breadth within available funds.
- **Cleaner analysis.** Bradley-Terry-Davidson scaling on pair outcomes is
  better-suited to the predictor-training objective than averaged Likert
  scores.

The substantive purpose and risk profile of the study are unchanged.

## 3. What is changing

| Aspect | Approved | Amended |
|---|---|---|
| Instrument | 7-point Likert, 7 dimensions, 30 cards/session | 2AFC, 2 dimensions (purchase intent, aesthetic), 60 pairs/session |
| Main-study sample size | n = 300 | n = 150 |
| System-eval sample size | n = 200 | n = 100 |
| Pilot | Separate study, n = 50 | Rolled into first 40 main completes |
| Card scope | Full taxonomy (29 occasions) | Birthday only (4 sub-occasions) |
| Sub-score labels | All five dimensions from humans | Two from humans, three from LLM pseudo-labels |
| Compensation rate | £9/hour pro-rata | Unchanged (£9/hour pro-rata) |
| Compensation per session | £3.00 | £1.50 (main) / £1.80 (system eval) |
| Total budget | ~£1,470 | ~£350 |

## 4. What is **not** changing

- Risk profile: still low / no-risk consumer-research questionnaire.
- Stimuli: greeting card images and texts, drawn from publicly visible
  marketplace stock and system-generated outputs. No personal-identifying
  content shown to participants.
- Recruitment: Prolific, UK-resident, 18+, balanced quotas as before.
- Compensation rate: unchanged.
- Right to withdraw: unchanged. Participants may withdraw at any time before
  submission and request data deletion.
- Data handling: pseudonymised Prolific IDs only; mapping discarded after
  payment; data retention duration unchanged.
- Free-text responses: instrument no longer collects per-card free text in
  the main 2AFC flow. A single optional post-session text box is offered
  (≤300 chars) for general feedback.
- Attention checks: still present (3 trapdoor pairs per session in place of
  the previous directed-response Likert items).
- Information sheet and consent flow: unchanged except for the updated task
  description ("you will compare pairs of cards" instead of "you will rate
  cards individually").

## 5. Participant materials

The participant-facing information sheet and consent form are updated in
two places only:

1. **Task description** changes from "you will rate 30 greeting cards on a
   scale of 1–7" to "you will see 60 pairs of greeting cards, and for each
   pair you will choose which card you would prefer for a given occasion".
2. **Time estimate** changes from ~20 minutes to ~10–12 minutes; updated
   compensation amount reflected.

All other text (purpose, voluntariness, data protection, contact details,
withdrawal procedure) is retained verbatim.

## 6. Data protection

No change to data minimisation. Pair outcomes (left/right/tie choices,
response times, attention-check status) plus pseudonymised Prolific ID
are the only new data items.

The new database table `survey_pairs` (`migrations/0002_pairwise.sql`)
follows the same retention and access controls as the existing
`survey_ratings` table.

## 7. Risks and mitigations

- **Trapdoor pairs (broken card variants).** A small fraction (~5%) of pairs
  pit a normal card against a deliberately-broken variant of the same card
  (e.g. blank cover or nonsense headline). These are not deceptive in any
  research sense; they are standard attention checks. Participants are paid
  regardless of trapdoor outcome; failure only affects whether their data is
  included in the analysis.
- **LLM pseudo-labels.** The amendment introduces LLM-generated sub-score
  labels for predictor training. This is a backend processing step that does
  not involve participants and does not change participant-facing risk.

## 8. Requested approval

We request the ethics committee approve this amendment so that the v2
pairwise instrument may be deployed for the main and system-evaluation
studies. No data collection under the v2 instrument will commence until
written approval is on file.

## 9. Attachments

- Updated information sheet and consent form (track-changes)
- Updated `pilot_protocol.md`, `main_protocol.md`, `system_eval_protocol.md`
- Pre-registration documents `main_v2.md`, `system_eval_v2.md`
- Power-simulation report `eval/sims/bt_power_report.json`
