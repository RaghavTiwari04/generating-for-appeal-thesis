# Survey Pilot Protocol — Greeting Cards

**Version:** v1
**Status:** Draft (pre-ethics-approval). Do not run until IRB approval is on file.

## Goal

Validate the rating instrument, anchor scales, calibrate task time, and surface obvious failure modes **before** spending main-study budget. Pilot data is also useful for instrument-level inter-rater reliability and as a sanity check on the saleability proxy.

## Recruitment

- **Platform:** Prolific
- **Sample size:** n = 50
- **Screening:**
  - UK residents (jurisdictional and stylistic alignment with the corpus)
  - Age ≥ 18
  - Balanced on gender (target ~50/50 binary; allow non-binary if available)
  - Approval rate ≥ 95% on Prolific
- **Compensation:** £9/hour pro-rata. Estimated 20 minutes => £3.00 per participant. Total: ~£150 base + ~33% Prolific fee (~£50) = ~£70 minus screening dropoff.

## Protocol

Each participant rates **30 cards** in one session. Session targets 20 minutes (~40 seconds per card).

For each card, present:
- Cover image at full resolution on a neutral grey (#f4f4f4) background
- Occasion context line: "Imagine you're shopping for a [occasion] card for [recipient]"
- Headline text (rendered) and inside message (rendered)

Then ask, in this order:

1. **Purchase intent** (1–7 Likert): "How likely would you be to buy this card for the described occasion?"
2. **Occasion fit** (1–7 Likert): "How well does this card fit the occasion?"
3. **Aesthetic appeal** (1–7 Likert): "How visually appealing is this card?"
4. **Emotional resonance** (1–7 Likert): "How well does this card capture the right feeling for the occasion?"
5. **Distinctiveness** (1–7 Likert): "How original or distinctive is this card compared to others you've seen?"
6. **Maximum price** (£ slider, £1–£15): "Given the design and quality, what is the maximum you would pay for this card?"
7. **Optional free text** (≤ 200 chars): "What works or doesn't work about this card?" (skippable)

Pre-session: short demographic + consent screen (no PII beyond Prolific ID).
Post-session: 3 attention-check questions interleaved throughout the 30 cards (e.g., "For this question, select 'Strongly disagree' regardless of the card"). Participants failing more than one attention check are excluded from analysis but still paid per Prolific policy.

## Card sampling

- 30 cards per participant
- Stratified by predicted-proxy score (low / med / high), 10 each
- Balanced across the top 5 occasions in the corpus
- Each card targeted to receive ~5 ratings across the pilot (50 × 30 / 30 unique ≈ 50 cards × 30 ratings; iterate sampler to converge on 5 each)

## Analysis (pilot only)

- Distribution checks: any ceiling/floor effects on each Likert?
- Time-on-task histogram; flag <3s responses as candidates for exclusion
- Inter-rater reliability: ICC(3,k) on purchase intent, target ≥ 0.5
- Attention-check pass rate (target > 90%)
- Free-text thematic skim: any systematic confusion about instructions?

## Outputs

- `survey_ratings` rows with `study_id = 'pilot_v1'`, `label_source = 'survey_pilot'`
- An instrument revision note (`survey/prolific/pilot_notes.md`) capturing any anchor / wording changes for the main study

## Ethics summary (for IRB attachment)

- Anonymous Prolific IDs only; no images/text uploaded by participants
- Free-text replies inspected by researcher; published only in aggregate or with explicit redaction
- Participants can withdraw at any time and request data deletion before session end
- No deception
- Cards are scraped marketplace stock + generated outputs (no participant-identifying content)
