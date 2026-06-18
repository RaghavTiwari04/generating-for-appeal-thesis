# Survey Pilot Protocol — Greeting Cards (v2, pairwise, birthday-only)

**Version:** v2 (cost-reduced; supersedes v1 Likert protocol)
**Status:** Draft (pre-ethics-approval). Do not run until IRB approval is on file.

## Change vs v1

- **2AFC pairwise** comparisons instead of per-card 7-point Likert. ~3× more comparisons per minute, lower variance, no scale-anchor drift.
- **Birthday-only** scope (matches `common/occasions.ACTIVE_OCCASIONS`).
- **Two questions only** per pair: purchase intent (primary), aesthetic (secondary).
- **Rolled into main study as warm-up batch** — no separate Prolific submission. The first 30–50 main-study sessions are analysed before the rest are released, replacing the previous standalone pilot.

Effective budget: **£0 incremental** (was ~£70). Cost folded into main-study budget; total still below v1 main alone.

## Goal

Validate the pairwise instrument, confirm comparison time per pair ≤ 8s, and check that Bradley-Terry scores stabilise at the planned comparison density before releasing the remaining main-study slots.

## Recruitment

- **Platform:** Prolific
- **Sample size:** first 40 completes of the main study (n=40 warm-up cohort, billed against main budget)
- **Screening:** identical to main study (`main_protocol.md`)
- **Compensation:** as main study (£9/hr pro-rata; 10-min session ≈ £1.50/participant)

## Card pool

- Pool of ~150 birthday cards drawn from scraped marketplace listings, stratified by `proxy_v1` score (low / med / high), constrained to `ACTIVE_OCCASIONS` (`birthday/general`, `birthday/milestone`, `birthday/kids`, `birthday/relationship`)
- Tones balanced across `warm-sincere`, `warm-humorous`, `funny-irreverent`, `formal-sincere`, `minimalist`, `sentimental`

## Instrument

Each participant judges **~60 pairs** in one ~10-minute session (~10s per pair including read time).

For each pair, present:
- Two card covers side-by-side at equal resolution on neutral grey (#f4f4f4)
- Headline + inside message rendered below each card
- Occasion context shown once at the top of the pair: "Imagine you're shopping for a [occasion] card for [recipient]"
- Left/Right side of each card randomised per pair

Questions per pair (2):
1. **Purchase intent** (forced choice): "Which card would you be more likely to buy for this occasion?" — answers: Left / Right / Hard to choose
2. **Aesthetic** (forced choice): "Which card looks more visually appealing?" — answers: Left / Right / About the same

Attention checks: 3 trapdoor pairs per session (a card paired with an obviously-broken variant; if participant picks the broken one twice or more they are excluded but paid).

## Pair sampling (`survey/instrument/sampler.py`)

- For each participant, sample 60 pairs from the 150-card pool such that:
  - Each card appears 4–8 times across the **whole study**, never twice in the same session
  - Pairs are drawn from a TrueSkill-style active-learning queue: prefer pairs whose current BT-score uncertainty is highest, plus a small uniform-random fraction (~20%) for graph connectivity
- Forced occasion balance: ≥ 12 pairs from each birthday sub-occasion per participant

## Analysis (pilot warm-up)

Triggered when warm-up cohort reaches n=40:

- BT fit on first 40 sessions (~2,400 pairs)
- Convergence check: BT log-likelihood plateau, max |Δs| < 1e-6 within 500 MM iterations
- Bootstrap 95% CIs on top-quartile vs bottom-quartile rank stability — target rank correlation ρ ≥ 0.7 between two halves of the data
- Attention-pair failure rate (target < 10%)
- Median pair-time histogram (flag participants with median < 3s)

If any of the above fail, **pause launch** and revise instrument before releasing the remaining ~110 slots.

## Outputs

- `survey_pairs` rows with `study_id = 'main_v2_warmup'` (then continues as `main_v2`)
- Instrument revision log at `survey/prolific/pilot_notes_v2.md`

## Ethics summary

Identical safeguards to v1 protocol. Pseudonymised Prolific ID only; right-to-withdraw before submission; no deception. The pairwise instrument exposes the same scraped marketplace stock + generated outputs already covered by the existing IRB approval; pre-submit an amendment noting the instrument change (Likert → 2AFC) — no substantive risk profile change.
