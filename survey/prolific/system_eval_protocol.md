# Survey System-Evaluation Protocol — Greeting Cards (v2, pairwise, birthday-only)

**Version:** v2 (cost-reduced; supersedes v1)
**Status:** Draft. Run only after IRB approval, after the generation pipeline is feature-complete, and after the main rating study has been analysed.

## Change vs v1

| Lever | v1 | v2 | Saving |
|---|---|---|---|
| Instrument | Likert per card, 4 conditions × 8 cards = 32 cards/session | 2AFC pairwise across conditions, ~50 pairs/session | session 25 → 12 min |
| Sample size | n = 200 | n = 100 | -50% |
| Conditions tested | 4 (A naive, B no-rerank, C pipeline+rerank, D bestsellers) | 4, but pair sampler over-samples the **decision-critical pairs** (C vs A, C vs B, C vs D), C↔A symmetric pairs etc. | tighter targeting → fewer pairs needed |
| Scope | full taxonomy | birthday-only | smaller stratification load |
| Total cost | ~£1000 | **~£250** | **-75%** |

## Goal

Provide the headline thesis result: a blinded, within-subject **pairwise** comparison of four conditions to test the three pre-registered hypotheses, restricted to birthday cards.

## Recruitment

- **Platform:** Prolific
- **Sample size:** n = 100 (effective ≈ 90 after exclusions)
- **Screening:** identical to main rating study; **must exclude** every participant from the main rating study and the warm-up pilot (Prolific custom exclusion list)
- **Compensation:** £9/hour pro-rata. ~12 min/session → £1.80 each. ≈ £180 base + Prolific fee → **~£250 total**

## Conditions (birthday-only)

| Tag | Description |
|---|---|
| A | Naive AI: SDXL with naive birthday prompt, no LoRA, no layout module, LLM message with naive prompt |
| B | Pipeline without rerank: full pipeline (brief + birthday LoRA + ControlNet + layout + message), N=1 |
| C | Pipeline with rerank: full pipeline, predictor-driven best-of-N with N=8 |
| D | Human bestsellers: top-rated marketplace bestsellers for birthday occasions (`proxy_v1` top-decile within `ACTIVE_OCCASIONS`) |

40 cards per condition (10 per birthday sub-occasion), 160 cards total. Same brief inputs across A/B/C; D matched on occasion/sub-occasion.

## Instrument

Each participant judges **~50 pairs** in one ~12-minute session.

For each pair:
- Two card covers side-by-side at equal resolution on neutral grey background; source-identifying metadata (URLs, watermarks) stripped from images
- Headline + inside message rendered below each card
- Occasion context line shown once at the top of the pair
- Left/Right side randomised per pair
- Condition labels **hidden** from participant

Two forced-choice questions per pair: purchase intent (primary) + aesthetic (secondary). Trapdoor pairs (~3/session) as in main study.

## Pair sampling

Across the full study (n=100 × 50 pairs = 5,000 pair instances):

- 60% **decision-critical pairs** distributed across the three pre-registered contrasts:
  - C vs A: ~1,000 pairs
  - C vs B: ~1,000 pairs
  - C vs D: ~1,000 pairs
- 30% **within-condition** anchor pairs (250 per condition) — needed so BT scores are comparable across the conditions, not just relative within each
- 10% trapdoors + active-learning uncertainty picks

Matched-pair design: every C↔X comparison uses cards generated from the **same brief** where possible (A/B/C share briefs; D matched on occasion + sub-occasion). This converts the analysis to a within-brief contrast and increases statistical power per pair.

## Pre-registered hypotheses (OSF)

- **H1** (one-sided): P(C wins | C vs A) > 0.5 — purchase intent
- **H2** (one-sided): P(C wins | C vs B) > 0.5 — purchase intent
- **H3** (two-sided equivalence): P(C wins | C vs D) within [0.4, 0.6] under a TOST equivalence test (margin = 10 percentage points around indifference)

## Statistical analysis

Primary: Bradley-Terry-Davidson model (handles ties) fitted with condition as a fixed effect, participant + brief as random effects. Pairwise contrasts on the four condition scores with Holm correction over 6 pairs. Equivalence test (H3) via TOST on the C-vs-D win probability.

Secondary: per-sub-occasion breakdowns (4 birthday cells), aesthetic-dimension BT scores, free-text thematic coding (`eval/failure_analysis.py`) — free text collected only on the bottom-quintile pairs by BT confidence.

## Outputs

- `survey_pairs` rows with `study_id = 'system_eval_v2'`
- OSF pre-registration document checked in at `survey/preregistration/system_eval_v2.md` before launch
- Figures: condition BT-score means + 95% CIs, per-sub-occasion grid, ablation curve from `eval/ablations/best_of_n_curve.py`

## Ethics summary

Identical safeguards to the main study. Generated birthday cards manually screened for incidental recognisable real-person likenesses before launch (no human likenesses expected from birthday prompts but verified). IRB amendment same as main study v2 — Likert → 2AFC instrument change.
