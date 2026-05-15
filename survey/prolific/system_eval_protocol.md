# Survey System-Evaluation Protocol — Greeting Cards

**Version:** v1
**Status:** Draft. Run only after IRB approval and after the generation pipeline is feature-complete.

## Goal

Provide the headline thesis result: a blinded, within-subject comparison of four conditions to test the three pre-registered hypotheses.

## Recruitment

- **Platform:** Prolific
- **Sample size:** n = 200 (effective ≈ 180 after exclusions)
- **Screening:** identical to main rating study; **must exclude** every participant from the main rating study and the pilot
- **Compensation:** £9/hour pro-rata. ~25 min/session => £3.75 each. ≈ £750 base + Prolific fee → ~£1000 total

## Conditions

| Condition | Description |
|---|---|
| A | Naive AI: SDXL with naive prompt, no LoRA, no layout module, LLM message with naive prompt |
| B | Pipeline without rerank: full pipeline (brief + LoRA + ControlNet + layout + message), N=1 |
| C | Pipeline with rerank: full pipeline, predictor-driven best-of-N with N=8 |
| D | Human bestsellers: top-rated marketplace bestsellers for the same occasions |

Each participant rates **32 cards** = 8 per condition × 4 conditions, **balanced across 8 occasions**. Order randomised within participant; source labels hidden. The cards used in A/B/C are pre-generated; D is sampled from `saleability_labels.proxy_v1` top-decile within occasion.

## Instrument

Same as `main_protocol.md` plus a post-card distractor question every ~6 items asking the participant to recall the previous card's occasion (further attention check). Source-identifying metadata (URLs, watermarks) stripped from images.

## Pre-registered hypotheses (OSF)

- **H1** (one-sided): mean purchase intent C > A
- **H2** (one-sided): mean purchase intent C > B
- **H3** (two-sided, equivalence): C is **not significantly worse** than D under a TOST equivalence test with margin ε = 0.5 Likert points

## Statistical analysis

Primary: mixed-effects model purchase_intent ~ condition + (1|participant) + (1|card). Pairwise contrasts with Holm correction over 6 pairs. Equivalence test for H3 separately via TOST.

Secondary: per-occasion breakdowns, per-dimension means, free-text thematic coding (see `eval/failure_analysis.py`).

## Outputs

- `survey_ratings` rows with `study_id = 'system_eval_v1'`
- OSF pre-registration document checked in at `survey/preregistration/system_eval_v1.md`
- Figures: condition means + 95% CIs, per-occasion grid, ablation curve from `eval/ablations/best_of_n_curve.py`

## Ethics summary

Identical safeguards to the main study. Generated cards have no human likenesses unless the diffusion model produces incidental faces; before launch the researcher manually screens out any card with recognisable real-person likenesses.
