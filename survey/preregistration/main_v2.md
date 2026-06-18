# OSF Pre-registration — Greeting Cards Main Survey v2

**Study title:** Pairwise saleability ratings of birthday greeting cards
**Researcher:** Raghav Gagan, MSc thesis
**Pre-registration date:** *to be timestamped on OSF submission*
**Planned data collection start:** *post-IRB-amendment approval*

## 1. Hypotheses (predictor evaluation)

H1. After training on the pairwise survey data, the saleability head of the
multi-headed predictor will achieve Spearman ρ ≥ 0.4 against held-out
Bradley-Terry purchase-intent scores (one-sided, α=0.05).

H2. The aesthetic head will achieve Spearman ρ ≥ 0.4 against held-out BT
aesthetic scores (one-sided, α=0.05).

H3 (descriptive only). LLM pseudo-labels (`label_source='llm_pseudo_v1'`) for
the occasion_fit, emotional_resonance, and distinctiveness heads will agree
with a 40-card human-rated calibration subset at Spearman ρ ≥ 0.5; if not,
those heads will be trained on the proxy alone and the limitation reported.

## 2. Design

- Within-subject 2AFC pairwise comparisons across a birthday-only card pool.
- 150 cards, 4 birthday sub-occasions (`birthday/general`, `birthday/milestone`,
  `birthday/kids`, `birthday/relationship`), 6 tone classes.
- Two questions per pair: purchase intent (primary), aesthetic (secondary).
- Each card appears 4–8 times across the study via active-learning pair queue
  (`survey/instrument/sampler.py:sample_pairs_main`).
- First 40 participant completes treated as warm-up cohort; release of
  remaining ~110 slots contingent on:
  - BT convergence (max |Δs| < 1e-6 within 500 MM iterations on 40-cohort data)
  - Bootstrap rank correlation ρ ≥ 0.7 between two halves of the warm-up data
  - Attention-pair failure rate < 10%
- Birthday-only scope enforced by `common/occasions.is_valid_occasion`.

## 3. Sample size and justification

n = 150 (effective ≈ 135 after exclusions). Justification: Monte Carlo
simulation in `eval/sims/bt_power.py` (pinned config: 30 replicates of the
n=150 / 60 pairs-per-participant / 150 cards design; results written to
`eval/sims/bt_power_report.json`):

- Expected BT rank-recovery Spearman ρ vs ground truth: mean 0.97
  (5th percentile 0.96) — BT scores are recovered with high fidelity.
- Power to detect a held-out predictor with true ρ_pred = 0.4 against the
  BT ranking: 0.70 at α=0.05 (one-sided) on a 30% held-out card subset.
- Power to detect a true ρ_pred = 0.3: 0.50.

Pre-launch we will rerun with n_sims=200 and pin the resulting report file as
the canonical power calc, replacing the n_sims=30 placeholder. We will not
lower n=150 from these numbers; if the higher-n_sims run shows power < 0.70
at ρ_pred=0.4 we will pre-emptively widen the held-out predictor synthesis
noise model and document the update before launch.

## 4. Card pool composition

| Slice | n |
|---|---|
| Marketplace, `proxy_v1` low tertile | ≈33 |
| Marketplace, `proxy_v1` mid tertile | ≈33 |
| Marketplace, `proxy_v1` high tertile | ≈33 |
| System-generated (mixed pipeline conditions) | 40 |
| Naive baseline (raw SDXL, no LoRA, no layout) | 10 |
| **Total** | **149–150** |

Stratified across the 4 birthday sub-occasions and 6 tone classes.

## 5. Exclusion criteria (pre-registered)

A participant's responses are **excluded from analysis but still paid** if any of:

- Failed ≥ 2 of 3 trapdoor pairs (a card paired with a known-broken variant)
- Median pair-time < 3 s across the session
- Did not complete all 60 pairs within the Prolific session window
- Submitted same answer for ≥ 90% of pairs (straight-lining)

A pair is **excluded from BT fitting** if:

- Either side is a trapdoor card (by construction; never persisted as a real comparison)
- Either side has missing image or text content
- Either side has been retracted from the pool post-launch (e.g., DMCA / copyright
  takedown of a marketplace listing)

## 6. Analysis plan

**Primary.** Fit Bradley-Terry MM (`survey/analysis/bradley_terry.py`) per question
dimension on the union of warm-up + main pairs. Persist as
`saleability_labels` with `label_source='survey_main_v2_bt'`.

Train predictor v2 (`models/predictor/train.py`) with:
- Saleability head: BT purchase-intent scores
- Aesthetic head: BT aesthetic scores
- Sub-score heads (occasion_fit, emotional_resonance, distinctiveness): LLM
  pseudo-labels from `data/labels/pseudo_labels.py`

Hold out 15% of cards (by seller_id stratification) for evaluation.
Report Spearman ρ + 95% bootstrap CIs per head.

**Secondary.** Per sub-occasion breakdowns; calibration ECE; sample-size
sensitivity by re-fitting BT on random 50% sub-samples.

## 7. Inference

H1 / H2: one-sided test on bootstrap distribution of Spearman ρ vs the lower
threshold (0.4). Reject H_0 if 5th percentile of bootstrap distribution > 0.4.

H3: descriptive — report observed correlations with 95% CIs.

## 8. Deviations from prior plan

This v2 supersedes the v1 (Likert) main protocol. Substantive changes:

- Instrument switched from 7-point Likert to 2AFC pairwise
- Sample size reduced from n=300 to n=150 (power justified above)
- Sub-score heads now LLM-pseudo-labelled, not human-rated
- Scope restricted to birthday cards (matches code-level `ACTIVE_OCCASIONS`)
- Pilot rolled into main as 40-participant warm-up cohort

All changes are pre-data-collection. An IRB amendment covering the Likert →
2AFC instrument change must be on file before launch
(`survey/ethics/irb_amendment_v2.md`).

## 9. Materials

- Survey instrument: `survey/instrument/app.py` (routes `/pairsurvey`, `/pair/*`)
- Pair sampler: `survey/instrument/sampler.py:sample_pairs_main`
- BT analysis: `survey/analysis/bradley_terry.py`
- Power simulation: `eval/sims/bt_power.py`
- Card pool snapshot: committed under `data/survey_pools/main_v2/`
- Anonymised pair-graph CSV will be released alongside the thesis.

## 10. Researcher degrees of freedom — locked decisions

- Tie handling: ties counted as 0.5 wins to each side in BT fitting
- BT prior strength: `prior_strength=0.1` (Beta(0.1, 0.1) pseudocount per pair)
- MM stopping criterion: max |Δs| < 1e-6 or 500 iterations
- Bootstrap resamples: 5,000 with stratified resampling by sub-occasion
- Decision threshold for H1/H2: 0.4 (matches the design-doc target)
