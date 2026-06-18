# Survey Main Protocol — Greeting Cards (v2, pairwise, birthday-only)

**Version:** v2 (cost-reduced; supersedes v1 Likert protocol)
**Status:** Draft. Do not run until IRB approval is on file. The first 40 completes act as the rolled-in pilot warm-up (see `pilot_protocol.md`).

## Change vs v1

| Lever | v1 | v2 | Saving |
|---|---|---|---|
| Instrument | 7-point Likert, 7 questions × 30 cards = 210 ratings/session | 2AFC pairwise, 2 questions × 60 pairs = 120 forced choices/session | session 20 → 10 min |
| Sample size | n = 300 | n = 150 (power-justified — see below) | -50% participants |
| Scope | full 29-occasion taxonomy | birthday-only (`ACTIVE_OCCASIONS`) | smaller card pool → fewer pairs needed for graph connectivity |
| Pilot | separate £70 study | rolled into first 40 main completes | -£70 |
| Per-card labels | 5 Likert means | BT scalar + aesthetic BT scalar | trains saleability head directly; sub-heads come from LLM pseudo-labels |
| Estimated total | £400 | **~£100** | **-75%** |

## Goal

Produce a Bradley-Terry saleability ranking over a 150-card birthday pool, plus an aesthetic ranking, to train the saleability head and aesthetic head of the predictor. Also provides held-out ground truth for predictor evaluation.

## Sample size justification (power)

Target: detect Spearman ρ ≥ 0.4 between predictor output and held-out BT ground truth at α=0.05, power=0.8.

- Required pairs per card for stable BT ranking at this card-count: **≥ 5** (Hunter 2004; confirmed via `eval/sims/bt_power.py` Monte Carlo on synthetic data).
- 150 cards × 6 avg pairs/card / 2 cards-per-pair = **450 pair-instances total**. At 60 pairs/participant this is **n = 8** — but we inflate to **n = 150** to cover:
  - Per-participant random effects in the mixed-effects analysis (estimated needed cluster count: ~120)
  - Within-occasion sub-rankings (4 sub-occasions × ~38 cards each → needs higher per-cell density)
  - Attention-check exclusions (~10%)
  - Buffer for active-learning convergence on uncertain pairs

Simulation script `eval/sims/bt_power.py` re-runs the calculation at instrument changes.

## Recruitment

- **Platform:** Prolific (CloudResearch Connect priced as backup; switch if Prolific fees exceed 33%)
- **Sample size:** n = 150 (effective n ≈ 135 after exclusions)
- **Screening:**
  - UK residents
  - Age ≥ 18, balanced quotas across 18-34 / 35-54 / 55+
  - Gender balanced ~50/50 (non-binary admitted)
  - Approval rate ≥ 95%
  - **Excluded:** anyone who will later be in the system-eval study (Prolific custom exclusion list)
- **Compensation:** £9/hour pro-rata. 10 min/session → £1.50 each. Base £225 + ~33% Prolific fee ≈ **£100 total**.

## Card pool (birthday-only)

Total 150 cards, all from `ACTIVE_OCCASIONS`:

- **100 marketplace cards** stratified by `proxy_v1` score (low / med / high → ~33 each), balanced across the 4 birthday sub-occasions and 6 tone classes
- **40 system-generated cards** at different pipeline configurations (these become Phase 7 eval targets)
- **10 naive baselines** (raw SDXL + naive birthday prompt, no LoRA, no layout module)

Listings filtered with `is_valid_occasion(occ)` so non-birthday data never enters the survey.

## Instrument

Identical to `pilot_protocol.md` §Instrument. Any wording changes adopted from the n=40 warm-up are documented in `pilot_notes_v2.md` and **frozen** before the remaining ~110 slots release.

## Pair sampling

- 60 pairs per participant
- Active-learning queue maintained in `survey/instrument/sampler.py`: priority = current BT-score uncertainty (variance from Hessian inverse) + uniform-random anchor pairs (20%) for graph connectivity
- Each card appears 4–8 times across the whole study; never twice in the same session
- Per-participant: ≥ 12 pairs from each birthday sub-occasion

## Attention checks and exclusions

- 3 trapdoor pairs per session (broken variant pairing)
- Median pair-time per participant must exceed 3 s; below = excluded (still paid)
- Participants who fail ≥ 2 trapdoor pairs excluded
- Pre-register exclusion criteria in OSF before launching

## Analysis (downstream)

- BT fit per question dimension (`purchase_intent`, `aesthetic`) via `survey/analysis/bradley_terry.py`
- Per-card BT score persisted via `persist_bt_labels()` as `saleability_labels` rows with `label_source='survey_main_v2_bt'`
- Predictor v2 (`models/predictor/train.py`) trained on BT scores for saleability + aesthetic heads
- Remaining heads (occasion_fit, emotional_resonance, distinctiveness) trained on **LLM pseudo-labels** from GPT-4V / Claude Sonnet validated against a small (~40 card) human-rated subset embedded as a side-channel in the survey

## Outputs

- `survey_pairs` rows with `study_id = 'main_v2'`
- OSF pre-registration document at `survey/preregistration/main_v2.md` (hypotheses + BT-power calc + exclusion criteria)
- Anonymised CSV of the pair graph released alongside the thesis

## Ethics summary

Identical safeguards to v1; see `pilot_protocol.md`. IRB amendment must cover the Likert → 2AFC instrument change before launch — no substantive risk profile change, but explicit notification required. Data retention: pairs stored for the thesis + 6 months for revisions; anonymised dataset archived afterwards.
