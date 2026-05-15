# Survey Main Protocol — Greeting Cards

**Version:** v1
**Status:** Draft. Do not run until IRB approval is on file **and** the pilot has been analysed and instrument refinements landed.

## Goal

Generate ~9,000 ratings on ~600 cards to (a) train the survey-supervised heads of the saleability predictor and (b) provide held-out ground truth for predictor evaluation.

(A separate main *system* evaluation cohort is described in `survey/prolific/system_eval_protocol.md` — that cohort is **disjoint** from this one via Prolific exclusion lists.)

## Recruitment

- **Platform:** Prolific
- **Sample size:** n = 300 (after exclusion of attention-check failures, target an effective n ≈ 270)
- **Screening:**
  - UK residents
  - Age ≥ 18, balanced quotas across 18-34 / 35-54 / 55+
  - Gender balanced ~50/50 (non-binary admitted)
  - Income brackets balanced where Prolific allows
  - Approval rate ≥ 95%
  - **Excluded:** anyone in the pilot study, anyone who will later be in the system-eval study (apply this as a Prolific custom exclusion)
- **Compensation:** £9/hour pro-rata. 20 min/session => £3.00 each. Base £900 + ~33% fee ≈ £400.

## Protocol

Identical instrument to the pilot (see `pilot_protocol.md` §Protocol). Any wording changes adopted from the pilot are documented in `pilot_notes.md` and **frozen** before main launch.

## Card sampling

The 600-card pool:

- **400 marketplace cards** stratified by proxy_v1 score (low / med / high), balanced across the canonical occasion taxonomy
- **150 system-generated cards** at different pipeline configurations (these become eval targets — see Phase 7)
- **50 naive baselines** (raw SDXL + naive occasion prompt, no LoRA, no layout module)

Each card receives **15 ratings on average, minimum 8**. Sampler must enforce the minimum.

Per-participant occasion balance: each participant sees ≥3 cards from each of the top 5 occasions plus 15 cards drawn from the long tail.

## Attention checks and exclusions

- 3 attention checks interleaved with normal items (one is a directed-response item; one is a low-effort flag; one is a time check)
- Median response time per participant must exceed 4s; participants under that floor are excluded from analysis (still paid)
- Participants who fail >1 attention check are excluded
- Pre-register exclusion criteria in OSF before launching

## Analysis (downstream)

- Aggregate ratings into per-card mean for each dimension; persist as `saleability_labels` with `label_source='survey_main'` (purchase intent) plus per-head survey labels
- ICC(3,k) reliability per dimension, reported in the predictor chapter
- Predictor v2 trained on these survey labels + proxy_v1 (see `models/predictor/train.py`)

## Outputs

- `survey_ratings` rows with `study_id = 'main_v1'`
- Pre-registration document on OSF including hypotheses, sample size, exclusion criteria, and analysis plan
- Anonymised CSV released alongside the thesis (Prolific ID → opaque participant code mapping deleted post-payment, per ethics requirement)

## Ethics summary

Identical safeguards to pilot; see `pilot_protocol.md`. Data retention: ratings stored for the duration of the thesis + 6 months for revisions, then anonymised version archived and source-bound personal data destroyed.
