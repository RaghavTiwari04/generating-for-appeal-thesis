# OSF Pre-registration — Greeting Cards System Evaluation v2

**Study title:** Pairwise comparison of AI-generated vs human-designed birthday greeting cards
**Researcher:** Raghav Gagan, MSc thesis
**Pre-registration date:** *to be timestamped on OSF submission*
**Planned data collection start:** *post-main-survey + pipeline-complete*

## 1. Hypotheses

**H1** (one-sided): P(C beats A | purchase intent) > 0.5.
   *Pipeline with rerank beats naive AI generation.*

**H2** (one-sided): P(C beats B | purchase intent) > 0.5.
   *Predictor-driven reranking beats no rerank.*

**H3** (two-sided equivalence): P(C beats D | purchase intent) lies inside
[0.40, 0.60] under a TOST equivalence test.
   *Pipeline with rerank is not substantively worse than human-designed
   bestsellers on the birthday occasion.*

## 2. Conditions

| Tag | Description |
|---|---|
| A | Naive AI: SDXL with naive birthday prompt; no LoRA, no layout module, LLM message with naive prompt |
| B | Pipeline without rerank: full pipeline (brief + birthday LoRA + ControlNet + layout + message); N=1 |
| C | Pipeline with rerank: full pipeline + predictor-driven best-of-N reranking, N=8 |
| D | Human bestsellers: top-decile `proxy_v1` marketplace listings restricted to birthday occasions |

40 cards per condition (10 per birthday sub-occasion), 160 cards total.
A/B/C share briefs where possible; D matched on occasion + sub-occasion.

## 3. Design

Within-subject 2AFC pairwise comparison. Per-participant 50 pairs.
Pair sampling budget across the whole study (5,000 pair instances):

- 60% decision-critical (C vs A, C vs B, C vs D — 20% each)
- 30% within-condition anchors (~7.5% per condition)
- 10% trapdoor pairs

Implementation: `survey/instrument/sampler.py:sample_pairs_system_eval`.

## 4. Sample size

n = 100 (effective ≈ 90 after exclusions). Each decision-critical contrast
collects ≈ 1,000 pair-instances at this n. Under the matched-brief design
and a true effect of |Δ-win-prob| = 0.10 from indifference, two-sided
binomial CI half-width is ≈ 3 percentage points at 1,000 trials — sufficient
to power the H1/H2 detections and the H3 equivalence test at margin 0.10.

## 5. Exclusion criteria (pre-registered)

Participant excluded from analysis (still paid) if any of:

- Failed ≥ 2 of 3 trapdoor pairs
- Median pair-time < 3 s
- Did not complete all 50 pairs
- Straight-lined ≥ 90% of pairs (always picked Left, always picked Right, or
  always picked the tie option)
- Was already in the main rating study or warm-up cohort (Prolific custom
  exclusion list)

A pair is excluded from analysis if either side card was retracted post-launch.

## 6. Analysis plan

Bradley-Terry-Davidson (handles ties explicitly) with condition as fixed
effect; participant + brief as random effects. Fit two separate BT models —
one for purchase_intent, one for aesthetic.

Convert BT scores to pairwise win probabilities for the four pre-registered
contrasts. Apply Holm correction over six pairwise comparisons {A,B,C,D}^2.

H3 (equivalence): TOST with equivalence margin ε = 0.10 on the win-probability
scale (i.e. C-vs-D win-probability inside [0.40, 0.60]).

Per-sub-occasion breakdowns reported descriptively (no formal pre-reg test).

## 7. Materials

- Survey instrument: `survey/instrument/app.py` (study_id `system_eval_v2`)
- Pair sampler: `survey/instrument/sampler.py:sample_pairs_system_eval`
- BT analysis: `survey/analysis/bradley_terry.py`
- Card pool snapshot: committed under `data/survey_pools/system_eval_v2/`
- Anonymised pair-graph CSV will be released alongside the thesis.

## 8. Researcher degrees of freedom — locked decisions

- Tie handling: BT-Davidson with shared tie parameter
- BT prior strength: 0.1 (matches main survey)
- H1/H2 rejection: 95% bootstrap-CI lower bound on win-probability > 0.5
- H3 rejection: 90% TOST CI inside [0.4, 0.6]
- Bootstrap resamples: 5,000 with stratified resampling on (participant, contrast)
- Manual screen for incidental real-person likenesses in generated cards before launch
