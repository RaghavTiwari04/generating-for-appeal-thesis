# Pre-Registration: System Evaluation Study
**Study:** AI-Generated Greeting Cards — End-to-End Evaluation
**Version:** v1 | **Date registered:** [fill on OSF submission day]
**OSF project:** [fill URL after submission — submit before running `make system-eval`]
**Corresponding researcher:** Gagandeep Singh, gagan708344@gmail.com

---

## 1. Hypotheses

Pre-registered before any data collection. All tests two-tailed unless noted.

| ID | Hypothesis | Test | Direction |
|----|------------|------|-----------|
| H1 | Mean purchase intent: Condition C > Condition A | Pairwise contrast from mixed-effects model, Holm corrected | one-sided |
| H2 | Mean purchase intent: Condition C > Condition B | As above | one-sided |
| H3 | Condition C is not significantly worse than Condition D (human bestsellers), within equivalence margin ε = 0.5 Likert points | Two one-sided t-tests (TOST) on marginal means | two-sided TOST |

**Primary outcome:** Mean purchase intent (1–7 Likert) per condition.
**Secondary outcomes:** occasion_fit, aesthetic, emotional_resonance, distinctiveness per condition; per-occasion breakdowns.

---

## 2. Study design

- **Between/within:** within-subject (each participant rates cards from all 4 conditions)
- **Blinding:** participants blinded to source; researcher blinded to condition assignment during data collection
- **Randomisation:** card order randomised per participant via `sampler.py` (stable seed = SHA256 of participant_id + study_id)
- **n:** 200 Prolific participants (see §4 for power analysis)
- **Cards per participant:** 32 (8 per condition × 4 conditions)
- **Occasions covered:** 8 (birthday/general, christmas/general, mothers_day, valentines_day, sympathy/bereavement, thank_you, graduation, anniversary/general)

---

## 3. Conditions

| Condition | Label | Description |
|-----------|-------|-------------|
| A | Naive AI | SDXL base, naive occasion prompt, no LoRA, no layout module, naive LLM inside message |
| B | Pipeline (no rerank) | Full pipeline (brief LLM → LoRA-conditioned SDXL → ControlNet → layout composer → message LLM), N=1 |
| C | Pipeline + rerank | Same as B but best-of-N with N=8, reranked by calibrated saleability predictor |
| D | Human bestsellers | Top-proxy-ranked real marketplace listings matched to same occasions |

---

## 4. Power analysis

Pilot ICC(3,1) on purchase intent: assumed ρ ≈ 0.35 (conservative for new instrument).
Expected effect size C vs A: d = 0.45 (medium; based on analogous human-eval studies of generative systems).
Mixed-effects power estimated via simulation (R `simr`): n = 200 participants achieves 80% power for d = 0.4 at α = 0.05 (Holm corrected across 6 pairwise contrasts).
TOST equivalence: n = 200 achieves 80% power to declare equivalence within ε = 0.5 at α = 0.05 if true difference ≤ 0.2.

---

## 5. Statistical model

Primary: linear mixed-effects model fitted in Python (`statsmodels.MixedLM`):

```
purchase_intent ~ C(condition) + (1 | participant_id) + (1 | card_key)
```

Pairwise contrasts extracted from fixed effects. Holm–Bonferroni correction over 6 planned pairs:
A vs B, A vs C, A vs D, B vs C, B vs D, C vs D.

H3 equivalence: two one-sided tests on the C vs D contrast, margin ε = 0.5.

Secondary analyses: same model per sub-dimension; per-occasion breakdown; no correction applied (exploratory).

---

## 6. Exclusion criteria

Pre-registered exclusions applied before any analysis:

1. Participant fails > 1 of 3 attention checks (directed-response items)
2. Median per-card response time < 3 000 ms for > 20% of their cards
3. All ratings on any one dimension are identical (straight-lining)
4. Participant already completed the main predictor rating study (Prolific exclusion list enforced at recruitment)

Excluded participants are replaced until n_effective = 200 post-exclusion.

---

## 7. Analysis code

Pre-registered analysis script: `eval/system_eval.py`, scaffold commit `72c1b9f4dba4c5b1667712dc91963988cf24b7fa`. Final analysis commit: [fill with `git rev-parse HEAD` on the day the system eval study launches].
Figures: `eval/reports/figures.py`.

---

## 8. Deviations policy

Any post-registration deviations (e.g. model changes, additional conditions) will be reported as unregistered analyses in a clearly labelled section of the thesis.

---

## 9. Data and materials availability

- Survey instrument code: `survey/instrument/` in this repository
- Anonymised ratings CSV: released alongside thesis submission
- Generated card images: subset of high/low scorers released; full set available on request
- Raw Prolific metadata: not released (personal data); Prolific ID → opaque code mapping destroyed post-payment
