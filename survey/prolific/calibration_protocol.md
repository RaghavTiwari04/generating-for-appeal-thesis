# VLM Calibration Study — Prolific Protocol

## Purpose
Validate VLM-generated saleability labels against human pairwise preferences.
If Spearman rho >= 0.5 per dimension, VLM labels are accepted as training
signal for the multi-headed predictor.

## Study Design

| Parameter | Value |
|-----------|-------|
| Design | 2AFC pairwise comparison |
| Dimensions | 5 (occasion_fit, aesthetic, emotional_resonance, distinctiveness, saleability) |
| Pairs per participant | 25 (+ 2 trapdoor attention checks) |
| Target N participants | 40 |
| Total pairs | ~1,000 |
| Total judgments | ~5,000 (5 dims x 1,000 pairs) |
| Pair sampling | VLM-score-stratified (40% cross-tercile, 40% adjacent, 20% within) |
| Est. time per participant | 3-4 minutes |
| Payment | £0.30 per participant (£9/hr rate) |
| Total study cost | ~£12 |

## Prolific Settings

- **Title**: "Compare Birthday Greeting Cards (3 min)"
- **Description**: "Look at pairs of birthday cards and tell us which is better on 5 quick questions. No right or wrong answers — just your opinion."
- **Estimated completion time**: 3 minutes
- **Reward per participant**: £0.30
- **Total participants**: 40
- **Eligibility**:
  - Location: United Kingdom (UK market focus)
  - Fluent English
  - Approval rate >= 95%
  - No previous participation in this project
- **Study URL**: `https://<YOUR_DOMAIN>/calibration?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID=calibration_v1&SESSION_ID={{%SESSION_ID%}}`
- **Completion URL**: Set via `PROLIFIC_COMPLETION_CODE_CALIBRATION` env var

## Attention Checks

- 2 trapdoor pairs per participant (top-5% vs bottom-5% VLM score)
- Expected: obvious winner on most dimensions
- Exclusion criterion: >= 3 ties on a trapdoor pair
- Excluded participants replaced

## Analysis Pipeline

```bash
# After data collection:
python -m survey.analysis.vlm_calibration validate \
    --study-id calibration_v1 \
    --vlm-source vlm_5head_v1
```

Outputs:
- `survey/analysis/calibration_report.json` — per-dimension rho + p-values
- `survey/analysis/calibration_report.txt` — human-readable summary

## Acceptance Criteria (Preregistered)

Per dimension, Spearman rho(VLM, human BT) >= 0.5.

If a dimension fails:
- Report in thesis with full transparency
- Fall back to proxy-only training for that head
- Document as limitation

## Timeline

1. Run VLM labeling (~2 hours, ~£30) ← PREREQUISITE
2. Deploy survey server
3. Create Prolific study
4. Collect data (~1 day)
5. Run analysis
6. Write thesis §4.4
