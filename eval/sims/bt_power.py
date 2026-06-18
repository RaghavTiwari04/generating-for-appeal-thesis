"""Monte Carlo power simulation for the pairwise (BT) survey design.

Justifies the n=150 / 60-pairs-per-participant design in `main_protocol.md`
by simulating ground-truth saleability scores → BT pair outcomes → predictor
fits, and measuring:

  1. Rank-recovery: Spearman ρ between fitted BT scores and ground truth.
  2. Predictor-vs-BT power: probability that a held-out predictor with true
     Spearman ρ_pred achieves observed ρ ≥ τ at α = 0.05.

Run:
    python -m eval.sims.bt_power --n-cards 150 --n-participants 150 \
                                 --pairs-per-participant 60 --n-sims 200

Outputs a small JSON report next to this file (or to --out).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from survey.analysis.bradley_terry import fit_bradley_terry


@dataclass
class SimResult:
    n_cards: int
    n_participants: int
    pairs_per_participant: int
    n_sims: int
    rank_corr_mean: float
    rank_corr_p05: float
    rank_corr_p95: float
    power_detect_rho_pred_0_4: float
    power_detect_rho_pred_0_3: float
    median_n_iter: float


def _simulate_one(
    *,
    rng: np.random.Generator,
    n_cards: int,
    n_participants: int,
    pairs_per_participant: int,
    tie_prob: float,
    response_noise: float,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Run one synthetic study; return (Spearman ρ vs truth, n_iter, fitted, truth)."""
    true_s = rng.normal(0, 1, size=n_cards)

    # Build pairs: each participant draws K random pairs from the card pool
    rows = []
    for _ in range(n_participants):
        pair_idx = rng.choice(n_cards, size=(pairs_per_participant, 2))
        for i, j in pair_idx:
            if i == j:
                continue
            s_i = true_s[i] + rng.normal(0, response_noise)
            s_j = true_s[j] + rng.normal(0, response_noise)
            p_i = 1.0 / (1.0 + np.exp(s_j - s_i))
            u = rng.random()
            if u < tie_prob:
                winner = "T"
            elif rng.random() < p_i:
                winner = "L"
            else:
                winner = "R"
            rows.append(
                {
                    "left_key": f"c{i}",
                    "right_key": f"c{j}",
                    "winner_side": winner,
                    "attention_check_pass": True,
                }
            )

    df = pd.DataFrame(rows)
    result = fit_bradley_terry(df, prior_strength=0.1, max_iter=500, tol=1e-5)

    fitted = np.zeros(n_cards)
    for k, card_key in enumerate(result.card_keys):
        fitted[int(card_key[1:])] = result.scores[k]

    rho = float(spearmanr(fitted, true_s).statistic)
    return rho, result.n_iter, fitted, true_s


def _power_for_rho(
    fitted: np.ndarray, true_s: np.ndarray, target_rho_pred: float, n_held_out: int, rng: np.random.Generator
) -> float:
    """Simulate a synthetic predictor with planned Spearman correlation `target_rho_pred`
    against ground truth; check whether the *observed* held-out Spearman is significant
    at α=0.05 under a one-sided test.

    Held-out sample = `n_held_out` cards drawn uniformly without replacement.
    Synthetic predictor = α·true_s + β·noise with α/β chosen to hit target_rho_pred.
    """
    n = len(true_s)
    n_held_out = min(n_held_out, n)
    idx = rng.choice(n, size=n_held_out, replace=False)

    noise = rng.normal(0, 1, size=n_held_out)
    # Construct predictor with desired correlation to true_s on the held-out set
    z_true = (true_s[idx] - true_s[idx].mean()) / true_s[idx].std(ddof=0)
    z_noise = (noise - noise.mean()) / noise.std(ddof=0)
    alpha = target_rho_pred
    beta = np.sqrt(max(0.0, 1.0 - alpha ** 2))
    predictor = alpha * z_true + beta * z_noise

    res = spearmanr(predictor, fitted[idx])
    rho, p = float(res.statistic), float(res.pvalue)
    return float((p / 2 < 0.05) and (rho > 0))


def run(
    *,
    n_cards: int,
    n_participants: int,
    pairs_per_participant: int,
    n_sims: int,
    tie_prob: float = 0.10,
    response_noise: float = 0.5,
    held_out_frac: float = 0.3,
    seed: int = 42,
) -> SimResult:
    rng = np.random.default_rng(seed)
    n_held_out = max(20, round(n_cards * held_out_frac))

    rank_corrs: list[float] = []
    iters: list[int] = []
    detect_at_04: list[float] = []
    detect_at_03: list[float] = []

    for _s in range(n_sims):
        rho, n_iter, fitted, truth = _simulate_one(
            rng=rng,
            n_cards=n_cards,
            n_participants=n_participants,
            pairs_per_participant=pairs_per_participant,
            tie_prob=tie_prob,
            response_noise=response_noise,
        )
        rank_corrs.append(rho)
        iters.append(n_iter)
        detect_at_04.append(_power_for_rho(fitted, truth, 0.4, n_held_out, rng))
        detect_at_03.append(_power_for_rho(fitted, truth, 0.3, n_held_out, rng))

    rc = np.array(rank_corrs)
    return SimResult(
        n_cards=n_cards,
        n_participants=n_participants,
        pairs_per_participant=pairs_per_participant,
        n_sims=n_sims,
        rank_corr_mean=float(rc.mean()),
        rank_corr_p05=float(np.percentile(rc, 5)),
        rank_corr_p95=float(np.percentile(rc, 95)),
        power_detect_rho_pred_0_4=float(np.mean(detect_at_04)),
        power_detect_rho_pred_0_3=float(np.mean(detect_at_03)),
        median_n_iter=float(np.median(iters)),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cards", type=int, default=150)
    p.add_argument("--n-participants", type=int, default=150)
    p.add_argument("--pairs-per-participant", type=int, default=60)
    p.add_argument("--n-sims", type=int, default=100)
    p.add_argument("--tie-prob", type=float, default=0.10)
    p.add_argument("--response-noise", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("eval/sims/bt_power_report.json"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run(
        n_cards=args.n_cards,
        n_participants=args.n_participants,
        pairs_per_participant=args.pairs_per_participant,
        n_sims=args.n_sims,
        tie_prob=args.tie_prob,
        response_noise=args.response_noise,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
