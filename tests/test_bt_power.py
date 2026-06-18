"""Quick smoke test of the BT power simulator.

Runs a tiny configuration to confirm the simulator end-to-end without
spending real CI time on the n=150 / n_sims=200 production calc.
"""

from __future__ import annotations

from eval.sims.bt_power import SimResult, run


def test_power_sim_runs_small_config():
    result = run(
        n_cards=20,
        n_participants=15,
        pairs_per_participant=15,
        n_sims=3,
        tie_prob=0.05,
        response_noise=0.3,
        seed=0,
    )
    assert isinstance(result, SimResult)
    # Plausibility checks
    assert -1.0 <= result.rank_corr_mean <= 1.0
    assert 0.0 <= result.power_detect_rho_pred_0_4 <= 1.0
    assert 0.0 <= result.power_detect_rho_pred_0_3 <= 1.0
    assert result.median_n_iter >= 1


def test_power_sim_rank_recovery_positive_at_reasonable_density():
    """With moderate density we should recover non-trivial rank correlation."""
    result = run(
        n_cards=15,
        n_participants=30,
        pairs_per_participant=20,
        n_sims=5,
        tie_prob=0.0,
        response_noise=0.0,
        seed=1,
    )
    # Noise-free, dense → recovery should be strongly positive
    assert result.rank_corr_mean > 0.5
