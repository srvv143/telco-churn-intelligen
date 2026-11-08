import numpy as np
import pytest

from churnintel.business import (
    RetentionAction,
    break_even_probability,
    compare_policies,
    expected_value,
    next_best_action,
    optimal_threshold,
    policy_summary,
    realized_value,
    risk_tier,
)

CALL = RetentionAction("call", "call", contact_cost=25, incentive_cost=60, save_rate=0.35)
EMAIL = RetentionAction("email", "email", contact_cost=2, incentive_cost=20, save_rate=0.12)


def test_expected_value_matches_hand_calculation():
    # p=0.5, CLV=700: 0.5*0.35*(700-60) - 25 - 0.5*60 = 112 - 25 - 30 = 57
    assert expected_value(0.5, 700, CALL) == pytest.approx(57.0)


def test_realized_value_credits_only_churners():
    rv = realized_value(np.array([1, 0]), np.array([700, 700]), CALL)
    assert rv[0] == pytest.approx(0.35 * 640 - 25)
    assert rv[1] == pytest.approx(-85.0)


def test_expected_value_is_mean_of_realized():
    p, clv = 0.3, 500.0
    ev = (
        p * realized_value(np.array([1]), np.array([clv]), CALL)[0]
        + (1 - p) * realized_value(np.array([0]), np.array([clv]), CALL)[0]
    )
    assert expected_value(p, clv, CALL) == pytest.approx(ev)


def test_break_even_probability_zeroes_ev():
    clv = np.array([300.0, 800.0])
    p0 = break_even_probability(clv, CALL)
    assert np.allclose(expected_value(p0, clv, CALL), 0.0, atol=1e-9)


def test_next_best_action_prefers_no_action_when_unprofitable():
    idx, ev = next_best_action(np.array([0.01, 0.9]), np.array([100.0, 900.0]), [CALL, EMAIL])
    assert idx[0] == -1 and ev[0] == 0.0
    assert idx[1] == 0 and ev[1] > 0


def test_optimal_threshold_is_argmax():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    clv = rng.uniform(200, 900, 500)
    t, grid, prof = optimal_threshold(p, y, clv, CALL)
    assert prof.max() == pytest.approx(realized_value(y, clv, CALL)[p >= t].sum())


def test_policy_summary_accounting_identity():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    clv = rng.uniform(100, 900, 300)
    idx = rng.integers(-1, 2, 300)
    s = policy_summary("x", idx, y, clv, [CALL, EMAIL])
    assert s["retained_value"] - s["campaign_cost"] == pytest.approx(s["net_profit"])
    manual = sum(realized_value(y[idx == k], clv[idx == k], a).sum() for k, a in enumerate([CALL, EMAIL]))
    assert s["net_profit"] == pytest.approx(manual)


def test_compare_policies_oracle_dominates():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.4 * y + rng.random(400) * 0.6, 0, 1)
    clv = rng.uniform(200, 900, 400)
    table = compare_policies(p, y, clv, rng.random(400), [CALL, EMAIL], "call", 0.4)
    assert table.loc["No campaign", "net_profit"] == 0
    oracle = table.loc["Oracle (perfect foresight, reference)", "net_profit"]
    assert (table["net_profit"] <= oracle + 1e-9).all()


def test_risk_tiers():
    assert list(risk_tier([0.1, 0.3, 0.7])) == ["low", "medium", "high"]
