"""Retention-campaign economics.

Profit decomposition (Neslin et al., 2006; Verbraken, Verbeke & Baesens, 2013). For a targeted
customer with churn probability ``p`` and value-if-retained ``CLV``, an action with contact cost
``c``, incentive cost ``d`` and save rate ``g`` yields:

* churner (prob. ``p``):       saved w.p. ``g`` -> ``+CLV - d``; contact always paid -> ``g(CLV - d) - c``
* non-churner (prob. ``1-p``): accepts the incentive anyway -> ``-(c + d)``

    EV = p * g * (CLV - d) - c - (1 - p) * d

Profit is incremental vs. doing nothing. **Realised** profit on a labelled set replaces ``p`` by the
true label ``y`` - unlike v3, value is only credited for customers who actually churn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RetentionAction:
    name: str
    label: str
    contact_cost: float
    incentive_cost: float
    save_rate: float

    @classmethod
    def from_config(cls, name: str, spec: dict[str, Any]) -> RetentionAction:
        return cls(
            name=name,
            label=str(spec.get("label", name)),
            contact_cost=float(spec["contact_cost"]),
            incentive_cost=float(spec["incentive_cost"]),
            save_rate=float(spec["save_rate"]),
        )

    def with_(self, **changes) -> RetentionAction:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "contact_cost": self.contact_cost,
            "incentive_cost": self.incentive_cost,
            "save_rate": self.save_rate,
        }


def load_actions(business_cfg: dict[str, Any]) -> list[RetentionAction]:
    return [RetentionAction.from_config(n, s) for n, s in business_cfg["actions"].items()]


def expected_value(p, clv, action: RetentionAction) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    clv = np.asarray(clv, dtype=float)
    g, c, d = action.save_rate, action.contact_cost, action.incentive_cost
    return p * g * (clv - d) - c - (1.0 - p) * d


def realized_value(y, clv, action: RetentionAction) -> np.ndarray:
    """Per-customer profit *if targeted*, using the true outcome instead of a probability."""
    y = np.asarray(y)
    clv = np.asarray(clv, dtype=float)
    g, c, d = action.save_rate, action.contact_cost, action.incentive_cost
    return np.where(y == 1, g * (clv - d) - c, -(c + d))


def break_even_probability(clv, action: RetentionAction) -> np.ndarray:
    """Churn probability above which targeting has positive expected value."""
    clv = np.asarray(clv, dtype=float)
    g, c, d = action.save_rate, action.contact_cost, action.incentive_cost
    denom = g * (clv - d) + d
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (c + d) / denom, np.inf)


def next_best_action(p, clv, actions: list[RetentionAction]) -> tuple[np.ndarray, np.ndarray]:
    """Per customer: index of the EV-maximising action (-1 = no action) and its EV."""
    ev = np.column_stack([expected_value(p, clv, a) for a in actions])
    best = ev.argmax(axis=1)
    best_ev = ev.max(axis=1)
    best = np.where(best_ev > 0, best, -1)
    return best, np.where(best >= 0, best_ev, 0.0)


def risk_tier(p, high: float = 0.5, medium: float = 0.25) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.where(p >= high, "high", np.where(p >= medium, "medium", "low"))


def optimal_threshold(p, y, clv, action: RetentionAction, grid: np.ndarray | None = None):
    """Grid-search the single probability threshold that maximises realised profit."""
    p = np.asarray(p, dtype=float)
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2) if grid is None else np.asarray(grid)
    rv = realized_value(y, clv, action)
    profits = np.array([rv[p >= t].sum() for t in grid])
    return float(grid[int(np.argmax(profits))]), grid, profits


def policy_summary(
    name: str,
    action_idx: np.ndarray,
    y,
    clv,
    actions: list[RetentionAction],
) -> dict[str, Any]:
    """Aggregate realised economics of a policy. ``action_idx[i] = -1`` means not targeted."""
    y = np.asarray(y)
    clv = np.asarray(clv, dtype=float)
    action_idx = np.asarray(action_idx)
    targeted = action_idx >= 0
    contact = incentive = gross = net = saves = 0.0
    for k, a in enumerate(actions):
        m = action_idx == k
        if not m.any():
            continue
        yk, vk = y[m], clv[m]
        contact += a.contact_cost * m.sum()
        incentive += a.incentive_cost * ((yk == 0).sum() + a.save_rate * (yk == 1).sum())
        gross += a.save_rate * vk[yk == 1].sum()
        saves += a.save_rate * (yk == 1).sum()
        net += realized_value(yk, vk, a).sum()
    cost = contact + incentive
    n_churners = int((y == 1).sum())
    tp = int((targeted & (y == 1)).sum())
    return {
        "policy": name,
        "customers_targeted": int(targeted.sum()),
        "pct_targeted": float(targeted.mean()),
        "churners_targeted": tp,
        "churner_capture_rate": tp / n_churners if n_churners else float("nan"),
        "precision": tp / targeted.sum() if targeted.any() else float("nan"),
        "expected_customers_saved": float(saves),
        "retained_value": float(gross),
        "campaign_cost": float(cost),
        "net_profit": float(net),
        "roi": float(net / cost) if cost > 0 else float("nan"),
    }


def profit_curve(scores, y, clv, action: RetentionAction, fractions: np.ndarray | None = None):
    """Realised profit when contacting the top-``f`` fraction ranked by ``scores``."""
    fractions = np.linspace(0, 1, 101) if fractions is None else np.asarray(fractions)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    rv = realized_value(np.asarray(y)[order], np.asarray(clv, dtype=float)[order], action)
    cum = np.concatenate([[0.0], np.cumsum(rv)])
    ks = np.array([math.ceil(f * len(rv)) for f in fractions])
    return fractions, cum[ks]


def compare_policies(
    p,
    y,
    clv,
    heuristic_scores,
    actions: list[RetentionAction],
    threshold_action: str,
    threshold: float,
) -> pd.DataFrame:
    """Benchmark targeting strategies on a labelled (held-out) set."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    n = len(y)
    names = [a.name for a in actions]
    ta = names.index(threshold_action)
    none = np.full(n, -1)

    rows = []
    rows.append(policy_summary("No campaign", none, y, clv, actions))
    rows.append(policy_summary("Blanket campaign (contact everyone)", np.full(n, ta), y, clv, actions))

    ml_mask = p >= threshold
    k = int(ml_mask.sum())
    idx = none.copy()
    idx[ml_mask] = ta
    rows.append(policy_summary(f"ML threshold policy (p >= {threshold:.2f})", idx, y, clv, actions))

    top_heur = np.argsort(-np.asarray(heuristic_scores, dtype=float), kind="mergesort")[:k]
    idx = none.copy()
    idx[top_heur] = ta
    rows.append(policy_summary("Rule-based heuristic (same volume)", idx, y, clv, actions))

    # Random targeting at the same volume: exact expectation instead of one noisy draw.
    rv = realized_value(y, clv, actions[ta])
    rand = policy_summary("Random targeting (same volume, expected)", none, y, clv, actions)
    rand.update(
        customers_targeted=k,
        pct_targeted=k / n,
        churners_targeted=int(round(k * y.mean())),
        churner_capture_rate=k / n,
        precision=float(y.mean()),
        net_profit=float(k * rv.mean()),
    )
    rand["campaign_cost"] = float(
        k
        * (
            actions[ta].contact_cost
            + actions[ta].incentive_cost * (1 - y.mean() * (1 - actions[ta].save_rate))
        )
    )
    rand["roi"] = rand["net_profit"] / rand["campaign_cost"] if rand["campaign_cost"] else float("nan")
    rand["expected_customers_saved"] = float(k * y.mean() * actions[ta].save_rate)
    rand["retained_value"] = float(rand["net_profit"] + rand["campaign_cost"])
    rows.append(rand)

    nba_idx, _ = next_best_action(p, clv, actions)
    rows.append(policy_summary("ML next-best-action (EV-optimal)", nba_idx, y, clv, actions))

    oracle = np.where(y == 1, ta, -1)
    rows.append(policy_summary("Oracle (perfect foresight, reference)", oracle, y, clv, actions))
    out = pd.DataFrame(rows).set_index("policy")
    out["net_profit_per_1k_customers"] = out["net_profit"] * 1000.0 / n
    return out


def sensitivity_grid(
    p_tune,
    y_tune,
    clv_tune,
    p_eval,
    y_eval,
    clv_eval,
    base_action: RetentionAction,
    save_rates,
    incentive_costs,
) -> pd.DataFrame:
    """Held-out profit of the threshold policy when the threshold is re-optimised (on tuning data)
    for each (save rate, incentive) scenario."""
    out = pd.DataFrame(
        index=pd.Index(save_rates, name="save_rate"), columns=list(incentive_costs), dtype=float
    )
    for s in save_rates:
        for d in incentive_costs:
            a = base_action.with_(save_rate=float(s), incentive_cost=float(d))
            t, _, _ = optimal_threshold(p_tune, y_tune, clv_tune, a)
            out.loc[s, d] = realized_value(y_eval, clv_eval, a)[np.asarray(p_eval) >= t].sum()
    out.columns.name = "incentive_cost"
    return out
