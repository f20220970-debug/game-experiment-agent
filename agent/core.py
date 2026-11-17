"""Experiment-review agent for game LiveOps A/B tests.

Given player-level results of an A/B test, the agent runs a fixed chain of checks and returns one decision
(SHIP, ROLLBACK, KEEP RUNNING or INVALID) with every number it used, as a structured log.

  1. Validity      sample-ratio mismatch (chi-square), minimum sample
  2. Primary       day-7 retention difference, with a sequential (alpha-spending) threshold so that checking
                   the test every day does not inflate false positives
  3. Guardrails    day-1 retention and rounds played must not drop by more than a set margin
  4. Decision      ship only if primary is a significant win and no guardrail is breached; roll back on a
                   significant loss or breach; otherwise keep running until the planned sample, then stop

A 'naive' agent (ship or roll back the first time p < 0.05, no validity check, no guardrails) is included as
the baseline every eval compares against.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats


@dataclass
class Policy:
    alpha: float = 0.05
    planned_n: int = 90_000            # total players planned for the test
    min_n_per_arm: int = 2_000
    srm_p: float = 0.001               # below this, the split itself is broken
    guard_d1_drop: float = 0.01        # tolerated absolute drop in day-1 retention
    guard_rounds_drop: float = 0.05    # tolerated relative drop in median-robust rounds (winsorised mean)
    expected_split: float = 0.5


@dataclass
class Decision:
    action: str
    reason: str
    checks: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def two_prop(x1, n1, x2, n2):
    """Difference in proportions (treatment - control), z-test p-value and 95% CI."""
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se0 = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se0 if se0 > 0 else 0.0
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {"control": p1, "treatment": p2, "diff": p2 - p1, "z": z, "p": 2 * stats.norm.sf(abs(z)),
            "ci": (p2 - p1 - 1.96 * se, p2 - p1 + 1.96 * se)}


OBF_FINAL_Z = 2.10     # O'Brien-Fleming boundary constant for ~14 daily looks at overall alpha 0.05


def obf_threshold(alpha, info, final_z=OBF_FINAL_Z):
    """O'Brien-Fleming group-sequential boundary: the p-value needed at information fraction `info` (0..1].
    Very strict early, close to alpha at the end, so daily looks stay honest.
    The first version used 1.96 here (the single-look value); the A/A eval measured 8.5% false decisions
    against a 5% target, which is how the missing multiple-look correction was found."""
    info = float(np.clip(info, 1e-6, 1.0))
    z = final_z / np.sqrt(info) if alpha == 0.05 else stats.norm.isf(alpha / 2) * (final_z / 1.96) / np.sqrt(info)
    return 2 * stats.norm.sf(z)


def winsor_mean(x, q=0.99):
    return float(np.minimum(x, np.quantile(x, q)).mean())


def review(control, treatment, policy: Policy = Policy()) -> Decision:
    """control / treatment: dicts of arrays 'd1', 'd7' (0/1) and 'rounds' for the players seen so far."""
    n1, n2 = len(control["d7"]), len(treatment["d7"])
    n = n1 + n2
    c = {"n_control": n1, "n_treatment": n2}
    exp = [n * (1 - policy.expected_split), n * policy.expected_split]
    srm_p = float(stats.chisquare([n1, n2], exp).pvalue) if n else 1.0
    c["srm_p"] = srm_p
    if srm_p < policy.srm_p:
        return Decision("INVALID", f"sample-ratio mismatch (p={srm_p:.2g}): the split is broken, fix assignment before reading results", c)
    c["warnings"] = ["split is uneven (p=%.3g): check assignment" % srm_p] if srm_p < 0.01 else []
    if min(n1, n2) < policy.min_n_per_arm:
        return Decision("KEEP RUNNING", "not enough players per arm yet", c)
    info = min(n / policy.planned_n, 1.0)
    thr = obf_threshold(policy.alpha, info)
    d7 = two_prop(control["d7"].sum(), n1, treatment["d7"].sum(), n2)
    d1 = two_prop(control["d1"].sum(), n1, treatment["d1"].sum(), n2)
    r1, r2 = winsor_mean(control["rounds"]), winsor_mean(treatment["rounds"])
    c.update({"info_fraction": info, "p_threshold": thr, "d7": d7, "d1": d1, "rounds_control": r1, "rounds_treatment": r2,
              "rounds_rel_change": r2 / r1 - 1 if r1 else 0.0})
    # a guardrail is breached only when the whole 95% interval sits below the tolerated drop
    breach = []
    if d1["ci"][1] < -policy.guard_d1_drop:
        breach.append("day-1 retention")
    if c["rounds_rel_change"] < -policy.guard_rounds_drop and stats.mannwhitneyu(control["rounds"], treatment["rounds"]).pvalue < thr:
        breach.append("rounds played")
    c["guardrail_breaches"] = breach
    sig = d7["p"] < thr
    if breach:
        return Decision("ROLLBACK", "guardrail breached: " + ", ".join(breach), c)
    if sig and d7["diff"] > 0:
        return Decision("SHIP", f"day-7 retention up {100*d7['diff']:.2f} pts (p={d7['p']:.3g} < {thr:.3g}), guardrails fine", c)
    if sig and d7["diff"] < 0:
        return Decision("ROLLBACK", f"day-7 retention down {abs(100*d7['diff']):.2f} pts (p={d7['p']:.3g} < {thr:.3g})", c)
    if info >= 1.0:
        return Decision("STOP: NO EFFECT", f"planned sample reached, day-7 difference {100*d7['diff']:+.2f} pts not significant: keep control", c)
    return Decision("KEEP RUNNING", f"no decision yet at {100*info:.0f}% of planned sample (p={d7['p']:.3g}, need < {thr:.3g})", c)


def naive_review(control, treatment, alpha=0.05) -> Decision:
    """Baseline: looks only at day-7 retention, ships or rolls back the first time p < alpha."""
    n1, n2 = len(control["d7"]), len(treatment["d7"])
    if min(n1, n2) < 100:
        return Decision("KEEP RUNNING", "too few players")
    d7 = two_prop(control["d7"].sum(), n1, treatment["d7"].sum(), n2)
    if d7["p"] < alpha:
        return Decision("SHIP" if d7["diff"] > 0 else "ROLLBACK", f"p={d7['p']:.3g}", {"d7": d7})
    return Decision("KEEP RUNNING", f"p={d7['p']:.3g}", {"d7": d7})


def run_daily(control, treatment, reviewer, days=14, seed=0, **kw):
    """Replay a test as it would arrive: players trickle in over `days`; the reviewer looks once a day and
    the test ends at the first final decision. Returns (final decision, day, log of every look)."""
    rng = np.random.default_rng(seed)
    o1, o2 = rng.permutation(len(control["d7"])), rng.permutation(len(treatment["d7"]))
    log = []
    for day in range(1, days + 1):
        k1, k2 = int(len(o1) * day / days), int(len(o2) * day / days)
        c = {k: v[o1[:k1]] for k, v in control.items()}
        t = {k: v[o2[:k2]] for k, v in treatment.items()}
        d = reviewer(c, t, **kw)
        log.append({"day": day, "action": d.action, "reason": d.reason})
        if d.action != "KEEP RUNNING":
            return d, day, log
    return d, days, log
