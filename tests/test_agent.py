import numpy as np

from agent.core import Policy, naive_review, obf_threshold, review


def arm(n, d7, d1=0.45, rounds=50, seed=0):
    r = np.random.default_rng(seed)
    return {"d7": (r.random(n) < d7).astype(int), "d1": (r.random(n) < d1).astype(int), "rounds": r.exponential(rounds, n)}


P = Policy(planned_n=80_000)


def test_ships_a_clear_win_and_rolls_back_a_clear_loss():
    assert review(arm(40000, .19), arm(40000, .22, seed=1), P).action == "SHIP"
    assert review(arm(40000, .19), arm(40000, .16, seed=1), P).action == "ROLLBACK"


def test_broken_split_is_invalid_even_if_it_looks_like_a_win():
    assert review(arm(30000, .19), arm(50000, .25, seed=1), P).action == "INVALID"


def test_guardrail_blocks_a_day7_win_that_hurts_day1():
    d = review(arm(40000, .19), arm(40000, .22, d1=.40, seed=1), P)
    assert d.action == "ROLLBACK" and "day-1" in d.reason


def test_early_looks_need_stronger_evidence():
    assert obf_threshold(.05, .1) < obf_threshold(.05, .5) < obf_threshold(.05, 1.0) < .05
    c, t = arm(4000, .19), arm(4000, .215, seed=1)                       # 10% of the planned sample
    assert naive_review(c, t).action == "SHIP" and review(c, t, P).action == "KEEP RUNNING"


def test_real_cookie_cats_decision():
    import pandas as pd
    from pathlib import Path
    d = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "cookie_cats.csv")
    f = lambda x: {"d1": x.retention_1.values.astype(int), "d7": x.retention_7.values.astype(int), "rounds": x.sum_gamerounds.values.astype(float)}
    out = review(f(d[d.version == "gate_30"]), f(d[d.version == "gate_40"]), Policy(planned_n=len(d)))
    assert out.action == "ROLLBACK" and out.checks["warnings"]           # gate 40 hurts day-7; split flagged as uneven
