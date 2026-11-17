"""1) Run the agent on the real Cookie Cats test.  2) Evaluate it on scenarios built from the real players.
python run.py -> results/decision_log.json, results/evals.csv, results/summary.json  (~2 min)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.core import Policy, naive_review, review, run_daily

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "data" / "cookie_cats.csv")
arm = lambda x: {"d1": x.retention_1.values.astype(int), "d7": x.retention_7.values.astype(int), "rounds": x.sum_gamerounds.values.astype(float)}
A, B = arm(d[d.version == "gate_30"]), arm(d[d.version == "gate_40"])
pol = Policy(planned_n=len(d))

# ---------------------------------------------------------------- 1. the real test
final, day, log = run_daily(A, B, review, policy=pol)
full = review(A, B, pol)
nfinal, nday, nlog = run_daily(A, B, naive_review)
(ROOT / "results" / "decision_log.json").write_text(json.dumps(
    {"test": "Cookie Cats: first gate at level 30 (control) vs level 40 (treatment)", "players": len(d),
     "guarded_agent": {"decision": final.to_dict(), "decided_on_day": day, "daily_log": log, "full_sample_review": full.to_dict()},
     "naive_agent": {"decision": nfinal.to_dict(), "decided_on_day": nday, "daily_log": nlog}}, indent=1, default=float))
print("REAL TEST guarded:", final.action, "| day", day, "|", final.reason)
print("REAL TEST naive  :", nfinal.action, "| day", nday, "|", nfinal.reason)

# ---------------------------------------------------------------- 2. evals on scenarios with a known right answer
rng = np.random.default_rng(0)
pool = arm(d)                                   # all real players, labels ignored
N = len(d)


def sample_arm(n, d7_shift=0.0, d1_shift=0.0, rounds_mult=1.0, r=rng):
    """Draw n real players; optionally flip some outcomes to create a known true effect."""
    ix = r.integers(0, N, n)
    x = {k: v[ix].copy() for k, v in pool.items()}
    for key, shift in (("d7", d7_shift), ("d1", d1_shift)):
        if shift > 0:
            z = np.flatnonzero(x[key] == 0); x[key][r.choice(z, int(shift * n), replace=False)] = 1
        elif shift < 0:
            z = np.flatnonzero(x[key] == 1); x[key][r.choice(z, int(-shift * n), replace=False)] = 0
    x["rounds"] = x["rounds"] * rounds_mult
    return x


SCENARIOS = {  # name: (control size, treatment kwargs + size, acceptable final actions)
    "A/A test (no real difference)": (45000, dict(n=45000), {"STOP: NO EFFECT", "KEEP RUNNING"}),
    "True win: day-7 +1.5 pts": (45000, dict(n=45000, d7_shift=0.015), {"SHIP"}),
    "True loss: day-7 -1.5 pts": (45000, dict(n=45000, d7_shift=-0.015), {"ROLLBACK"}),
    "Win on day-7 but day-1 falls 3 pts": (45000, dict(n=45000, d7_shift=0.015, d1_shift=-0.03), {"ROLLBACK"}),
    "Win on day-7 but rounds fall 15%": (45000, dict(n=45000, d7_shift=0.015, rounds_mult=0.85), {"ROLLBACK"}),
    "Broken split (55/45), no difference": (40500, dict(n=49500), {"INVALID"}),
}
R = 400
rows = []
for name, (n_c, kw, ok) in SCENARIOS.items():
    res = {"guarded": [], "naive": []}
    days = {"guarded": [], "naive": []}
    for i in range(R):
        r = np.random.default_rng(1000 + i)
        c, t = sample_arm(n_c, r=r), sample_arm(r=r, **kw)
        p = Policy(planned_n=n_c + kw["n"])
        g, gd, _ = run_daily(c, t, review, seed=i, policy=p)
        nv, nd, _ = run_daily(c, t, naive_review, seed=i)
        res["guarded"].append(g.action in ok); res["naive"].append(nv.action in ok)
        days["guarded"].append(gd); days["naive"].append(nd)
    rows.append({"scenario": name, "right_answer": " / ".join(sorted(ok - {"KEEP RUNNING"})),
                 "guarded_agent_correct_pct": 100 * np.mean(res["guarded"]), "naive_agent_correct_pct": 100 * np.mean(res["naive"]),
                 "guarded_avg_days": np.mean(days["guarded"]), "naive_avg_days": np.mean(days["naive"])})
    print(rows[-1], flush=True)
ev = pd.DataFrame(rows).round(1)
ev.to_csv(ROOT / "results" / "evals.csv", index=False)
S = {"players": len(d), "real_test": {"guarded": final.action, "guarded_day": day, "naive": nfinal.action, "naive_day": nday,
     "d7_control": full.checks["d7"]["control"], "d7_treatment": full.checks["d7"]["treatment"], "d7_diff_pts": 100 * full.checks["d7"]["diff"],
     "d7_p": full.checks["d7"]["p"], "d7_ci_pts": [100 * x for x in full.checks["d7"]["ci"]], "d1_diff_pts": 100 * full.checks["d1"]["diff"],
     "srm_p": full.checks["srm_p"], "rounds_rel_change": full.checks["rounds_rel_change"]},
     "eval_runs_per_scenario": R, "guarded_overall_correct_pct": float(ev.guarded_agent_correct_pct.mean()),
     "naive_overall_correct_pct": float(ev.naive_agent_correct_pct.mean())}
(ROOT / "results" / "summary.json").write_text(json.dumps(S, indent=1, default=float))
print(json.dumps(S, indent=1, default=float))
