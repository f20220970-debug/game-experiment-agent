# Experiment-Review Agent for Game LiveOps

[![tests](https://github.com/f20220970-debug/game-experiment-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/game-experiment-agent/actions/workflows/tests.yml)

**[Live demo](https://f20220970-debug.github.io/game-experiment-agent/)**: enter any A/B test and get the agent's decision.

## In 30 seconds
- An agent that reviews game A/B tests every day and decides **ship, roll back, keep running or invalid**, with guardrails and a full decision log.
- Evaluated on 2,400 replays built from **90,189 real players**: it makes the right call **97.9%** of the time, against **46.3%** for a naive agent that acts the first time p < 0.05.
- On the real test (Cookie Cats, gate at level 30 vs 40) it **rolls back** the change: day-7 retention fell 0.8 points.

Built for **Chaos Engine Labs' PS-II project "Agentic systems for live operations (LiveOps)"**, whose expected
outcome is agents "with guardrails, observability, evals and business impact measured through experiments".
My earlier Chaos Engine projects ([LiveOps agent](https://github.com/f20220970-debug/chaos-liveops-agent),
[UA budget agent](https://github.com/f20220970-debug/game-ua-budget-agent),
[ad floor agent](https://github.com/f20220970-debug/ad-floor-agent)) run on simulations. This one runs on
**real player data**.

> **Data:** the public Cookie Cats A/B test (Tactile Entertainment's mobile puzzle game): 90,189 players, the
> first gate placed at level 30 or level 40, with day-1 retention, day-7 retention and rounds played.

## What the agent does (`agent/core.py`)
| Step | Check | Why |
| --- | --- | --- |
| 1. Validity | Is the split between groups what it should be? (sample-ratio check) | A broken split makes every result untrustworthy |
| 2. Primary metric | Day-7 retention, with a threshold that is very strict early and relaxes as data arrives (O'Brien-Fleming boundary) | Looking every day without this produces false wins |
| 3. Guardrails | Day-1 retention and rounds played must not drop beyond a set margin | A retention "win" that kills engagement isn't a win |
| 4. Decision | Ship, roll back, keep running, stop with no effect, or invalid; every number saved to a log | A person can audit any decision afterwards |

## Result on the real test (`results/decision_log.json`)
| | Day-7 retention | Decision | Decided on |
| --- | --- | --- | --- |
| Gate at level 30 (control) | 19.02% | | |
| Gate at level 40 | 18.20% (−0.82 pts, 95% CI −1.33 to −0.31) | | |
| **Guarded agent** | | **Roll back**, and it warns that the split is uneven (44,700 vs 45,489 players) | day 7 of 14 |
| Naive agent | | Roll back at p = 0.039 | day 4 |

Both got this one right. The naive agent was faster, and the evals show what that speed costs.

## Evals: does it make the right call when the answer is known?
Each scenario resamples the real players and, where needed, flips outcomes to create a known true effect.
The test is replayed over 14 days with one look per day. 400 replays per scenario.

| Scenario | Right answer | Guarded agent | Naive agent |
| --- | --- | --- | --- |
| No real difference (A/A) | stop, no effect | **93.5%** | 77.8% |
| True win: day-7 +1.5 pts | ship | 99.8% | 100% |
| True loss: day-7 −1.5 pts | roll back | 100% | 100% |
| Day-7 win, but day-1 retention falls 3 pts | roll back | **94.5%** | 0% |
| Day-7 win, but rounds played fall 15% | roll back | **99.5%** | 0% |
| Broken split (55/45), no difference | invalid | **100%** | 0% |
| **Average** | | **97.9%** | 46.3% |

- **The cost of safety is time:** the guarded agent takes 5.9 days to ship a true win, the naive one 2.5.
- **False decisions with no real effect: 6.5% vs 22%.** The target is 5%; the remainder comes from the
  guardrail and split checks adding their own small false-alarm rates.

## A mistake the evals caught
My first sequential threshold used 1.96 (the value for a single look) as its base. The A/A eval measured
**8.5% false decisions against a 5% target**. The fix was the proper boundary constant for 14 daily looks
(2.10), which brought it to 6.5%. Without the eval I would have shipped an agent that was quietly too eager.

## Run it
```bash
pip install -r requirements.txt
python run.py      # real test + 2,400 eval replays (~2 min) -> results/
pytest -q          # 5 tests, including the decision on the real data
```

## Limitations and next steps
- One real test with three metrics; revenue isn't in the data, so there is no revenue guardrail yet.
- Effects in the evals are created by flipping outcomes at random, so they don't vary by player type.
- Next: several tests running at once (false-discovery control), segment-level decisions (new vs returning
  players), and an LLM-written summary whose numbers are checked against the log before it's sent.

Built by Pavitra Seth (BITS Pilani) as preparation for Chaos Engine Labs' PS-II. Not affiliated with Chaos
Engine Labs or Tactile Entertainment.
