# RL train/infer placement

Decide how to place RL training and rollout inference on a GPU fleet: share the same cards in time, or split the fleet in space. The tool is a simulator that takes cluster, model and workload and returns, for any deploy policy, the time to target and the policy lag.

Moved here from `~/casual/rl-train-infer-placement` on 2026-11-27. The original discussion record stays there.

## Layout

| Path | What |
|---|---|
| `rlsim.py` | The simulator: inputs, engine, deploy policies (`Phased`, `Split`, `Dynamic`), calibration helpers |
| `scenarios.py` | Scenarios A to M and the self-tests |
| `factors.json` | Efficiency and overhead factors. Currently simulator defaults, not measurements |
| `results.txt` | Output of the last full scenario run |
| `docs/design-and-findings.md` | Model design, results, findings, corrections, limits |
| `docs/experiment-plan.md` | Experiments E1 to E9 to measure the factors on real cards |
| `STATUS.md` | Where the job stands and what is next. Keep it current |

## Commands

```
python3 scenarios.py selftest     # closed-form checks and invariants, a few seconds
python3 scenarios.py              # all scenarios, about 1.5 minutes on 20 cores
python3 scenarios.py A L          # selected scenarios
```

Needs Python 3 with numpy. `run_many` and `evaluate` use a process pool, so call them from a file, not from stdin.

## Constraints

- This machine has no GPU. Nothing here can be measured locally.
- The user can run experiments on H800 or H200 cards elsewhere, 8 or 16 at a time, and bring numbers back.
- Keep every factor overridable through `factors.json`; do not hard-code measured values in `rlsim.py`.
- The H800 entry in `CARDS` uses H100 compute numbers as a placeholder. Its specifications are unverified.
- Default factors come from two published H100 numbers read in search summaries, not verified at source. Treat absolute hours as provisional.

## Working rules

- Run `selftest` after any change to `rlsim.py`. Add a closed-form or invariant check with each new mechanism.
- When a result changes an earlier claim, record the correction in `docs/design-and-findings.md` rather than silently replacing it.
- Report simulator output as predictions, and say which factors they rest on.
