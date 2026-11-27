# RL train/infer placement

A digital twin of RL training on a GPU fleet, built to understand the trade-off between sharing the same cards in time (rollout inference and training alternate) and splitting the fleet in space. The aim is the model, not a verdict on which layout is fastest: it must explain where the card-time goes and why, and it must be checkable against the output of real runs. The full aim and what it requires are in `STATUS.md`.

The simulator takes cluster, model and workload and returns, for any deploy policy, the time to target and the policy lag.

Moved here from `~/casual/rl-train-infer-placement` on 2026-11-27. The original discussion record stays there.

## Layout

| Path | What |
|---|---|
| `rlsim.py` | The simulator: inputs, engine, deploy policies (`Phased`, `Split`, `Dynamic`), calibration helpers |
| `scenarios.py` | Scenarios A to M and the self-tests |
| `factors.json` | Efficiency and overhead factors. Currently simulator defaults, not measurements |
| `results.txt` | Output of the last full scenario run |
| `testbed/` | The real counterpart, built here and run on the cards by the user. `probe.py` describes the machine; `smoke.py` is first contact with the inference engine; `results/` keeps what the user brings back |
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
- No real RL system exists to check the twin against. It is built in this repo under `testbed/`.
- The user has 8 x H200 for verification. The driver reports them as "NVIDIA L20X"; the user first called them H800. I write the code here; the user pulls it there, runs it, and pastes the printed output back. Each testbed script must therefore run unattended, fail loudly, and print everything that matters, since the output file may not come back.
- That machine has SGLang and torch installed and no vLLM. Use what is installed; do not install an engine that would replace its torch. It reaches ModelScope and hf-mirror, not huggingface.co.
- Code under `testbed/` that touches a card cannot be tested here. Say which paths were tested and which were not.
- Keep every factor overridable through `factors.json`; do not hard-code measured values in `rlsim.py`.
- The H800 entry in `CARDS` uses H100 compute numbers as a placeholder. Its specifications are unverified.
- Default factors come from two published H100 numbers read in search summaries, not verified at source. Treat absolute hours as provisional.

## Working rules

- Run `selftest` after any change to `rlsim.py`. Add a closed-form or invariant check with each new mechanism.
- When a result changes an earlier claim, record the correction in `docs/design-and-findings.md` rather than silently replacing it.
- Report simulator output as predictions, and say which factors they rest on.
