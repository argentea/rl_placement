# Status

Last updated: 2026-11-27

## Aim

Set by the user on 2026-11-27. It replaces the earlier aim of picking the fastest layout.

Build a digital twin of an RL system that trains an LLM policy on a GPU fleet. It has two jobs:

1. **Explain.** For any cluster, model, workload and deploy policy, show where every card-second goes and why, so the trade-off between sharing cards in time and splitting them in space can be read off the mechanisms and not only off a final number.
2. **Be checkable.** Take the output of a real run and say, quantity by quantity, whether the real system behaved as the twin predicts. Where it did not, point at the mechanism that is off.

Which layout is fastest is a question put to the twin. It is not the aim.

No real system exists to check the twin against, so there is a third job:

3. **Build the real counterpart.** A minimal RL training loop under `testbed/` that does real generation and real gradient steps on 8 x H800, under the same deploy policies as the twin, and writes the same log. I write the code here; the user runs it on the cards and brings the output back.

What ties the two together: one deploy policy code driving two backends (simulated engine, real engine), one workload sampler with a shared seed, one log schema. This is the intended design, not yet built.

The real system is established in stages:

| Stage | What runs on the cards | What it checks in the twin |
|---|---|---|
| 0. Machine | `testbed/probe.py` | Card memory and count; whether the H800 placeholder numbers are plausible |
| 1. Components | One serving replica under load, one training step, start and stop, a weight push (E1 to E6) | Each law on its own |
| 2. Closed loop | An orchestrator running the deploy policies end to end, with a synthetic environment drawn from the twin's workload sampler (E8) | How the laws compose: time per update, lag, where the card-time goes |
| 3. Real task | A task with a reward (E9) | The value of lag. Only if lag turns out to matter |

## What the aim requires, and where the simulator stands

| Requirement | Now |
|---|---|
| Itemised account of card-time: idle by cause (environment wait, waiting on the slowest episodes, serving held by the lag bound, trainer waiting for data), overhead by cause, wasted work by cause | `Result.shares` has one idle number |
| The difference between two deploys attributed to mechanisms, with the items summing to the total | Missing |
| Timeline of a run, per update and per card | Aggregates only |
| A reduced model in a few terms that reproduces the event simulation | Only the three limiting cases in `selftest` |
| One log schema written by both the twin and a real run | Missing |
| Replay of a real episode trace as the workload | Marginal distributions only (`Dist.values`) |
| Comparison report: predicted against observed at component, phase and run level, with a tolerance | Missing. `fit_inference` uses three points of the sweep and reports no misfit |
| Any law replaceable by a measured curve | Step-time and train-time laws are fixed formulas with scalar factors |
| The real controller expressible as a deploy policy | Yes, four primitives. Draining before a sync and syncing less often are missing |

## Done

- Simulator with deploy policies over four primitives (roles, admission, train, sync). Self-tests pass.
- Scenarios A to M run; results in `results.txt`, findings in `docs/design-and-findings.md`.
- Factor file, fitting helpers (`fit_inference`, `calibrate`, `load_factors`), experiment plan.

## What the simulator predicts so far, provisional

With default factors, at small batch relative to fleet concurrency:

- A split that keeps in-flight KV caches across weight syncs is 1.5 to 2 times faster than any time-shared variant, at 0.4 to 1.5 updates of lag.
- If caches must be rebuilt after each sync, the split roughly ties with time-sharing plus a straggler cut.
- Plain time-sharing is slowest. At large batch all layouts are within 30%.

This rests on unmeasured factors and an unpriced lag.

In scenario A, plain time-sharing and the stale-cache split spend about the same card-hours generating (2.2) and training (1.9 to 2.0). The 3.3 times gap in wall time sits in idle card-time (66% against 15%) and switching (10% against 3%). The simulator does not yet say what the idle time is made of.

## Next, proposed order, not yet confirmed by the user

On the cards:

1. **Stage 0.** User runs `python3 testbed/probe.py --bench` on the H800 machine and returns `testbed/out/probe.json`. The probe is written; its facts path is tested here, its `--bench` path is not (no torch and no card on this machine).
2. **Choose the stack and model** from what the probe finds: inference engine, trainer, model size, how models are downloaded.
3. **Stage 1 scripts**, then stage 2. Each script runs unattended and writes one file in the shared schema.

Here, in parallel, needing no cards:

4. **Itemised account and timeline in the engine.** Idle, overhead and waste by cause; one record per update. Invariant: the items sum to total card-time.
5. **Attribution and reduced model.** Explain the gap between any two deploys by mechanism; write the few-term model and test it against the event simulation.
6. **Log schema and comparison report.** The twin writes the schema; the report reads it. Test by feeding the twin its own output, then output from a twin with one factor changed, which the report must detect and localise.

## Open decisions for the user

- Which model, inference engine and trainer the testbed should use. Factors do not transfer across stacks. I will propose them once the probe is back.
- The real workload: prompt, turn, generation and latency distributions, batch size. Scenarios use invented agentic and reasoning workloads.
- Whether training on episodes generated partly under older weights (stale KV cache) is acceptable to the algorithm.

## Known gaps in the model

- Training memory. The twin spreads the training state over the training cards (`train_state_bytes / n`), which holds only for sharded training. Plain data-parallel training keeps a full copy on every card: 128 GB for an 8B full fine-tune at 16 bytes per parameter, more than an 80 GB card. The twin needs both laws; the testbed needs sharded training, adapters or a smaller model. From reasoning, not from a run.
- Draining a replica before syncing it, as a third option besides stale cache and rebuild.
- Syncing less often than every update.
- Prefix sharing between samples of one prompt.
- A judge model's own memory footprint.
- Trainer-side length bucketing.
- The straggler cut's length bias is reported but not priced.
