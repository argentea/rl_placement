# Status

Last updated: 2026-11-27

## Aim

Set by the user on 2026-11-27. It replaces the earlier aim of picking the fastest layout.

Build a digital twin of an RL system that trains an LLM policy on a GPU fleet. It has two jobs:

1. **Explain.** For any cluster, model, workload and deploy policy, show where every card-second goes and why, so the trade-off between sharing cards in time and splitting them in space can be read off the mechanisms and not only off a final number.
2. **Be checkable.** Take the output of a real run and say, quantity by quantity, whether the real system behaved as the twin predicts. Where it did not, point at the mechanism that is off.

Which layout is fastest is a question put to the twin. It is not the aim.

No real system exists to check the twin against, so there is a third job:

3. **Build the real counterpart.** A minimal RL training loop under `testbed/` that does real generation and real gradient steps on 8 x H200, under the same deploy policies as the twin, and writes the same log. I write the code here; the user runs it on the cards and brings the output back.

What ties the two together: one deploy policy code driving two backends (simulated engine, real engine), one workload sampler with a shared seed, one log schema. This is the intended design, not yet built.

The real system is established in stages:

| Stage | What runs on the cards | What it checks in the twin | State |
|---|---|---|---|
| 0. Machine | `testbed/probe.py` | Card memory and count; whether the card numbers in `CARDS` are plausible | Done, two runs |
| 1. Components | One serving replica under load, one training step, start and stop, a weight push (E1 to E6) | Each law on its own | `testbed/smoke.py` is first contact with the engine. The user's first run printed nothing for a long time; cause unknown, the server log is needed. The script now reports progress |
| 2. Closed loop | An orchestrator running the deploy policies end to end, with a synthetic environment drawn from the twin's workload sampler (E8) | How the laws compose: time per update, lag, where the card-time goes | Not started |
| 3. Real task | A task with a reward (E9) | The value of lag. Only if lag turns out to matter | Not started |

## The machine, measured 2026-11-27

Raw output in `testbed/results/2026-11-27-probe.txt` and `2026-11-27-probe-v2.txt`. The user first said 8 x H800; the cards are 8 x H200, which the driver reports as "NVIDIA L20X".

| Measured | Value | What the twin assumes | Reading |
|---|---|---|---|
| Card memory | 140 GiB each | `mem_gb` 141 | Consistent. The exact byte count is in the user's `probe.json`, not yet here |
| Dense 16-bit matrix product | 673 to 711 TFLOP/s | Nominal peak 989, scaled by `infer_eff` 0.15 and `train_eff` 0.55 | A bare product reaches 0.68 to 0.72 of nominal, so no efficiency factor can sit much above 0.7. The default `train_eff` is 79% of that ceiling and will probably come in lower |
| Copy within a card | 4050 to 4280 GB/s, read plus written | 4800 x `membw_eff` 0.85 = 4080 GB/s to read the weights | Same range. Not the same operation, so a sanity check only |
| Host to card | 55.5 GB/s | `load_gbs` 8 | The link is not the limit. Real loads are bounded by disk and deserialisation; to be measured |
| Card to card | 395 GB/s | `intra_gbs` 8 | Same. Sync cost will be set by the software path, not by the link |

Other facts:

- A container, running as root. 800 GiB RAM, 404 GiB of `/dev/shm`, 795 GiB of local disk that probably does not survive a restart. The repo sits on NFS, which does.
- CPU is not a limit: the quota is about 102 cores, and 32 busy processes ran 32 times the work of one. But `os.cpu_count()` returns 2 there, so anything that sizes its threads or workers from it will use two. The testbed must set thread and worker counts itself.
- Installed: torch 2.13.0 (CUDA 13.0), SGLang 0.5.19 development build, transformers, peft, accelerate, deepspeed. No vLLM.
- ModelScope and hf-mirror answer; huggingface.co does not. No models on disk.

## The inference engine: SGLang

SGLang is what the machine has, built against its torch. Installing vLLM would pull in its own torch, so the testbed uses SGLang.

From the SGLang documentation (the "SGLang for RL" and native API pages, read 2026-11-27). The installed build is a development version and may differ; `testbed/smoke.py` checks each call.

| Twin primitive | SGLang call | Note |
|---|---|---|
| `sync`, stale KV cache | `pause_generation` mode `in_place`, then `update_weights_from_*` with `flush_cache` false, then `continue_generation` | Running requests keep their cache |
| `sync`, KV rebuild | The same with mode `retract` and `flush_cache` true | Running requests go back to the queue and their cache is recomputed |
| Not in the twin | The same with mode `abort` | Running requests are returned to the caller at once |
| `set_roles`, resident | `release_memory_occupation` and `resume_memory_occupation`; the server needs `--enable-memory-saver` | Needs an idle server. Releasing the KV cache flushes it |
| `set_roles`, restart | Stop and start the server | |
| Lag per token | `weight_versions` in each response: spans of output tokens with the version that produced them | The engine reports the quantity the twin computes |
| Weight transport | From disk, from tensors in memory, or over a process group between cards | Three paths with different costs. The twin has one rate, `intra_gbs` |

Not documented, and so for the smoke test to find out: what a weight update does to a running request when nothing pauses it first.

Checked against the installed build's source by probe version 2:

- Every call above is present, with request fields that match the documentation. The session calls `begin_weight_update` and `end_weight_update` are absent.
- There is a fourth weight path the documentation did not show, `UpdateWeightsFromIPCReqInput`, which takes handles and no data. Adapters can also be loaded from tensors.
- The launch flags could not be read from source, because the server's main flags are not written there as literals. `smoke.py` reads them from the server's `--help`.

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
| The real controller expressible as a deploy policy | Yes, four primitives. Draining before a sync, syncing less often and aborting running requests are missing |

## Done

- Simulator with deploy policies over four primitives (roles, admission, train, sync). Self-tests pass.
- Scenarios A to M run; results in `results.txt`, findings in `docs/design-and-findings.md`.
- Factor file, fitting helpers (`fit_inference`, `calibrate`, `load_factors`), experiment plan.
- `testbed/probe.py`, run once on the cards.

## What the simulator predicts so far, provisional

With default factors, at small batch relative to fleet concurrency:

- A split that keeps in-flight KV caches across weight syncs is 1.5 to 2 times faster than any time-shared variant, at 0.4 to 1.5 updates of lag.
- If caches must be rebuilt after each sync, the split roughly ties with time-sharing plus a straggler cut.
- Plain time-sharing is slowest. At large batch all layouts are within 30%.

This rests on unmeasured factors and an unpriced lag.

In scenario A, plain time-sharing and the stale-cache split spend about the same card-hours generating (2.2) and training (1.9 to 2.0). The 3.3 times gap in wall time sits in idle card-time (66% against 15%) and switching (10% against 3%). The simulator does not yet say what the idle time is made of.

## Next, proposed order, not yet confirmed by the user

On the cards:

1. **User runs `python3 testbed/smoke.py`** and pastes the output, and the server log if a step stalls. It starts SGLang on one card with a 0.5B model and tries generation, weight sync under each way of treating a running request, and memory release. Tested here against stand-ins only: no card, no engine on this machine.
2. **Choose the models.** Proposal: 8B as the main model, where every split of the eight cards is feasible; 32B as the scaling check, with factors fitted at 8B and the twin asked to predict 32B; 72B only with adapters. The 0.5B model is for checking scripts.
3. **Stage 1 scripts**, then stage 2. Each script runs unattended, prints what matters, and writes one file.

Here, in parallel, needing no cards:

4. **Itemised account and timeline in the engine.** Idle, overhead and waste by cause; one record per update. Invariant: the items sum to total card-time.
5. **Attribution and reduced model.** Explain the gap between any two deploys by mechanism; write the few-term model and test it against the event simulation.
6. **Log schema and comparison report.** The twin writes the schema; the report reads it. Test by feeding the twin its own output, then output from a twin with one factor changed, which the report must detect and localise.

## Open decisions for the user

- The models for measurements (see proposal above), and whether training is a full fine-tune or adapters. The user asked for something larger and more capable than 0.5B on 2026-11-27.
- The real workload: prompt, turn, generation and latency distributions, batch size. Scenarios use invented agentic and reasoning workloads.
- Whether training on episodes generated partly under older weights (stale KV cache) is acceptable to the algorithm.

## Known gaps in the model

- Training memory. The twin spreads the training state over the training cards (`train_state_bytes / n`), which holds only for sharded training. Plain data-parallel training keeps a full copy on every card: 128 GB for an 8B full fine-tune at 16 bytes per parameter, plus activations, which is more than the usable part of a 141 GB card. The twin needs both laws; the testbed needs sharded training, a leaner optimizer state, adapters or a smaller model. From reasoning, not from a run.
- The engine's KV cache is a pool reserved at start, sized as a fraction of the card. The twin computes slots from free memory per episode. To be compared once `server_info` reports the pool size.
- Draining a replica before syncing it, as a third option besides stale cache and rebuild.
- Aborting running requests at a sync, which SGLang offers.
- Syncing less often than every update.
- Prefix sharing between samples of one prompt.
- A judge model's own memory footprint.
- Trainer-side length bucketing.
- The straggler cut's length bias is reported but not priced.
