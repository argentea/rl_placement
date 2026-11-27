# RL: should training and inference share cards or get separate ones?

Dates: 2026-11-25 (question, first model), 2026-11-27 (scope audit, rewrite, experiment plan)

## Question

In an RL system with an LLM policy (rollouts are generation, updates are gradient steps), the two workloads can be placed two ways:

- **Time-shared**: the same cards alternate. Everyone generates a batch, then everyone trains on it.
- **Space-split**: some cards only run inference, the others only train, at the same time.

What is the key tradeoff, and how do we answer it with a model instead of an outline?

## Shape of the model

The user's framing: a function from the setup to a function over deployments.

```
sim = make_simulator(cluster, model, workload)
sim(deploy)  ->  Result(time to target, policy lag, value per update, length bias, waste, card-time shares)
```

- `cluster`: one or more card types with counts, node size, transfer speeds, switch costs. Efficiency factors live on each card type and load from `factors.json`.
- `model`: parameters, layers, width, dense or MoE, full fine-tune or adapters.
- `workload`: prompt length, turns, tokens per turn, environment latency and tokens (distributions, optionally correlated and drifting over the run), environment worker limit, grader cost, batch size, target updates, and the price of lag.
- `deploy`: a policy written against four engine primitives.

Files: `rlsim.py` (model), `scenarios.py` (scenarios and self-tests), `results.txt` (full output), `factors.json` (measured factors, currently defaults), `experiment-plan.md` (how to measure them).

## Deploy as a policy over primitives

| Primitive | What it does |
|---|---|
| `set_roles(n_train, carry)` | Chooses which cards train and which serve; pays switch costs; suspends or discards episodes on cards that change role |
| `want(eng)` / `admit()` | Admission control: batch size, oversampling, lag bound |
| `train()` | One update on the oldest finished batch |
| `sync(rolling, recompute)` | Pushes new weights to serving replicas; optionally rebuilds in-flight KV caches |

Three short policies cover the layouts:

- `Phased(resident, oversample, carry)`: time-sharing; with `oversample` a straggler cut; with `carry` partial rollouts.
- `Split(n_train, max_outstanding, rolling, recompute)`: fixed split with a lag bound.
- `Dynamic(...)`: a split whose boundary moves when the trainer starves or serving is held back.

## What is modelled

- **Rollout time**, event by event: a decode step takes `max(weights / memory bandwidth, load x FLOPs per token / compute) + overhead`; concurrency is capped by KV-cache memory; prefill counts as heavy load for few steps; environment waits hold a slot but no compute; large models use multi-card replicas whose width is chosen for throughput per card.
- **Training time**, closed-form from the batch's token counts, with a memory feasibility check.
- **Switch overhead**: engine restart plus reloading state, or a near-free resident switch with less KV-cache room.
- **Transfer overhead**: replicas pause to pull changed weights, faster on the trainer's node than across nodes.
- **KV rebuild**: after a sync, in-flight episodes either continue on a stale cache or re-prefill their context.
- **Grader cost**: CPU latency through the environment worker pool, or GPU tokens on the serving cards.
- **Off-policy measure**: per episode, token-weighted number of updates between the weights that generated it and the weights being updated.
- **Price of lag**: an episode of lag L counts as `1 / (1 + lag_alpha x L)` of a fresh one; the run ends at a target of on-policy-equivalent updates.

## Checks

| Check | Simulation | Expected |
|---|---|---|
| Time-shared, constant episodes, 3 updates | 79.06 s | 79.06 s |
| Split, trainer is the bottleneck, per update | 2.16 s | 2.15 s |
| Split, generation is the bottleneck, per update | 1087.1 s | 1089.0 s |
| Lag pricing: updates used at value 0.749 for target 20 | 27 | 26.7 |
| Calibration: recover four factors from synthetic measurements | exact | exact |

Also on eight policies with every feature on: episodes are conserved, card-time shares sum to one, and policies that should be on-policy report zero lag.

## Results

H100 cards, default factors, lag not priced (`lag_alpha` 0). Hours to target, averaged over 3 to 5 seeds; spread across seeds is under 3%.

| Scenario | Time-shared | Cut +25%, discard | Cut +25%, carry | Best split, stale KV | Best split, KV rebuild | Dynamic, KV rebuild |
|---|---|---|---|---|---|---|
| A. Agentic, 8B, 8 cards, batch 256 | 2.13 | 1.01 | 1.11 | 0.64 (4 train) | 1.08 | 0.93 |
| B. Reasoning, 8B, 8 cards, batch 256 | 1.53 | 0.63 | 0.75 | 0.34 (3) | 0.53 | 0.59 |
| C. Reasoning, 8B, 8 cards, batch 4096 | 1.36 | 1.07 | 1.27 | 1.30 (4) | 1.42 | 1.35 |
| D. Agentic, 8B, 2 cards | 3.42 | 2.38 | 2.70 | infeasible | infeasible | infeasible |
| E. Agentic, 70B full fine-tune, 32 cards | 3.01 | 2.21 | 2.32 | 3.01 (18) | 3.22 | 3.22 |
| F. Agentic, 70B adapters, 32 cards | 2.99 | 2.19 | 2.30 | 2.01 (10) | 2.22 | 2.53 |
| G. As B, generation length grows 6x | 3.25 | 0.91 | 1.09 | 0.46 (3) | 0.74 | 0.71 |
| H. Agentic, 8B, 4 H100 + 8 A100 | 2.46 | 1.17 | 1.29 | 0.72 (3, on H100) | 1.27 | 1.27 |
| I. As A, plus GPU judge, 64 env workers, correlated lengths | 3.74 | 1.37 | 1.54 | 0.84 (3) | 1.36 | 1.32 |

Mean lag of the split layouts is 0.4 to 1.5 updates at the default bound of two batches outstanding. "Cut, carry" has lag 0.1 to 0.2.

Price of lag, scenario A, hours to target:

| `lag_alpha` | Cut, discard | Cut, carry | Split 4, stale KV | Split 4, KV rebuild |
|---|---|---|---|---|
| 0 | 1.01 | 1.10 | 0.64 | 1.08 |
| 0.2 | 1.01 | 1.16 | 0.79 | 1.35 |
| 0.5 | 1.01 | 1.19 | 0.99 | 1.70 |
| 1.0 | 1.01 | 1.24 | 1.33 | 2.30 |

## What the model says

1. **Whether in-flight KV caches survive a weight sync is the largest single switch.** With stale caches the split is 1.5 to 2 times faster than the best time-shared variant in the small-batch scenarios. With a rebuild after every sync it takes 55 to 75% longer in those scenarios and roughly ties with a straggler cut. The first version of the model missed this cost.
2. **Plain time-sharing is the slowest feasible layout in every scenario.** In the small-batch scenarios the cards sit idle 54 to 78% of the time while rollouts are latency-bound.
3. **The straggler cut is strong but biased.** Discarding the unfinished gives batches whose mean length is 0.67 to 0.90 of the true mean. Carrying them over removes the bias for about 5 to 20% more time and a lag of 0.2.
4. **Batch size against fleet concurrency still decides the gap.** At batch 4096 on 8 cards (C) all layouts are within 30% and the cut wins.
5. **The split with stale caches stays ahead of the cut until lag is expensive.** In scenario A the break-even is near `lag_alpha` 0.5, meaning an update on data one update old is worth two thirds of a fresh one. With KV rebuild the split is behind at any price.
6. **Drift did not move the best ratio.** Longer generations lengthen training in proportion, so 3 training cards is best in both B and G. The audit's claim that drift is the main case for a dynamic layout is not supported here. The dynamic policy's use is finding a near-best ratio without a sweep: within 13% of the best fixed split in every scenario except the mixed fleet, where it is 28% worse.
7. **Memory feasibility can remove the choice.** A 2-card fleet cannot split an 8B full fine-tune; a 70B full fine-tune needs 18 of 32 cards for optimizer state, and then the split only matches plain time-sharing.
8. **On a mixed fleet, put the trainer on the fast cards.** 0.72 h with 3 H100 training against 0.80 h with 8 A100 training.
9. **The answer moves with the inference efficiency factor.** At 0.08 the cut beats the split with rebuild (1.18 against 1.38 h); at 0.30 the split wins (0.65 against 0.92 h). This is why the factors need measuring.

## Corrections to earlier statements

- The first outline said generation usually dominates. The first model said training is the larger side in compute, about 1:2.2. That ratio came from assumed efficiency factors; with the current defaults generation is about 1.2 times training. The ratio is not known until measured.
- The scope audit said drift moves the best split. See finding 6.
- 2026-11-27, aim restated by the user. This document treats "which layout is faster" as the goal. The goal is the model itself: a digital twin that explains where the card-time goes and can be checked against real runs (see `STATUS.md`, Aim). The results table above stays as a set of questions put to the model, not as the deliverable.

## Scope audit, status

| Gap | Status |
|---|---|
| Deploy is not a general interface | Fixed: policies over four primitives; dynamic split, partial rollouts, rolling sync are parameters |
| Time and lag cannot be combined | Fixed in mechanism: `lag_alpha` gives time to target. The value of `lag_alpha` needs experiment E9 |
| Stationary lengths | Fixed: `gen_drift`, `turns_drift`, `len_corr` |
| Missing costs | Fixed: KV rebuild, GPU judge, CPU grader latency, environment worker limit |
| Single seed | Fixed: results are means over seeds with spread |
| No calibration | Partly: `calibrate`, `fit_inference`, `factors.json` exist and round-trip. No measured run yet; see `experiment-plan.md` |
| One card type, no topology | Fixed: mixed fleets, nodes, intra-node and cross-node sync |
| Lag tagged at episode start | Fixed: token-weighted by the weights that generated each token |
| Length bias of the cut | Reported as an output; not priced |

## Factor defaults and where they come from

No GPU is available on the machine where this was built. Defaults were set loosely from two published numbers found by web search. I read them in search summaries and did not verify them at the source.

- Inference: a vLLM submission to MLPerf Inference v5.1 reports about 5,777 tokens/s for an 8B model in FP8 on one H100 ([Red Hat](https://www.redhat.com/en/blog/efficient-and-reproducible-llm-inference-red-hat-mlperf-inference-v51-results)). Counting output tokens only, that is an efficiency of about 0.09 in this model's terms, a lower bound. Default 0.15.
- Training: an NVIDIA recipe reports about 111,000 tokens/s for 8B pretraining on 8 H100 in FP8 ([NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dgxc-benchmarking/resources/llama31-8b-dgxc-benchmarking-d)), about 0.7 in this model's FLOP convention. RL fine-tuning on ragged batches should be lower. Default 0.55.

## Limits

- Factors are unmeasured. Absolute hours and the exact best ratio depend on them.
- `lag_alpha` is an input with an assumed functional form.
- Not modelled: draining a replica before syncing it (a third option besides stale cache and rebuild), syncing less often than every update, prefix sharing between samples of one prompt, a judge model's own memory footprint, trainer-side length bucketing.

## Open questions

- Does the inference engine keep in-flight KV caches across a weight swap, and is training on such mixed-policy episodes acceptable to the algorithm?
- What is `lag_alpha` for the method in question?
- Do the measured factors on H200 or H800 land near the defaults?
