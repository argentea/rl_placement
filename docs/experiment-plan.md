# Experiment plan: collecting the simulator's factors on H200 or H800

Date: 2026-11-27

## Purpose

The simulator's absolute numbers rest on about fifteen factors that are currently guesses. This plan measures them on real cards, writes them to `factors.json`, and then checks the simulator's prediction against one real run of each layout.

## Recommended setting

**8 x H200, one node, first.** It covers every factor except the two that need a second node. Move to 16 cards only for the items marked "16" below.

| | 8 cards, one node | 16 cards, two nodes |
|---|---|---|
| Single-card inference and training factors | Yes | Same |
| Training scaling | 1, 2, 4, 8 cards | Adds the 16-card point and cross-node scaling |
| Role-switch times | Yes | Same |
| Weight sync, same node | Yes | Same |
| Weight sync, across nodes | No | Yes |
| Validation of both layouts | Yes | Yes, and a second fleet size to test the model's scaling |
| Models larger than one card (tensor-parallel) | Up to about 70B | Needed for a full fine-tune of 70B |

Why H200 over H800 for the first pass: 141 GB leaves room to run the inference engine and the trainer on the same card, which is needed to measure the resident switch, and it fits a 70B replica on two cards. H800 is in the simulator with H100 compute numbers as a placeholder; I have not verified its specifications. If the production fleet is H800, repeat E1, E4 and E7 there (one to four cards is enough), since the factors are stored per card type.

Fixed across all experiments: the model you intend to train (or an 8B stand-in), 16-bit weights, and the same inference engine and trainer you would use in production. Factors measured on a different stack do not transfer.

## Experiments

Each row gives what to run, what to record, and which factor it sets.

| # | Cards | Run | Record | Sets |
|---|---|---|---|---|
| E1 | 1 | Inference engine, fixed 1000-token prompt, fixed 500 output tokens (ignore end-of-sequence). Concurrency 1, 2, 4, 8, 16, 32, 64, 128, 256, 512. Two minutes each after warm-up. | Total tokens/s (prompt plus output) at each concurrency | `infer_eff`, `membw_eff`, `step_overhead_s` via `fit_inference` |
| E2 | 1 | Same engine, contexts of 4k, 8k, 16k tokens. Raise concurrency until the engine queues or preempts. | Largest concurrency that runs without preemption at each context | `mem_usable`; checks the KV-cache size per token |
| E3 | 1 | Prompts of 4k tokens with one output token, at high concurrency | Prompt tokens/s | Checks the assumption that a prefill token costs the same compute as a saturated decode token |
| E4 | 1, 2, 4, 8 (16) | Trainer with the real optimizer and precision, sequence length 4096, batch sized to fill memory. Repeat at 1k and 8k on one card. | Tokens/s per card; peak memory per card | `train_eff`, `train_scale_eff`, `train_bytes_per_param`, `train_act_gb` |
| E5 | 8 | Five cycles of: stop the engine, start the trainer and run one step, stop the trainer, start the engine and serve one token. Repeat with both processes kept resident if memory allows. | Seconds from "stop" to first training step; seconds from "stop" to first served token | `switch_to_train_s`, `switch_to_infer_s`, `load_gbs`, `switch_resident_s` |
| E6 | 8 (16) | Running engine under load; swap in new weights from the trainer's checkpoint. Full weights and adapter-only. Same node, and across nodes on 16. | Seconds the engine stops serving; whether in-flight requests keep their KV cache or recompute it | `sync_fixed_s`, `intra_gbs`, `net_gbs`; decides `recompute` |
| E7 | 2, 4 | Only if the model needs more than one card: E1's sweep on a 2-card and a 4-card replica | Saturated tokens/s | `tp_eff` |
| E8 | 8 (16) | Ten updates of the real workload under time-sharing, and ten under the split the simulator recommends | Seconds per update, mean policy lag, card utilisation | Validation: compare with the simulator run on the fitted factors |
| E9 | 8 | Same RL task and update count, with the training batch artificially delayed by 0, 1, 2 and 4 updates | Updates needed to reach a fixed reward | `lag_alpha` |

E6 matters more than its size suggests. In the simulator, rebuilding KV caches after every sync adds 55 to 75% to the split layout's time in the small-batch 8B scenarios, which is the difference between the split clearly winning and roughly tying with a straggler cut.

E9 is the expensive one and the only one that depends on the RL algorithm. It can use a small model. Without it the simulator still gives time and lag separately.

## Order and cost

1. E1, E2, E3 on one card: about two hours.
2. E4: about two hours.
3. E5, E6: about two hours.
4. Put the numbers in `factors.json`, rerun `python scenarios.py L` for the prediction.
5. E8: the length of twenty updates of your workload.
6. E9 when the layouts are close enough that lag decides.

Times are my estimates, not measured.

## Feeding the results back

```python
from rlsim import CARDS, calibrate, cluster, fit_inference, load_factors

card = fit_inference(CARDS["H200"], model, sweep=[(1, 190), (2, 375), ...], ctx=1250)   # E1
card = calibrate(card, model, train_tok_s=9500, train_seq=4096)                         # E4
print(card)   # copy infer_eff, membw_eff, step_overhead_s, train_eff into factors.json

cl = load_factors(cluster(CARDS["H200"], 8), "factors.json")
```

The numbers in that snippet are placeholders. `factors.json` currently holds the simulator defaults; every key maps to the experiment that sets it.

## What the simulator predicts for these settings

With default factors, agentic 8B workload, batch 256, 40 updates. These are the numbers E8 would test.

| Deploy | 8 x H200 | 16 x H200 |
|---|---|---|
| Time-shared | 1.99 h | 1.79 h |
| Time-shared, cut +25%, discard | 0.96 h | 0.77 h |
| Time-shared, cut +25%, carry | 1.06 h | 0.81 h |
| Best fixed split, stale KV cache | 0.64 h (4 train) | 0.35 h (7 train) |
| Best fixed split, KV rebuild | 1.01 h | 0.60 h |
| Dynamic split, KV rebuild | 0.89 h | 0.57 h |

A useful property of E8: time-sharing barely improves from 8 to 16 cards (latency-bound) while the split nearly halves. If the real runs show that pattern, the model's main mechanism is confirmed even before the factors are exact.

## Not prepared

I have not written the load-generation or timing scripts for these experiments, because there is no GPU on this machine to test them against.
