# Status

Last updated: 2026-11-27

## Done

- Simulator with deploy policies over four primitives (roles, admission, train, sync). Self-tests pass.
- Scenarios A to M run; results in `results.txt`, findings in `docs/design-and-findings.md`.
- Factor file, fitting helpers (`fit_inference`, `calibrate`, `load_factors`), experiment plan.

## Current answer, provisional

With default factors, at small batch relative to fleet concurrency:

- A split that keeps in-flight KV caches across weight syncs is 1.5 to 2 times faster than any time-shared variant, at 0.4 to 1.5 updates of lag.
- If caches must be rebuilt after each sync, the split roughly ties with time-sharing plus a straggler cut.
- Plain time-sharing is slowest. At large batch all layouts are within 30%.

This rests on unmeasured factors and an unpriced lag.

## Next, in order

1. **Write the measurement scripts for E1 to E6** (`docs/experiment-plan.md`). Not started. They cannot be tested here; design them to run unattended on the remote cards and emit JSON that maps onto `factors.json`.
2. **User runs E1 to E6 on 8 x H200** (or H800) and returns the numbers.
3. **Fit factors, rerun scenario L**, then compare with E8 (ten real updates under each layout).
4. **E9**: measure `lag_alpha` if the layouts are close enough that lag decides.

## Open decisions for the user

- Which model, inference engine and trainer the experiments should use. Factors do not transfer across stacks.
- The real workload: prompt, turn, generation and latency distributions, batch size. Scenarios use invented agentic and reasoning workloads.
- Whether training on episodes generated partly under older weights (stale KV cache) is acceptable to the algorithm.

## Known gaps in the model

- Draining a replica before syncing it, as a third option besides stale cache and rebuild.
- Syncing less often than every update.
- Prefix sharing between samples of one prompt.
- A judge model's own memory footprint.
- Trainer-side length bucketing.
- The straggler cut's length bias is reported but not priced.
