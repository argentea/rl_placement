"""Scenarios for rlsim. Run: python scenarios.py [selftest | NAME ...]"""
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from rlsim import (CARDS, GB, Cluster, Dist, Dynamic, Engine, Model, Phased, Result, Split, best_split,
                   calibrate, cluster, evaluate, fit_inference, load_factors, make_simulator, run_many)

H100, A100 = CARDS["H100"], CARDS["A100"]
M8 = Model("8B dense", 8e9, 32, 4096)
M70 = Model("70B dense", 70e9, 80, 8192)
M70_LORA = replace(M70, name="70B dense + adapters", trainable_frac=0.005)

AGENT = Workload = None
from rlsim import Workload  # noqa: E402

AGENT = Workload(  # multi-turn tool use: short generations, slow environment
    prompt_tokens=Dist(2000, 0.3), gen_tokens=Dist(300, 0.8, cap=4096),
    turns=Dist(6, 0.6, cap=30), env_latency_s=Dist(3.0, 1.0), env_tokens=Dist(500, 0.8, cap=8000),
    episodes_per_update=256, updates=40)

REASON = Workload(  # single turn, long heavy-tailed generation, no environment
    prompt_tokens=Dist(500, 0.3), gen_tokens=Dist(3000, 0.9, cap=16000),
    episodes_per_update=256, updates=40)


def A():
    "agentic, 8B, 8 x H100, batch 256"
    evaluate(cluster(H100, 8), M8, AGENT)


def B():
    "reasoning, 8B, 8 x H100, batch 256"
    evaluate(cluster(H100, 8), M8, REASON)


def C():
    "reasoning, 8B, 8 x H100, batch 4096 (enough to saturate the fleet)"
    evaluate(cluster(H100, 8), M8, replace(REASON, episodes_per_update=4096, updates=10), seeds=3)


def D():
    "agentic, 8B, 2 x H100 (small fleet)"
    evaluate(cluster(H100, 2), M8, AGENT)


def E():
    "agentic, 70B full FT, 32 x H100, batch 512"
    evaluate(cluster(H100, 32), M70, replace(AGENT, episodes_per_update=512, updates=20), seeds=3)


def F():
    "agentic, 70B adapters, 32 x H100, batch 512"
    evaluate(cluster(H100, 32), M70_LORA, replace(AGENT, episodes_per_update=512, updates=20), seeds=3)


def G():
    "drift: reasoning, 8B, 8 x H100, generation length grows 6x over the run"
    evaluate(cluster(H100, 8), M8, replace(REASON, gen_tokens=Dist(1000, 0.9, cap=16000), gen_drift=6.0, updates=60))


def H():
    "mixed fleet: 4 x H100 + 8 x A100, agentic 8B; trainer on the fast cards, then on the slow cards"
    for pref in ("fast", "slow"):
        print(f"-- trainer takes the {pref} cards first")
        evaluate(Cluster(((H100, 4), (A100, 8)), train_on=pref), M8, AGENT, seeds=3)


def I():
    "scenario A plus the missing costs: GPU judge, 64 env workers, correlated lengths"
    w = replace(AGENT, judge_tokens=Dist(3000, 0.5), env_workers=64, len_corr=0.6, reward_latency_s=Dist(2.0, 0.5))
    evaluate(cluster(H100, 8), M8, w)


def J():
    "price of lag: scenario A, time to target as lag_alpha rises"
    pols = [Phased(), Phased(oversample=0.25), Phased(oversample=0.25, carry=True),
            Split(4, recompute=True), Split(4, max_outstanding=1.0, recompute=True), Dynamic(recompute=True)]
    for i, p in enumerate(pols):
        print(f"  [{i}] {p.name}")
    print(f"{'lag_alpha':<10}" + "".join(f"{'[%d]' % i:>9}" for i in range(len(pols))))
    with ProcessPoolExecutor() as pool:
        for a in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
            s = make_simulator(cluster(H100, 8), M8, replace(AGENT, lag_alpha=a))
            res = run_many(s, pols, 3, pool)
            print(f"{a:<10}" + "".join(f"{r.wall_s/3600:>7.2f} h" for r in res))
    print()


def K():
    "sensitivity: scenario A as the inference efficiency factor moves"
    with ProcessPoolExecutor() as pool:
        for ie in (0.08, 0.15, 0.30):
            s = make_simulator(cluster(replace(H100, infer_eff=ie), 8), M8, AGENT)
            n, hours = best_split(s, pool=pool)
            res = run_many(s, [Phased(), Phased(oversample=0.25), Split(n, recompute=True)], 3, pool)
            print(f"infer_eff {ie:.2f}: best split {n} train | " +
                  " | ".join(f"{r.name} {r.wall_s/3600:.2f} h" for r in res))
    print()


def L():
    "proposed experiment: 8 x H200, one node, agentic 8B (prediction with default factors)"
    evaluate(load_factors(cluster(CARDS["H200"], 8), "factors.json"), M8, AGENT)


def M():
    "proposed experiment: 16 x H200, two nodes, agentic 8B (prediction with default factors)"
    evaluate(load_factors(cluster(CARDS["H200"], 16), "factors.json"), M8, AGENT)


SCENARIOS = dict(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J, K=K, L=L, M=M)


def selftest():
    """Compare the event simulation against cases with closed-form answers, and check invariants."""
    card = replace(H100, step_overhead_s=0.0)
    cl = cluster(card, 4)
    PFC = 256.0
    # 1. Time-shared, constant single-turn episodes: every stream decodes G steps at a fixed load.
    P, G, B, U = 512, 1000, 64, 3
    w = Workload(Dist(P), Dist(G), episodes_per_update=B, updates=U)
    s = make_simulator(cl, M8, w, slots=10_000)
    rp = s.replica_perf(card)
    per = B // 4
    phase = (P / PFC) * rp.tau(per * PFC) + G * rp.tau(per)
    to_train = cl.switch_to_train_s + M8.train_state_bytes / 4 / GB / cl.load_gbs
    to_infer = cl.switch_to_infer_s + M8.infer_bytes / GB / cl.load_gbs
    expect = U * (phase + to_train + s.t_train([P + G] * B, 4)) + (U - 1) * to_infer
    got = s(Phased()).wall_s
    print(f"time-shared, constant episodes : sim {got:9.2f} s  closed form {expect:9.2f} s")
    assert abs(got - expect) / expect < 1e-6

    # 2. Split, generation over-provisioned and sync free: period = train step.
    w2 = Workload(Dist(P), Dist(50), episodes_per_update=B, updates=50)
    s2 = make_simulator(cluster(replace(card, infer_eff=10.0), 4, sync_fixed_s=0.0, net_gbs=1e9, intra_gbs=1e9), M8, w2)
    r = s2(Split(2))
    expect = s2.t_train([P + 50] * B, 2)
    print(f"split, trainer-bound           : sim {r.wall_s/r.updates:9.2f} s  closed form {expect:9.2f} s  lag {r.lag_mean:.2f}")
    assert abs(r.wall_s / r.updates - expect) / expect < 0.05

    # 3. Split, trainer instant: period = batch work / replica throughput + sync pause.
    w3 = Workload(Dist(P), Dist(4000), episodes_per_update=2048, updates=12)
    s3 = make_simulator(cluster(replace(card, train_eff=1e3), 4), M8, w3, slots=4096)
    rp = s3.replica_perf(s3.order[0][0])
    r = s3(Split(3, max_outstanding=4.0))
    expect = 2048 * (P + 4000) * rp.c + cl.sync_fixed_s + M8.sync_bytes / GB / cl.intra_gbs
    print(f"split, generation-bound        : sim {r.wall_s/r.updates:9.2f} s  closed form {expect:9.2f} s  lag {r.lag_mean:.2f}")
    assert abs(r.wall_s / r.updates - expect) / expect < 0.10

    # 4. Lag pricing: raw updates needed = target / mean value of an update.
    s4 = make_simulator(cluster(H100, 8), M8, replace(AGENT, lag_alpha=0.3, updates=20))
    r = s4(Split(4))
    print(f"lag pricing                    : {r.updates} updates at value {r.eff:.3f} for a target of 20 "
          f"(expected {20 / r.eff:.1f})")
    assert abs(r.updates - 20 / r.eff) <= 1.0 and r.eff < 1

    # 5. Invariants on every policy, on a workload that uses every feature.
    w5 = replace(AGENT, updates=8, judge_tokens=Dist(2000, 0.5), env_workers=48, len_corr=0.5,
                 reward_latency_s=Dist(1.0, 0.5), gen_drift=2.0, lag_alpha=0.1)
    cl5 = Cluster(((H100, 4), (A100, 4)))
    s5 = make_simulator(cl5, M8, w5)
    pols = [Phased(), Phased(resident=True), Phased(oversample=0.25), Phased(oversample=0.25, carry=True),
            Split(3), Split(3, recompute=True), Split(3, rolling=True, recompute=True), Dynamic(recompute=True)]
    for p in pols:
        eng = Engine(s5, p, 1).run()
        parked = sum(1 for _, _, fn, a in eng.ev if fn == eng._env_done and a[0].rep is None and not a[0].dead)
        parked += sum(1 for e in eng.env_q if e.rep is None and not e.dead)
        left = len(eng.buffer) + eng.inflight + len(eng.suspended) + eng.n_dead + parked
        assert eng.launch_n == eng.updates * 256 + left, (p.name, eng.launch_n, eng.updates * 256, left)
        r = s5(p, 1)
        assert r.feasible and abs(sum(r.shares) - 1) < 1e-9 and min(r.shares) > -1e-9, (p.name, r.shares)
        assert r.lag_mean >= -1e-9 and r.waste >= -1e-9
        if isinstance(p, Phased) and not p.carry:
            assert r.lag_max == 0, p.name
    print(f"invariants hold for {len(pols)} policies (episodes conserved, card-time closes, on-policy stays on-policy)")

    # 6. Calibration round-trips: synthesise measurements from known factors, recover them.
    true = replace(H100, infer_eff=0.22, membw_eff=0.6, step_overhead_s=0.003, train_eff=0.47)
    sc = make_simulator(cluster(true, 1), M8, Workload(Dist(1000), Dist(1)))
    rp = sc.replica_perf(true)
    ctx = sc.ep_tokens
    sweep = [(a, a / rp.tau(a)) for a in (1, 8, 64, 512, 1024)]
    fit = fit_inference(H100, M8, sweep, 0.6 * ctx)
    seq = 4096
    fit = calibrate(fit, M8, train_tok_s=true.flops * true.train_eff / (3 * M8.flops_per_token(seq / 2)), train_seq=seq)
    print(f"calibration round-trip         : infer_eff {fit.infer_eff:.3f}, membw_eff {fit.membw_eff:.3f}, "
          f"overhead {fit.step_overhead_s*1e3:.2f} ms, train_eff {fit.train_eff:.3f}")
    for k in ("infer_eff", "membw_eff", "step_overhead_s", "train_eff"):
        assert math.isclose(getattr(fit, k), getattr(true, k), rel_tol=1e-6), k


if __name__ == "__main__":
    args = sys.argv[1:]
    if "selftest" in args:
        selftest()
    else:
        for k in args or SCENARIOS:
            print(f"=== {k}. {SCENARIOS[k].__doc__}")
            SCENARIOS[k]()
