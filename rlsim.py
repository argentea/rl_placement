"""Placement simulator for RL on LLM policies: share cards in time, or split them in space?

    sim = make_simulator(cluster, model, workload)     # inputs -> a function
    res = sim(Phased())                                # deploy policy -> Result
    res = sim(Split(n_train=5))
    res = sim(Dynamic())

A deploy is a policy over a small set of engine primitives:

    eng.set_roles(n_train, carry=...)   which cards train, which serve; pays switch costs
    eng.admit() / policy.want(eng)      admission control (batch size, oversampling, backpressure)
    eng.train()                         one update on the oldest finished batch
    eng.sync(rolling=, recompute=)      push new weights to the serving replicas

Time-sharing, straggler cut, partial rollouts, static split, rolling sync and dynamic split
are all parameters of three short policies built on those primitives.

Result reports the time to reach a target number of on-policy-equivalent updates, the policy
lag of the consumed data, and where the card-time went. Rollouts are simulated event by event;
training steps, switches and transfers are closed-form costs. Efficiency factors live on Card
and can be fitted from measurements with calibrate().
"""
from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

GB = 1e9
PF, GEN, ENV, XPF, JUDGE = 0, 1, 2, 3, 4   # XPF rebuilds a KV cache; JUDGE is grader work on the GPU
PF_CHUNK = 256.0


# ============================================================================= inputs

@dataclass(frozen=True)
class Card:
    name: str
    flops: float       # peak dense 16-bit FLOP/s
    mem_gb: float
    membw_gbs: float   # memory bandwidth, GB/s
    # Achieved fractions of peak. Defaults are loose fits to published numbers (see the write-up);
    # use calibrate() with your own measurements.
    infer_eff: float = 0.15
    train_eff: float = 0.55
    membw_eff: float = 0.85
    step_overhead_s: float = 0.001


CARDS = {
    "A100": Card("A100-80G", 312e12, 80, 2039),
    "H100": Card("H100-80G", 989e12, 80, 3350),
    "H200": Card("H200-141G", 989e12, 141, 4800),
    "H800": Card("H800-80G", 989e12, 80, 3350),   # H100 compute with a slower card-to-card link
}


@dataclass(frozen=True)
class Cluster:
    groups: tuple                  # ((Card, count), ...), in rack order
    cards_per_node: int = 8
    train_on: str = "fast"         # which cards the trainer takes first: "fast" or "slow"
    train_scale_eff: float = 0.97  # multi-card training efficiency per doubling
    tp_eff: float = 0.85           # tensor-parallel inference efficiency per doubling
    mem_usable: float = 0.90
    load_gbs: float = 8.0          # host RAM -> device, per card (role switch)
    intra_gbs: float = 8.0         # weight sync when trainer and replica share a node
    net_gbs: float = 10.0          # weight sync across nodes
    switch_to_train_s: float = 5.0
    switch_to_infer_s: float = 10.0   # engine re-init and warm-up
    switch_resident_s: float = 1.0    # both states stay on the card
    sync_fixed_s: float = 2.0
    train_step_fixed_s: float = 0.5

    @property
    def n(self):
        return sum(c for _, c in self.groups)

    def order(self):
        """[(card, node)], trainer-preferred first."""
        cards = [card for card, c in self.groups for _ in range(c)]
        o = [(card, i // self.cards_per_node) for i, card in enumerate(cards)]
        return sorted(o, key=lambda x: -x[0].flops if self.train_on == "fast" else x[0].flops)


def cluster(card: Card, n: int, **kw) -> Cluster:
    return Cluster(((card, n),), **kw)


@dataclass(frozen=True)
class Model:
    name: str
    params: float
    n_layers: int
    d_model: int
    active_params: float | None = None  # MoE: params touched per token; None = dense
    kv_frac: float = 0.25               # kv heads / attention heads (GQA)
    infer_bytes_per_param: float = 2.0
    train_bytes_per_param: float = 16.0 # fp32 weight + grad + Adam m, v
    trainable_frac: float = 1.0         # < 1 means adapters on a frozen 16-bit base
    train_fwd_mult: float = 3.0         # backward ~ 2x forward
    train_act_gb: float = 8.0           # activation headroom per card

    @property
    def p_act(self):
        return self.active_params or self.params

    def flops_per_token(self, ctx):
        return 2 * self.p_act + 4 * self.n_layers * self.d_model * ctx

    @property
    def infer_bytes(self):
        return self.params * self.infer_bytes_per_param

    @property
    def train_state_bytes(self):
        f = self.trainable_frac
        return self.params * ((1 - f) * 2.0 + f * self.train_bytes_per_param)

    @property
    def sync_bytes(self):
        return self.params * self.trainable_frac * self.infer_bytes_per_param

    @property
    def kv_bytes_per_token(self):
        return 2 * self.n_layers * self.d_model * self.kv_frac * 2.0


_erf = np.vectorize(math.erf, otypes=[float])


@dataclass(frozen=True)
class Dist:
    """Lognormal by mean and sigma, clipped; or an empirical sample via `values`."""
    mean: float = 0.0
    sigma: float = 0.0
    cap: float = math.inf
    lo: float = 0.0
    values: tuple | None = None

    def from_z(self, z):
        """Map standard normals to samples, so two quantities can share a correlated z."""
        if self.values is not None:
            return np.quantile(np.asarray(self.values, float), 0.5 * (1 + _erf(z / math.sqrt(2))))
        if self.mean <= 0:
            return np.zeros(len(z))
        if self.sigma == 0:
            return np.clip(np.full(len(z), float(self.mean)), self.lo, self.cap)
        return np.clip(np.exp(math.log(self.mean) - self.sigma ** 2 / 2 + self.sigma * z), self.lo, self.cap)

    def sample(self, rng, n):
        return self.from_z(rng.standard_normal(n))


@dataclass(frozen=True)
class Workload:
    """The data set and the RL recipe, as seen by the hardware."""
    prompt_tokens: Dist
    gen_tokens: Dist                 # generated per turn
    turns: Dist = Dist(1)
    env_latency_s: Dist = Dist(0)    # wait between turns, GPU not involved
    env_tokens: Dist = Dist(0)       # tokens the environment adds between turns
    len_corr: float = 0.0            # correlation between an episode's turn count and its tokens per turn
    gen_drift: float = 1.0           # multiplier on tokens per turn reached at the end of the run
    turns_drift: float = 1.0         # same for the turn count; both interpolate geometrically in progress
    env_workers: int | None = None   # concurrent environment / grader workers; None = unlimited
    reward_latency_s: Dist = Dist(0) # CPU grading at the end of an episode (uses an env worker)
    judge_tokens: Dist = Dist(0)     # GPU grading: tokens a judge model reads, served by the inference cards
    judge_cost: float = 1.0          # judge FLOPs per token relative to the policy model
    episodes_per_update: int = 256
    updates: int = 100               # target, in on-policy-equivalent updates
    epochs: int = 1
    lag_alpha: float = 0.0           # an episode of lag L counts as 1 / (1 + alpha * L) of a fresh one


class Episode:
    __slots__ = ("segs", "i", "tokens", "rep", "stale", "in_env", "dead", "ctx", "vsum", "gtok")

    def __init__(self, segs, tokens):
        self.segs, self.i, self.tokens = segs, 0, tokens
        self.rep, self.stale, self.in_env, self.dead = None, False, False, False
        self.ctx = self.vsum = self.gtok = 0.0


def sample_episodes(w: Workload, rng, n, p=0.0):
    """p in [0, 1] is training progress; it moves the length distributions if drift is set."""
    gs, ts = w.gen_drift ** p, w.turns_drift ** p
    z = rng.standard_normal(n)
    prompt = np.maximum(1, w.prompt_tokens.sample(rng, n))
    turns = np.maximum(1, np.rint(w.turns.from_z(z) * ts)).astype(int)
    tot = int(turns.sum())
    zt = w.len_corr * np.repeat(z, turns) + math.sqrt(1 - w.len_corr ** 2) * rng.standard_normal(tot)
    gen = np.maximum(1, np.rint(w.gen_tokens.from_z(zt) * gs))
    lat = w.env_latency_s.sample(rng, tot)
    etok = np.rint(w.env_tokens.sample(rng, tot))
    rlat = w.reward_latency_s.sample(rng, n)
    jtok = w.judge_tokens.sample(rng, n) * w.judge_cost
    out, j = [], 0
    for e in range(n):
        segs, tokens = [(PF, float(prompt[e]))], prompt[e]
        for t in range(turns[e]):
            segs.append((GEN, float(gen[j])))
            tokens += gen[j]
            if t < turns[e] - 1:
                if lat[j] > 0:
                    segs.append((ENV, float(lat[j])))
                if etok[j] > 0:
                    segs.append((PF, float(etok[j])))
                    tokens += etok[j]
            j += 1
        if rlat[e] > 0:
            segs.append((ENV, float(rlat[e])))
        if jtok[e] > 0:
            segs.append((JUDGE, float(jtok[e])))
        out.append(Episode(segs, float(tokens)))
    return out


def calibrate(card: Card, model: Model, *, stream_tok_s=None, sat_tok_s=None, sat_ctx=1000,
              train_tok_s=None, train_seq=4096) -> Card:
    """Fit the efficiency factors of a card from measurements of `model` on it.

    stream_tok_s: decode speed of one stream at low concurrency.
    sat_tok_s:    total tokens/s (prefill + decode) of one card at high concurrency, mean context sat_ctx.
    train_tok_s:  training tokens/s per card at sequence length train_seq.
    """
    kw = {}
    if stream_tok_s:
        t = 1.0 / stream_tok_s - card.step_overhead_s
        kw["membw_eff"] = model.infer_bytes / (card.membw_gbs * GB * t)
    if sat_tok_s:
        kw["infer_eff"] = sat_tok_s * model.flops_per_token(sat_ctx) / card.flops
    if train_tok_s:
        kw["train_eff"] = train_tok_s * model.train_fwd_mult * model.flops_per_token(train_seq / 2) / card.flops
    return replace(card, **kw)


def fit_inference(card: Card, model: Model, sweep, ctx) -> Card:
    """Fit the three inference factors from a concurrency sweep on ONE card.

    sweep: [(concurrent streams, total tokens/s processed)], from 1 stream to well past the knee.
    ctx:   mean context length during the sweep.
    The step time a / tok_s is max(tau_mem, a * c) + tau0. The two most loaded points give the
    slope c and intercept tau0; the single-stream point then gives tau_mem."""
    sw = sorted(sweep)
    (a1, t1), (a2, t2) = [(a, a / tps) for a, tps in sw[-2:]]
    c = (t2 - t1) / (a2 - a1)
    tau0 = max(0.0, t2 - a2 * c)
    tau_mem = max(1e-6, sw[0][0] / sw[0][1] - tau0)
    return replace(card, step_overhead_s=tau0,
                   infer_eff=model.flops_per_token(ctx) / (c * card.flops),
                   membw_eff=model.infer_bytes / (card.membw_gbs * GB * tau_mem))


def with_factors(cl: Cluster, factors: dict) -> Cluster:
    """Override measured factors. factors = {"cards": {card name: {field: value}}, "cluster": {field: value}}.
    Card fields: infer_eff, train_eff, membw_eff, step_overhead_s (also flops, mem_gb, membw_gbs).
    Cluster fields: any Cluster field, e.g. switch_to_infer_s, load_gbs, net_gbs, tp_eff, train_scale_eff."""
    cards = factors.get("cards", {})
    groups = tuple((replace(card, **cards.get(card.name, {})), n) for card, n in cl.groups)
    return replace(cl, groups=groups, **factors.get("cluster", {}))


def load_factors(cl: Cluster, path) -> Cluster:
    import json

    def strip(d):
        return {k: strip(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith("_")}

    with open(path) as f:
        return with_factors(cl, strip(json.load(f)))


# ============================================================================= derived performance

@dataclass
class RepPerf:
    """One inference replica: k cards holding one copy of the weights."""
    k: int
    c: float          # seconds of replica compute per unit of load per step
    tau_mem: float    # floor on a decode step: one pass over the weights
    tau0: float
    slots: int        # concurrent episodes
    slots_mem: int

    def tau(self, load):
        return max(self.tau_mem, load * self.c) + self.tau0

    @property
    def tok_s_sat(self):
        return 1.0 / self.c

    @property
    def tok_s_stream(self):
        return 1.0 / (self.tau_mem + self.tau0)


class System:
    def __init__(self, cl: Cluster, model: Model, workload: Workload, sat_factor=3.0, slots=None):
        self.cl, self.m, self.w = cl, model, workload
        self.sat_factor, self.slots_override = sat_factor, slots
        self.order = cl.order()
        self.N = len(self.order)
        st = []
        for p in (0.0, 1.0):
            probe = sample_episodes(workload, np.random.default_rng(12345), 4000, p)
            st.append([float(np.mean([e.tokens for e in probe]))] + [
                float(np.mean([sum(a for k, a in e.segs if k in kinds) for e in probe]))
                for kinds in ((GEN,), (PF, JUDGE), (ENV,))])
        st = np.asarray(st)
        self.ep_tokens, self.ep_gen, self.ep_pf, self.ep_env = st.mean(0)
        self.ep_tokens_kv = st[:, 0].max()   # size the KV cache for the longer end of the drift
        self._rp, self._chunks = {}, {}

    def __call__(self, policy, seed=0):
        return run(self, policy, seed)

    # -- inference
    def replica_perf(self, card: Card, reserved_gb=0.0, k=None) -> RepPerf | None:
        """Width k is chosen for the best estimated tokens/s per card: wider replicas lose
        tensor-parallel efficiency but leave more room for KV cache, hence more slots."""
        key = (card.name, round(reserved_gb, 3), k)
        if key not in self._rp:
            usable = (card.mem_gb * self.cl.mem_usable - reserved_gb) * GB
            best, score = None, 0.0
            if usable > 0:
                k_min = max(1, math.ceil(self.m.infer_bytes / usable + 1e-9))
                for kk in ([k] if k else range(k_min, k_min + 5)):
                    rp, load = self._perf_k(card, kk, usable)
                    if rp is not None and load / rp.tau(load) / kk > score * 1.02:
                        best, score = rp, load / rp.tau(load) / kk
            self._rp[key] = best
        return self._rp[key]

    def _perf_k(self, card, k, usable):
        cl, m = self.cl, self.m
        c = m.flops_per_token(0.6 * self.ep_tokens) / (k * cl.tp_eff ** math.log2(k) * card.flops * card.infer_eff)
        tau_mem = m.infer_bytes / (k * card.membw_gbs * GB * card.membw_eff)
        slots_mem = int((k * usable - m.infer_bytes) / (m.kv_bytes_per_token * self.ep_tokens_kv))
        if slots_mem < 1:
            return None, 0.0
        rp = RepPerf(k, c, tau_mem, card.step_overhead_s, slots_mem, slots_mem)
        # Enough concurrency to sit sat_factor past the compute knee, counting slots parked on the env.
        load_star = self.sat_factor * (tau_mem + rp.tau0) / c
        steps = self.ep_gen + self.ep_pf / PF_CHUNK
        t_gpu = steps * rp.tau(load_star)
        duty = t_gpu / (t_gpu + self.ep_env)
        w_avg = (self.ep_gen + self.ep_pf) / steps
        rp.slots = self.slots_override or max(1, min(slots_mem, math.ceil(load_star / (duty * w_avg))))
        return rp, rp.slots * duty * w_avg

    def chunks(self, n_train, reserved_gb=0.0):
        """Replicas formed from the cards after the first n_train: [(card indices, RepPerf)].
        Cards are grouped by (type, node) and chunked from the tail, so moving the boundary
        only disturbs the replica next to it."""
        key = (n_train, round(reserved_gb, 3))
        if key not in self._chunks:
            out, i = [], self.N
            while i > n_train:
                j = i
                while j > n_train and (self.order[j - 1][0].name, self.order[j - 1][1]) == \
                        (self.order[i - 1][0].name, self.order[i - 1][1]):
                    j -= 1
                rp = self.replica_perf(self.order[i - 1][0], reserved_gb)
                if rp is not None:
                    e = i
                    while e - rp.k >= j:
                        out.append((tuple(range(e - rp.k, e)), rp))
                        e -= rp.k
                i = j
            self._chunks[key] = out
        return self._chunks[key]

    # -- training
    def train_ok(self, n, extra_gb=0.0):
        if n < 1 or n > self.N:
            return False
        need = self.m.train_state_bytes / n / GB + self.m.train_act_gb + extra_gb
        return need <= min(c.mem_gb for c, _ in self.order[:n]) * self.cl.mem_usable

    def train_rate(self, n):
        return sum(c.flops * c.train_eff for c, _ in self.order[:n]) * self.cl.train_scale_eff ** math.log2(n)

    def t_train(self, tokens, n):
        """tokens: per-episode token counts of the batch."""
        m = self.m
        t = np.asarray(tokens, float)
        fl = m.train_fwd_mult * self.w.epochs * (2 * m.p_act * t.sum() + 2 * m.n_layers * m.d_model * (t ** 2).sum())
        return fl / self.train_rate(n) + self.cl.train_step_fixed_s

    def describe(self):
        m, w = self.m, self.w
        B = w.episodes_per_update
        fleet = ", ".join(f"{c} x {card.name}" for card, c in self.cl.groups)
        lines = [
            f"{fleet} | {m.name}: {m.params/1e9:.0f}B params, "
            f"{'full FT' if m.trainable_frac == 1 else f'adapters {m.trainable_frac:.1%}'}",
            f"episode (run average): {self.ep_tokens:.0f} tok ({self.ep_gen:.0f} generated), "
            f"{self.ep_env:.1f} s env wait | batch {B}, target {w.updates} updates, lag_alpha {w.lag_alpha}",
        ]
        ch = self.chunks(0)
        for name in dict.fromkeys(c.name for c, _ in self.order):
            card = next(c for c, _ in self.order if c.name == name)
            rp = self.replica_perf(card)
            if rp:
                lines.append(
                    f"replica on {name}: {rp.k} card(s), {rp.tok_s_stream:.0f} tok/s per stream, "
                    f"{rp.tok_s_sat:.0f} tok/s saturated, {rp.slots} slots (memory allows {rp.slots_mem})")
        if self.train_ok(self.N):
            tt = self.t_train(np.full(B, self.ep_tokens), self.N)
            lines.append(f"all cards serving: {sum(rp.slots for _, rp in ch)} slots | "
                         f"all cards training: {tt:.1f} s per step")
        return "\n".join(lines)


def make_simulator(cl, model, workload, **kw) -> System:
    return System(cl, model, workload, **kw)


# ============================================================================= engine

class Replica:
    __slots__ = ("rp", "cards", "V", "t", "load", "heap", "used", "paused", "version", "stamp",
                 "util", "alive", "eps")

    def __init__(self, rp, cards):
        self.rp, self.cards = rp, cards
        self.V = self.t = self.load = self.util = 0.0
        self.heap, self.used, self.paused, self.version, self.stamp = [], 0, 0, 0, 0
        self.alive, self.eps = True, set()


class Engine:
    """Event loop plus the primitives that deploy policies are written in.

    Each replica has a virtual clock that ticks once per decode step; a step takes
    max(tau_mem, load * c) + tau0 seconds. A decoding stream is load 1; a prefill is load
    PF_CHUNK for tokens / PF_CHUNK steps."""

    def __init__(self, s: System, policy, seed=0):
        self.s, self.policy, self.w = s, policy, s.w
        self.rng = np.random.default_rng(seed)
        self.ev, self.cnt, self.now = [], itertools.count(), 0.0
        self.reps, self.trainer_n, self.trainer_ready, self.training = [], 0, 0.0, False
        self.started = False
        self.version, self.updates, self.progress, self.end = 0, 0, 0.0, 0.0
        self.buffer, self.suspended, self.pool = [], deque(), []
        self.env_free, self.env_q = s.w.env_workers, deque()
        self.admitted = self.n_dead = 0
        self.ps = {}                       # policy state
        # controller signals
        self.trainer_free_at = self.last_done = 0.0
        self.starved_last = self.last_period = 0.0
        self.blocked, self.blocked_since = 0.0, None
        # accounting
        self.gen_cs = self.train_cs = self.switch_cs = 0.0
        self.lags, self.effs = [], []
        self.tok_used = self.tok_wasted = self.tok_rebuild = 0.0
        self.launch_tok = self.launch_n = 0.0
        self.roles_log = []

    # -- event loop
    def push(self, t, fn, *args):
        heapq.heappush(self.ev, (t, next(self.cnt), fn, args))

    def run(self):
        self.policy.start(self)
        self.started = True
        self.admit()
        while self.progress < self.w.updates - 1e-9:
            if not self.ev:
                raise RuntimeError(f"{self.policy.name}: stalled at t={self.now:.0f}s, update {self.updates}")
            t, _, fn, args = heapq.heappop(self.ev)
            self.now = t
            fn(*args)
        return self

    # -- replica mechanics
    def _advance(self, r):
        dt = self.now - r.t
        if dt > 0:
            if not r.paused and r.load > 0:
                tau = r.rp.tau(r.load)
                r.V += dt / tau
                r.util += dt * min(1.0, r.load * r.rp.c / tau)
            r.t = self.now

    def _resched(self, r):
        r.stamp += 1
        if r.alive and r.heap and not r.paused:
            dv = max(0.0, r.heap[0][0] - r.V)
            self.push(self.now + dv * r.rp.tau(r.load), self._on_rep, r, r.stamp)

    @staticmethod
    def _account(ep, kind, amt, version):
        if kind == PF:
            ep.ctx += amt
        elif kind == GEN:
            ep.ctx += amt
            ep.vsum += version * amt
            ep.gtok += amt

    def _start_seg(self, ep):
        """Caller has advanced ep.rep."""
        kind, amt = ep.segs[ep.i]
        if kind == ENV:
            ep.in_env = True
            if self.env_free is None:
                self.push(self.now + amt, self._env_done, ep)
            elif self.env_free > 0:
                self.env_free -= 1
                self.push(self.now + amt, self._env_done, ep)
            else:
                self.env_q.append(ep)
            return
        r = ep.rep
        if r is None:
            self.suspended.append(ep)
            return
        if ep.stale:
            ep.stale = False
            if ep.ctx > 0:   # KV cache is gone or belongs to old weights: rebuild it first
                ep.segs.insert(ep.i, (XPF, ep.ctx))
                kind, amt = XPF, ep.ctx
                self.tok_rebuild += amt
        w = 1.0 if kind == GEN else PF_CHUNK
        heapq.heappush(r.heap, (r.V + amt / w, next(self.cnt), ep, w))
        r.load += w

    def _next(self, ep):
        ep.i += 1
        if ep.i < len(ep.segs):
            self._start_seg(ep)
            return
        r = ep.rep
        if r is not None:
            r.used -= 1
            r.eps.discard(ep)
            ep.rep = None
        self.buffer.append(ep)
        self.push(self.now, self.policy.on_episode, self)

    def _on_rep(self, r, stamp):
        if stamp != r.stamp or not r.alive:
            return
        self._advance(r)
        while r.heap and r.heap[0][0] <= r.V + 1e-7:
            _, _, ep, w = heapq.heappop(r.heap)
            r.load -= w
            kind, amt = ep.segs[ep.i]
            self._account(ep, kind, amt, r.version)
            self._next(ep)
        if not r.heap:
            r.load = 0.0
        self._resched(r)
        self.admit()

    def _env_done(self, ep):
        ep.in_env = False
        if self.env_free is not None:
            self.env_free += 1
            while self.env_q and self.env_free > 0:
                nxt = self.env_q.popleft()
                if not nxt.dead:
                    self.env_free -= 1
                    self.push(self.now + nxt.segs[nxt.i][1], self._env_done, nxt)
        if ep.dead:
            return
        r = ep.rep
        if r is not None:
            self._advance(r)
        self._next(ep)
        if r is not None:
            self._resched(r)
        else:
            self.admit()

    def _strip(self, r):
        """Stop every GPU segment on r, crediting partial progress. Returns the episodes stopped."""
        self._advance(r)
        eps = []
        for vfin, _, ep, w in r.heap:
            kind, amt = ep.segs[ep.i]
            if kind == XPF:          # a half-built cache is worth nothing
                self.tok_rebuild -= min(amt, max(0.0, (vfin - r.V) * w))
                del ep.segs[ep.i]
                ep.stale = True
            else:
                rem = min(amt, max(0.0, (vfin - r.V) * w))
                self._account(ep, kind, amt - rem, r.version)
                ep.segs[ep.i] = (kind, max(rem, 1e-6))
            eps.append(ep)
        r.heap, r.load = [], 0.0
        r.stamp += 1
        return eps

    def _detach(self, r, carry):
        """Tear a replica down. carry=True suspends its episodes for later; False discards them."""
        self._strip(r)
        for ep in r.eps:
            ep.rep = None
            if carry:
                ep.stale = True
                if not ep.in_env:
                    self.suspended.append(ep)
            else:
                ep.dead = True
                self.n_dead += 1
                self.tok_wasted += ep.ctx
        r.eps = set()
        r.used, r.alive = 0, False
        self.gen_cs += r.rp.k * r.util
        self.reps.remove(r)

    def _invalidate(self, r):
        """New weights arrived: every in-flight episode on r must rebuild its KV cache."""
        eps = self._strip(r)
        for ep in r.eps:
            ep.stale = True
        for ep in eps:
            self._start_seg(ep)

    def _pause(self, reps, dur, recompute):
        """Pause a group of replicas for dur seconds; they come back together so admission stays balanced."""
        reps = [r for r in reps if r.alive]
        for r in reps:
            self._advance(r)
            r.paused += 1
            r.stamp += 1
            self.switch_cs += r.rp.k * dur
        self.push(self.now + dur, self._unpause, reps, recompute)

    def _unpause(self, reps, recompute):
        for r in reps:
            if not r.alive:
                continue
            self._advance(r)
            r.paused -= 1
            if not r.paused:
                r.version = self.version
                if recompute:
                    self._invalidate(r)
                self._resched(r)
        self.admit()

    # -- primitives for policies
    @property
    def inflight(self):
        return sum(r.used for r in self.reps)

    def _fresh(self):
        if not self.pool:
            p = min(1.0, self.progress / self.w.updates)
            self.pool = sample_episodes(self.w, self.rng, 64, p)
        ep = self.pool.pop()
        self.launch_tok += ep.tokens
        self.launch_n += 1
        return ep

    def admit(self):
        touched = set()
        while True:
            r = min((x for x in self.reps if not x.paused and x.used < x.rp.slots),
                    key=lambda x: x.used / x.rp.slots, default=None)
            want = r is not None and self.policy.want(self)
            if r is not None and not want:
                if self.blocked_since is None:
                    self.blocked_since = self.now
            elif self.blocked_since is not None:
                self.blocked += self.now - self.blocked_since
                self.blocked_since = None
            if not want:
                break
            ep = self.suspended.popleft() if self.suspended else self._fresh()
            self._advance(r)
            r.used += 1
            r.eps.add(ep)
            ep.rep = r
            self.admitted += 1
            self._start_seg(ep)
            touched.add(r)
        for r in touched:
            self._resched(r)

    def take_blocked(self):
        """Seconds since the last call during which free slots went unused because the policy refused."""
        if self.blocked_since is not None:
            self.blocked += self.now - self.blocked_since
            self.blocked_since = self.now
        b, self.blocked = self.blocked, 0.0
        return b

    def set_roles(self, n_train, carry=True, resident=False, then=None):
        """First n_train cards (in trainer-preferred order) train; the rest serve.
        Replicas whose cards change role are torn down; new ones come up after the switch."""
        s, cl, m = self.s, self.s.cl, self.s.m
        reserved = m.train_state_bytes / s.N / GB if resident else 0.0
        want = dict(s.chunks(n_train, reserved))
        for r in list(self.reps):
            if r.cards not in want:
                self._detach(r, carry)
        have = {r.cards for r in self.reps}
        groups = {}
        for cards, rp in want.items():
            if cards in have:
                continue
            delay = 0.0 if not self.started else (
                cl.switch_resident_s if resident else
                cl.switch_to_infer_s + m.infer_bytes / rp.k / GB / cl.load_gbs)
            r = Replica(rp, cards)
            r.t = self.now
            self.reps.append(r)
            groups.setdefault(delay, []).append(r)
        for delay, reps in groups.items():
            self._pause(reps, delay, False)
        if n_train != self.trainer_n:
            delay = 0.0
            if self.started and n_train > self.trainer_n:
                delay = cl.switch_resident_s if resident else (
                    cl.switch_to_train_s + m.train_state_bytes / n_train / GB / cl.load_gbs)
            self.trainer_n, self.trainer_ready = n_train, self.now + delay
            self.switch_cs += n_train * delay
        self.roles_log.append((self.now, n_train))
        if then is not None:
            self.push(max(self.now, self.trainer_ready), then, self)

    def train(self):
        """Start one update on the oldest finished batch."""
        B, alpha = self.w.episodes_per_update, self.w.lag_alpha
        assert not self.training and len(self.buffer) >= B and self.trainer_n > 0
        batch = self.buffer[:B]
        del self.buffer[:B]
        lag = np.array([self.version - (e.vsum / e.gtok if e.gtok else self.version) for e in batch])
        eff = float(np.mean(1.0 / (1.0 + alpha * lag)))
        self.lags.extend(lag)
        self.effs.append(eff)
        self.tok_used += sum(e.tokens for e in batch)
        tt = self.s.t_train([e.tokens for e in batch], self.trainer_n)
        start = max(self.now, self.trainer_ready)
        self.starved_last = max(0.0, start - max(self.trainer_free_at, self.trainer_ready))
        self.training = True
        self.train_cs += self.trainer_n * tt
        self.push(start + tt, self._train_done, eff)

    def _train_done(self, eff):
        self.training = False
        self.version += 1
        self.updates += 1
        self.progress += eff
        self.last_period, self.last_done = self.now - self.last_done, self.now
        self.end = self.trainer_free_at = self.now
        if self.progress < self.w.updates - 1e-9:
            self.policy.on_train_done(self)

    def sync(self, rolling=False, recompute=False):
        """Push the latest weights to every serving replica. Each pauses while it loads.
        rolling: one replica at a time. recompute: in-flight episodes rebuild their KV cache
        under the new weights (otherwise they continue on a cache from the old ones)."""
        cl, m = self.s.cl, self.s.m
        nodes = {node for _, node in self.s.order[:self.trainer_n]}
        t0, groups = 0.0, {}
        for r in list(self.reps):
            local = self.s.order[r.cards[0]][1] in nodes
            dur = cl.sync_fixed_s + m.sync_bytes / GB / (cl.intra_gbs if local else cl.net_gbs)
            if rolling:
                self.push(self.now + t0, self._pause, [r], dur, recompute)
                t0 += dur
            else:
                groups.setdefault(dur, []).append(r)
        for dur, reps in groups.items():
            self._pause(reps, dur, recompute)


# ============================================================================= deploy policies

@dataclass(frozen=True)
class Phased:
    """Time-sharing: all cards serve until a batch has finished, then all cards train on it.

    resident:   training state and inference weights both stay on the cards. Near-free switch,
                less room for KV cache.
    oversample: admit (1 + oversample) * B episodes per phase and cut the phase when B are done.
    carry:      at the cut, suspend the unfinished episodes and resume them next phase (partial
                rollouts) instead of discarding them. They pay a KV rebuild and carry lag.
    """
    resident: bool = False
    oversample: float = 0.0
    carry: bool = False

    @property
    def name(self):
        n = "time-shared"
        if self.resident:
            n += ", resident"
        if self.oversample:
            n += f", cut +{self.oversample:.0%}" + (" carry" if self.carry else " discard")
        return n

    def check(self, s: System):
        res = s.m.train_state_bytes / s.N / GB if self.resident else 0.0
        ch = s.chunks(0, res)
        if not ch:
            return "inference weights and KV cache do not fit"
        extra = max(s.m.infer_bytes / rp.k / GB for _, rp in ch) if self.resident else 0.0
        return None if s.train_ok(s.N, extra) else "training state does not fit"

    def start(self, eng):
        eng.ps = dict(base=0, cutting=False)
        eng.set_roles(0, resident=self.resident)

    def want(self, eng):
        n = round(eng.w.episodes_per_update * (1 + self.oversample))
        return not eng.ps["cutting"] and eng.admitted - eng.ps["base"] < n

    def on_episode(self, eng):
        if not eng.ps["cutting"] and len(eng.buffer) >= eng.w.episodes_per_update:
            eng.ps["cutting"] = True
            eng.set_roles(eng.s.N, carry=self.carry, resident=self.resident, then=Engine.train)

    def on_train_done(self, eng):
        eng.ps.update(base=eng.admitted, cutting=False)
        eng.set_roles(0, resident=self.resident)


@dataclass(frozen=True)
class Split:
    """Fixed split: n_train cards train continuously, the rest serve continuously.

    max_outstanding: admission stops when finished-but-unused plus in-flight episodes reach this
                     many batches. This bounds lag when generation is over-provisioned.
    rolling / recompute: see Engine.sync.
    """
    n_train: int
    max_outstanding: float = 2.0
    rolling: bool = False
    recompute: bool = False

    @property
    def name(self):
        n = f"split {self.n_train} train"
        if self.max_outstanding != 2.0:
            n += f", bound {self.max_outstanding:g}"
        return n + (", rolling sync" if self.rolling else "") + (", KV rebuild" if self.recompute else "")

    def check(self, s: System):
        if not s.train_ok(self.n_train):
            return "training state does not fit"
        return None if s.chunks(self.n_train) else "no room for an inference replica"

    def start(self, eng):
        eng.set_roles(self.n_train)

    def want(self, eng):
        out = len(eng.buffer) + eng.inflight + len(eng.suspended)
        return out < self.max_outstanding * eng.w.episodes_per_update

    def _try(self, eng):
        if not eng.training and len(eng.buffer) >= eng.w.episodes_per_update:
            eng.train()
            eng.admit()

    def on_episode(self, eng):
        self._try(eng)

    def on_train_done(self, eng):
        eng.sync(self.rolling, self.recompute)
        self._try(eng)


@dataclass(frozen=True)
class Dynamic(Split):
    """Split whose boundary moves. After each update: if the trainer waited for data, give a
    replica's worth of cards to serving; if serving was held back by the lag bound while a batch
    was already waiting, give cards to training. Moves pay the role-switch cost and suspend the
    episodes on the cards that move."""
    n_train: int = 0          # 0 = start near half the fleet
    dwell: int = 3            # updates between moves
    thresh: float = 0.10      # fraction of the update period that counts as starved / blocked

    @property
    def name(self):
        return "dynamic split" + (", KV rebuild" if self.recompute else "")

    def _n0(self, s):
        if self.n_train:
            return self.n_train
        cands = [n for n in range(1, s.N) if s.train_ok(n) and s.chunks(n)]
        return min(cands, key=lambda n: abs(n - s.N / 2)) if cands else 0

    def check(self, s: System):
        return None if self._n0(s) else "no feasible split"

    def start(self, eng):
        eng.ps = dict(n=self._n0(eng.s), since=0)
        eng.set_roles(eng.ps["n"])

    def on_train_done(self, eng):
        s, st = eng.s, eng.ps
        st["since"] += 1
        blocked = eng.take_blocked()
        if st["since"] >= self.dwell and eng.last_period > 0:
            n, new = st["n"], None
            if eng.starved_last > self.thresh * eng.last_period:
                k = (s.replica_perf(s.order[n - 1][0]) or RepPerf(1, 0, 0, 0, 0, 0)).k
                if s.train_ok(n - k):
                    new = n - k
            elif blocked > self.thresh * eng.last_period:
                k = (s.replica_perf(s.order[n][0]) or RepPerf(1, 0, 0, 0, 0, 0)).k
                if n + k < s.N and s.chunks(n + k):
                    new = n + k
            if new is not None:
                st.update(n=new, since=0)
                eng.set_roles(new, carry=True)
        eng.sync(self.rolling, self.recompute)
        self._try(eng)


# ============================================================================= output

@dataclass
class Result:
    name: str
    feasible: bool = True
    note: str = ""
    wall_s: float = math.nan      # time to the target
    updates: int = 0              # raw updates it took
    lag_mean: float = 0.0
    lag_p95: float = 0.0
    lag_max: float = 0.0
    eff: float = 1.0              # mean value of an update relative to a fresh one
    length_bias: float = 1.0      # mean tokens of consumed episodes / mean of launched ones
    waste: float = 0.0            # discarded or rebuilt tokens / tokens trained on
    shares: tuple = (0, 0, 0, 0)  # card-time fractions: generate, train, switch+sync, idle
    wall_sd: float = 0.0
    roles: tuple = ()

    HEADER = (f"{'deploy':<36} {'to target':>9} {'+-':>5} {'updates':>7} {'s/upd':>7}  "
              f"{'lag':>5} {'p95':>4} {'value':>5} {'len':>5} {'waste':>5}   "
              f"{'gen':>3} {'trn':>3} {'sw':>3} {'idle':>4}  note")

    def row(self):
        if not self.feasible:
            return f"{self.name:<36} infeasible: {self.note}"
        g, t, sw, idle = (100 * x for x in self.shares)
        return (f"{self.name:<36} {self.wall_s/3600:7.2f} h {self.wall_sd/3600:5.2f} {self.updates:7.0f} "
                f"{self.wall_s/self.updates:7.1f}  {self.lag_mean:5.2f} {self.lag_p95:4.1f} {self.eff:5.2f} "
                f"{self.length_bias:5.2f} {100*self.waste:4.0f}%   {g:3.0f} {t:3.0f} {sw:3.0f} {idle:4.0f}  {self.note}")


def run(s: System, policy, seed=0) -> Result:
    why = policy.check(s)
    if why:
        return Result(policy.name, False, why)
    eng = Engine(s, policy, seed).run()
    wall = eng.end
    gen = eng.gen_cs
    for r in eng.reps:
        eng._advance(r)
        gen += r.rp.k * r.util
    tot = s.N * wall
    lags = np.asarray(eng.lags)
    ns = sorted({n for _, n in eng.roles_log})
    note = f"train cards {ns[0]}..{ns[-1]}" if len(ns) > 1 and not isinstance(policy, Phased) else ""
    return Result(
        policy.name, wall_s=wall, updates=eng.updates,
        lag_mean=float(lags.mean()), lag_p95=float(np.quantile(lags, .95)), lag_max=float(lags.max()),
        eff=float(np.mean(eng.effs)),
        length_bias=(eng.tok_used / (eng.updates * s.w.episodes_per_update)) / (eng.launch_tok / eng.launch_n),
        waste=(eng.tok_wasted + eng.tok_rebuild) / eng.tok_used,
        shares=(gen / tot, eng.train_cs / tot, eng.switch_cs / tot, 1 - (gen + eng.train_cs + eng.switch_cs) / tot),
        note=note, roles=tuple(eng.roles_log))


def _run(args):
    return run(*args)


def _mean(rs):
    if not rs[0].feasible:
        return rs[0]
    f = lambda k: float(np.mean([getattr(r, k) for r in rs]))
    return Result(rs[0].name, note=rs[0].note, wall_s=f("wall_s"), updates=f("updates"), lag_mean=f("lag_mean"),
                  lag_p95=f("lag_p95"), lag_max=max(r.lag_max for r in rs), eff=f("eff"),
                  length_bias=f("length_bias"), waste=f("waste"),
                  shares=tuple(np.mean([r.shares for r in rs], 0)),
                  wall_sd=float(np.std([r.wall_s for r in rs])), roles=rs[0].roles)


def run_many(s: System, policies, seeds=5, pool=None):
    """Each policy over several seeds, in parallel. Returns one averaged Result per policy; wall_sd is the spread."""
    jobs = [(s, p, seed) for p in policies for seed in range(seeds)]
    if pool is None:
        with ProcessPoolExecutor() as ex:
            out = list(ex.map(_run, jobs))
    else:
        out = list(pool.map(_run, jobs))
    return [_mean(out[i * seeds:(i + 1) * seeds]) for i in range(len(policies))]


def best_split(s: System, seeds=2, pool=None, **kw):
    """Sweep the fixed split ratio. Returns (best n_train, {n_train: hours})."""
    ns = [n for n in range(1, s.N) if Split(n).check(s) is None]
    if not ns:
        return None, {}
    res = run_many(s, [Split(n, **kw) for n in ns], seeds, pool)
    hours = {n: r.wall_s / 3600 for n, r in zip(ns, res)}
    return min(hours, key=hours.get), hours


def evaluate(cl, model, workload, seeds=5, verbose=True, **kw):
    """Run the standard set of deploy policies on one set of inputs and print the comparison."""
    s = make_simulator(cl, model, workload, **kw)
    with ProcessPoolExecutor() as pool:
        n, hours = best_split(s, pool=pool)
        pols = [Phased(), Phased(resident=True), Phased(oversample=0.25), Phased(oversample=0.25, carry=True)]
        if n is not None:
            pols += [Split(n), Split(n, recompute=True), Split(n, rolling=True, recompute=True),
                     Split(n, max_outstanding=1.0), Dynamic(), Dynamic(recompute=True)]
        res = run_many(s, pols, seeds, pool)
    if verbose:
        print(s.describe())
        if hours:
            print("split sweep, hours by training cards: " + "  ".join(f"{k}:{v:.2f}" for k, v in hours.items()))
        print(Result.HEADER)
        for r in res:
            print(r.row())
        print()
    return res
