#!/usr/bin/env python3
"""First contact with the inference engine. Starts SGLang on one card with a small model and
tries, one at a time, the calls the testbed will be built on:

    start and stop     how long they take, how much card memory the server holds
    generate           fixed-length output; the token counts and weight version it reports
    prefix cache       whether a repeated prompt reuses its cache
    weight sync        reload the weights while a request is in flight, under each way of
                       treating that request: leave it, pause in place, retract, abort
    memory release     hand the card's memory back without stopping the server, then retake it

    python3 testbed/smoke.py                      downloads Qwen/Qwen2.5-0.5B-Instruct, about 1 GB
    python3 testbed/smoke.py --model-path /path   use a model already on disk
    python3 testbed/smoke.py --card 3

Standard library only, Python 3.8+. Takes a few minutes and uses one card. Prints one line per
step as it goes and writes testbed/out/smoke.json and testbed/out/smoke-server.log. A step that
fails is recorded and the later steps still run. These are checks of what the calls do, on a
small model; they are not measurements for the simulator.
"""
import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "out")
LOG = os.path.join(OUTDIR, "smoke-server.log")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # local calls never go through a proxy
PROMPT = "The following is a long and careful description of how a river reaches the sea. " * 12
S = {"base": None, "key": None}
STEPS = []


# ----------------------------------------------------------------------------- plumbing

def call(method, path, body=None, admin=False, timeout=300):
    """One HTTP call. Returns {"status", "json", "s"}; status is None if nothing answered."""
    data = None
    if method == "POST":
        data = json.dumps({} if body is None else body).encode()
    headers = {"Content-Type": "application/json"}
    if admin:
        headers["Authorization"] = "Bearer " + S["key"]
    t = time.time()
    try:
        req = urllib.request.Request(S["base"] + path, data=data, headers=headers, method=method)
        with OPENER.open(req, timeout=timeout) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
        e.close()
    except Exception as e:
        return {"status": None, "json": "%s: %s" % (type(e).__name__, str(e)[:160]), "s": round(time.time() - t, 3)}
    try:
        js = json.loads(raw)
    except ValueError:
        js = raw[:600].decode("utf-8", "replace")
    return {"status": status, "json": js, "s": round(time.time() - t, 3)}


def short(v, n=140):
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True)
    return s if len(s) <= n else s[:n - 3] + "..."


def slim(d, n=600):
    """Drop bulky values so the output file stays readable."""
    if not isinstance(d, dict):
        return d
    return {k: (v if len(json.dumps(v)) <= n else "<%d characters>" % len(json.dumps(v))) for k, v in d.items()}


def brief(r):
    js = r["json"]
    extra = ""
    if isinstance(js, dict) and js.get("success") is False:
        extra = " success=false " + short(js.get("message", ""), 80)
    elif r["status"] != 200:
        extra = " " + short(js, 80)
    return "%s in %.2f s%s" % (r["status"], r["s"], extra)


def good(r):
    return r["status"] == 200 and not (isinstance(r["json"], dict) and r["json"].get("success") is False)


def step(name, fn):
    t = time.time()
    try:
        ok, note, detail = fn()
    except Exception as e:
        ok, note, detail = False, "%s: %s" % (type(e).__name__, str(e)[:300]), None
    rec = {"step": name, "ok": bool(ok), "s": round(time.time() - t, 2), "note": note, "detail": detail}
    STEPS.append(rec)
    print("%-26s %-6s %7.2f s  %s" % (name, "ok" if ok else "FAILED", rec["s"], note), flush=True)
    return rec


def card_mib(card):
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(card)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        return int(p.stdout.strip().splitlines()[0])
    except Exception:
        return None


def log_tail(n=40):
    try:
        with open(LOG, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------- model and server

def download(model_id):
    tries = [("modelscope", "from modelscope import snapshot_download", {}),
             ("hf mirror", "from huggingface_hub import snapshot_download",
              {"HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")})]
    last = ""
    for name, imp, env in tries:
        code = "%s\nprint('PATH=' + snapshot_download(%r))" % (imp, model_id)
        p = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=3600, env=dict(os.environ, **env))
        for ln in p.stdout.splitlines():
            if ln.startswith("PATH="):
                return ln[5:], name
        last = p.stdout[-300:]
    raise RuntimeError("download failed: " + last)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(30)
            return
        except subprocess.TimeoutExpired:
            pass


def start(args, path):
    """Try the launch command in up to three forms. Returns (process, form used)."""
    port = free_port()
    S["base"], S["key"] = "http://127.0.0.1:%d" % port, secrets.token_hex(8)
    if args.server_cmd:
        forms = [("given command", args.server_cmd.format(model=path, port=port, key=S["key"]).split())]
    else:
        base = [sys.executable, "-m", "sglang.launch_server", "--model-path", path, "--host", "127.0.0.1",
                "--port", str(port), "--log-level", "info"] + args.server_arg
        key = ["--admin-api-key", S["key"]]
        forms = [("with memory saver", base + key + ["--enable-memory-saver"]),
                 ("without memory saver", base + key),
                 ("admin key from the environment only", base)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.card), SGLANG_ADMIN_API_KEY=S["key"])
    why = []
    for name, cmd in forms:
        with open(LOG, "a") as log:
            log.write("\n===== launch, %s: %s\n" % (name, " ".join(cmd).replace(S["key"], "<key>")))
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        t = time.time()
        while time.time() - t < args.start_timeout:
            if proc.poll() is not None:
                why.append("%s: exited with code %s after %.0f s" % (name, proc.returncode, time.time() - t))
                break
            if call("GET", "/health", timeout=5)["status"] == 200:
                return proc, name, why
            time.sleep(1)
        else:
            why.append("%s: not healthy after %d s" % (name, args.start_timeout))
            stop(proc)
    raise RuntimeError("; ".join(why))


# ----------------------------------------------------------------------------- the calls under test

def generate(text, n, timeout=900):
    r = call("POST", "/generate", {"text": text, "sampling_params": {
        "max_new_tokens": n, "ignore_eos": True, "temperature": 0.0}}, timeout=timeout)
    js = r["json"][0] if isinstance(r["json"], list) and r["json"] else r["json"]
    return r, (js.get("meta_info") or {}) if isinstance(js, dict) else {}


def versions(meta):
    """Weight versions a request reports, compactly: spans become name[start:end]."""
    out = []
    for k, v in meta.items():
        if "version" not in k:
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) and "version" in x for x in v):
            v = " ".join("%s[%s:%s]" % (x["version"], x.get("start"), x.get("end")) for x in v)
        out.append("%s=%s" % (k, short(v, 200)))
    return ", ".join(out) or "none reported"


def finish(meta):
    f = meta.get("finish_reason")
    return f.get("type") if isinstance(f, dict) else f


def t_generate(n):
    r, meta = generate(PROMPT, n)
    note = "%s, prompt %s, output %s, cached %s, finish %s, versions %s" % (
        r["status"], meta.get("prompt_tokens"), meta.get("completion_tokens"), meta.get("cached_tokens"),
        finish(meta), versions(meta))
    return r["status"] == 200 and meta.get("completion_tokens") == n, note, {"meta_info": slim(meta), "s": r["s"]}


def t_load(c, n):
    out = [None] * c

    def one(i):
        out[i] = generate("%d. %s" % (i, PROMPT), n)

    t = time.time()
    ths = [threading.Thread(target=one, args=(i,)) for i in range(c)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.time() - t
    okn = sum(1 for r, _ in out if r["status"] == 200)
    toks = sum(m.get("completion_tokens") or 0 for _, m in out)
    bad = next((r for r, _ in out if r["status"] != 200), None)
    note = "%d of %d requests ok, %d output tokens, %.0f tokens/s in total" % (okn, c, toks, toks / wall)
    if bad:
        note += "; first failure: %s %s" % (bad["status"], short(bad["json"], 120))
    return okn == c, note, {"concurrency": c, "wall_s": round(wall, 3), "output_tokens": toks}


def t_sync(mode, path, n):
    """Reload the weights from disk one second into a long request."""
    box = {}

    def bg():
        r, meta = generate("[%s] %s" % (mode, PROMPT), n)
        box.update(r=r, meta=meta, end=time.time())

    th = threading.Thread(target=bg)
    th.start()
    time.sleep(1.0)
    seq = []
    if mode != "none":
        seq.append(("pause", call("POST", "/pause_generation", {"mode": mode}, admin=True)))
    body = {"model_path": path, "weight_version": "smoke-" + mode, "flush_cache": mode != "in_place"}
    up = call("POST", "/update_weights_from_disk", body, admin=True, timeout=900)
    if up["status"] == 400:      # some builds want the update inside a session
        seq.append(("update outside a session", up))
        seq.append(("begin", call("POST", "/begin_weight_update", {"selector": "all"}, admin=True)))
        up = call("POST", "/update_weights_from_disk", body, admin=True, timeout=900)
        seq.append(("update", up))
        seq.append(("end", call("POST", "/end_weight_update", {}, admin=True)))
    else:
        seq.append(("update", up))
    up_end = time.time()
    if mode != "none":
        seq.append(("continue", call("POST", "/continue_generation", {}, admin=True)))
    th.join(900)
    if th.is_alive():            # do not leave the server paused for the next step
        seq.append(("continue again", call("POST", "/continue_generation", {}, admin=True)))
        th.join(120)
    info = call("GET", "/get_model_info")
    now = info["json"].get("weight_version") if isinstance(info["json"], dict) else None
    parts = ["%s %s" % (k, brief(r)) for k, r in seq]
    if "r" in box:
        m = box["meta"]
        parts.append("request: %s, %s of %d tokens, finish %s, versions %s, ended %.1f s %s the update" % (
            box["r"]["status"], m.get("completion_tokens"), n, finish(m),
            versions(m), abs(box["end"] - up_end), "after" if box["end"] > up_end else "before"))
    else:
        parts.append("request: never returned")
    parts.append("server version now %s" % now)
    detail = {"calls": [{"call": k, "status": r["status"], "s": r["s"], "json": slim(r["json"])} for k, r in seq],
              "request": {"status": box["r"]["status"], "s": box["r"]["s"], "meta_info": slim(box["meta"])} if "r" in box else None,
              "request_end_minus_update_end_s": round(box["end"] - up_end, 3) if "r" in box else None,
              "weight_version_after": now}
    return good(up), " | ".join(parts), detail


def t_memory(path, card, what):
    before = card_mib(card)
    r = call("POST", "/%s_memory_occupation" % what, {}, admin=True, timeout=600)
    return good(r), "%s, card memory %s -> %s MiB" % (brief(r), before, card_mib(card)), {"json": slim(r["json"])}


def t_info(path, keys):
    r = call("GET", path)
    js = r["json"]
    found = {}
    if isinstance(js, dict):
        flat = dict(js)
        for v in js.values():
            if isinstance(v, dict):
                flat.update(v)
        found = {k: flat[k] for k in keys if k in flat}
    return r["status"] == 200, "%s, %s" % (r["status"], short(found, 400)), {"json": slim(js, 4000)}


# ----------------------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="model id to download")
    ap.add_argument("--model-path", help="a model folder already on disk; skips the download")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--long-tokens", type=int, default=3000, help="length of the request that a weight sync interrupts")
    ap.add_argument("--start-timeout", type=int, default=900)
    ap.add_argument("--server-arg", action="append", default=[], help="extra argument for the server; repeatable")
    ap.add_argument("--server-cmd", help="replace the launch command; {model}, {port} and {key} are filled in")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    open(LOG, "w").close()
    box = {}

    def do_model():
        if args.model_path:
            box["path"] = args.model_path
            return os.path.isdir(args.model_path), "on disk", None
        box["path"], src = download(args.model)
        return True, "%s from %s" % (args.model, src), None

    def do_start():
        before = card_mib(args.card)
        box["proc"], form, why = start(args, box["path"])
        return True, "%s; card memory %s -> %s MiB%s" % (form, before, card_mib(args.card),
                                                         "; earlier tries: " + "; ".join(why) if why else ""), {"form": form, "failed": why}

    def do_stop():
        stop(box.get("proc"))
        time.sleep(2)
        return box["proc"].poll() is not None, "card memory now %s MiB" % card_mib(args.card), None

    try:
        if not step("model", do_model)["ok"] or not step("start server", do_start)["ok"]:
            print("\nlast lines of the server log:\n" + log_tail())
            return 1
        step("health_generate", lambda: (lambda r: (r["status"] == 200, brief(r), None))(call("GET", "/health_generate", timeout=300)))
        step("get_model_info", lambda: t_info("/get_model_info", ["weight_version", "model_path", "is_generation"]))
        step("server_info", lambda: t_info("/server_info", [
            "version", "max_total_num_tokens", "max_req_input_len", "max_running_requests", "mem_fraction_static",
            "chunked_prefill_size", "max_prefill_tokens", "schedule_policy", "disable_radix_cache", "cuda_graph_max_bs",
            "enable_memory_saver", "dtype", "tp_size", "attention_backend"]))
        step("generate 64", lambda: t_generate(64))
        step("same prompt again", lambda: t_generate(64))
        for c in (1, 16, 128):
            step("load, %d streams" % c, lambda c=c: t_load(c, 128))
        for mode in ("none", "in_place", "retract", "abort"):
            step("sync, request %s" % ("left alone" if mode == "none" else mode),
                 lambda mode=mode: t_sync(mode, box["path"], args.long_tokens))
        step("generate after syncs", lambda: t_generate(16))
        step("release memory", lambda: t_memory(box["path"], args.card, "release"))
        step("resume memory", lambda: t_memory(box["path"], args.card, "resume"))
        step("generate after resume", lambda: t_generate(16))
    finally:
        if box.get("proc") is not None:
            step("stop server", do_stop)
        bad = [s["step"] for s in STEPS if not s["ok"]]
        out = {"smoke_version": 1, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "model": args.model_path or args.model, "card": args.card, "steps": STEPS, "failed": bad}
        if bad:
            out["server_log_tail"] = log_tail(80)
        with open(os.path.join(OUTDIR, "smoke.json"), "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print("\n%d of %d steps ok%s" % (len(STEPS) - len(bad), len(STEPS), "; failed: " + ", ".join(bad) if bad else ""))
        print("wrote testbed/out/smoke.json and testbed/out/smoke-server.log")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
