#!/usr/bin/env python3
"""Describe the machine the testbed will run on. Standard library only, Python 3.8+.

    python3 testbed/probe.py                        facts only; allocates nothing on the cards
    python3 testbed/probe.py --bench                also a few seconds of work on each idle card (needs torch)
    python3 testbed/probe.py --models-dir /data/m   also list model folders found under that path

Writes testbed/out/probe.json and prints a summary.

The file holds hardware, software versions, free disk, which download sites answer, and the
names of models already on disk. It leaves out host name, user name, process names, card
serial numbers and the values of proxy variables. The home directory is written as ~.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "probe.json")
GIB = 2.0 ** 30


def tidy(path):
    return path.replace(HOME, "~", 1) if len(HOME) > 1 and path.startswith(HOME) else path


def sh(cmd, timeout=30):
    """Run a command. Returns (code, stdout, stderr); code is None if it is absent or hangs."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return None, "", "not found"
    except subprocess.TimeoutExpired:
        return None, "", "timed out after %d s" % timeout
    except OSError as e:
        return None, "", type(e).__name__


def num(s):
    for kind in (int, float):
        try:
            return kind(s)
        except (TypeError, ValueError):
            pass
    return s


def is_num(v):
    return isinstance(v, (int, float))


# ----------------------------------------------------------------------------- cards

GPU_FIELDS = ["index", "name", "memory.total", "memory.used", "utilization.gpu", "temperature.gpu",
              "power.draw", "power.limit", "compute_mode", "persistence_mode", "pci.bus_id", "uuid"]
GPU_FIELDS_MIN = ["index", "name", "memory.total", "memory.used", "uuid"]


def parse_gpu_csv(text, fields):
    cards = []
    for line in text.splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(fields):
            cards.append({k: num(v) for k, v in zip(fields, vals)})
    return cards


def gpus():
    out = {"cards": [], "error": None}
    err = ""
    for fields in (GPU_FIELDS, GPU_FIELDS_MIN):
        rc, so, se = sh(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
        if rc == 0:
            break
        err = se or so or "nvidia-smi failed"
    else:
        out["error"] = err[:300]
        return out
    cards = parse_gpu_csv(so, fields)

    # Other people's jobs. Inside a container this list is often empty even when the cards are
    # busy, so memory.used is the signal to trust.
    rc, so, _ = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
    busy = {}
    if rc == 0:
        for line in so.splitlines():
            parts = [v.strip() for v in line.split(",")]
            if len(parts) == 2:
                n, mib = busy.get(parts[0], (0, 0.0))
                used = num(parts[1])
                busy[parts[0]] = (n + 1, mib + (used if is_num(used) else 0.0))
    for c in cards:
        n, mib = busy.get(c.pop("uuid", None), (0, 0.0))
        c["processes"], c["processes_mib"] = n, mib
    out["cards"] = cards

    rc, so, _ = sh(["nvidia-smi"])
    if rc == 0:
        m = re.search(r"Driver Version:\s*([\w.]+)", so)
        out["driver"] = m.group(1) if m else None
        m = re.search(r"CUDA Version:\s*([\w.]+)", so)
        out["driver_cuda"] = m.group(1) if m else None
    rc, so, se = sh(["nvidia-smi", "topo", "-m"])
    out["topology"] = so[:8000] if rc == 0 else "unavailable: " + (se or so)[:200]
    rc, so, se = sh(["nvidia-smi", "nvlink", "-s"])
    out["nvlink"] = so[:12000] if rc == 0 else "unavailable: " + (se or so)[:200]
    out["mps_running"] = sh(["pgrep", "-f", "nvidia-cuda-mps"])[0] == 0
    return out


# ----------------------------------------------------------------------------- host

def fstype(path):
    best, kind = "", "?"
    try:
        real = os.path.realpath(path)
        with open("/proc/mounts") as f:
            for line in f:
                p = line.split()
                if len(p) >= 3 and (real == p[1] or real.startswith(p[1].rstrip("/") + "/")) and len(p[1]) >= len(best):
                    best, kind = p[1], p[2]
    except OSError:
        pass
    return kind


def disk(path):
    while path and not os.path.exists(path):
        path = os.path.dirname(path)
    try:
        u = shutil.disk_usage(path)
        return {"path": tidy(path), "fs": fstype(path), "total_gib": round(u.total / GIB, 1),
                "free_gib": round(u.free / GIB, 1)}
    except OSError as e:
        return {"path": tidy(path), "error": type(e).__name__}


def hf_hub_dir():
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(k):
            return os.environ[k]
    return os.path.join(os.environ.get("HF_HOME", os.path.join(HOME, ".cache", "huggingface")), "hub")


def host():
    out = {"os": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count()}
    try:
        out["cpus_usable"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        out["cpu_model"] = m.group(1).strip() if m else None
        out["sockets"] = len(set(re.findall(r"physical id\s*:\s*(\d+)", txt))) or None
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    out["ram_" + k[3:].lower() + "_gib"] = round(int(v.split()[0]) / 2 ** 20, 1)
    except (OSError, ValueError, IndexError):
        pass
    try:
        out["numa_nodes"] = len([d for d in os.listdir("/sys/devices/system/node") if re.match(r"node\d+$", d)])
    except OSError:
        pass
    # A container may cap memory below what /proc/meminfo shows.
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p) as f:
                v = f.read().strip()
            out["cgroup_mem_gib"] = None if v == "max" or int(v) > 2 ** 60 else round(int(v) / GIB, 1)
            break
        except (OSError, ValueError):
            pass
    out["disks"] = {"repo": disk(HERE), "home": disk(HOME), "tmp": disk("/tmp"), "shm": disk("/dev/shm"),
                    "hfhub": disk(hf_hub_dir())}
    return out


ENV_NAMES = ["http_proxy", "https_proxy", "all_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
             "HF_ENDPOINT", "HF_HOME", "HF_HUB_CACHE", "HF_TOKEN", "MODELSCOPE_CACHE", "PIP_INDEX_URL",
             "CONDA_PREFIX", "VIRTUAL_ENV", "SLURM_JOB_ID", "KUBERNETES_SERVICE_HOST", "LD_LIBRARY_PATH", "CUDA_HOME"]
ENV_VALUES = ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NVIDIA_VISIBLE_DEVICES"]
TOOLS = ["git", "pip", "pip3", "uv", "conda", "mamba", "docker", "podman", "apptainer", "singularity",
         "nvidia-container-cli", "sinfo", "srun", "kubectl", "tmux", "screen", "nvcc", "gcc", "cmake", "numactl",
         "ibstat", "modelscope", "huggingface-cli", "hf", "vllm"]


def environment():
    out = {"root": hasattr(os, "geteuid") and os.geteuid() == 0}
    cg = ""
    try:
        with open("/proc/1/cgroup") as f:
            cg = f.read()
    except OSError:
        pass
    out["container"] = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv") or \
        any(k in cg for k in ("docker", "kubepods", "containerd", "lxc"))
    out["tools"] = sorted(t for t in TOOLS if shutil.which(t))
    out["env_set"] = sorted(k for k in ENV_NAMES if os.environ.get(k))       # names only, never values
    out["env"] = {k: os.environ[k] for k in ENV_VALUES if k in os.environ}
    try:
        import resource
        out["open_files_limit"] = list(resource.getrlimit(resource.RLIMIT_NOFILE))
        out["locked_mem_limit"] = list(resource.getrlimit(resource.RLIMIT_MEMLOCK))
    except (ImportError, OSError, ValueError):
        pass
    rc, so, _ = sh(["nvcc", "--version"])
    m = re.search(r"release\s+([\d.]+)", so) if rc == 0 else None
    out["nvcc"] = m.group(1) if m else None
    return out


# ----------------------------------------------------------------------------- python

PKGS = ["torch", "vllm", "transformers", "tokenizers", "peft", "accelerate", "datasets", "safetensors",
        "huggingface-hub", "modelscope", "trl", "deepspeed", "bitsandbytes", "flash-attn", "xformers", "triton",
        "sglang", "openai", "aiohttp", "httpx", "fastapi", "uvicorn", "ray", "numpy", "nvidia-ml-py", "pynvml",
        "nvidia-cudnn-cu12", "nvidia-nccl-cu12", "mpi4py"]

TORCH_SNIPPET = r"""
import json, torch
d = {"version": torch.__version__, "cuda_build": torch.version.cuda,
     "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
try:
    d["cudnn"] = torch.backends.cudnn.version()
except Exception:
    d["cudnn"] = None
try:
    d["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
except Exception:
    d["nccl"] = None
print("PROBE" + json.dumps(d))
"""


def python_info():
    out = {"version": platform.python_version(), "executable": tidy(sys.executable),
           "venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("CONDA_PREFIX"))}
    pk = {}
    try:
        from importlib import metadata
        for name in PKGS:
            for alias in (name, name.replace("-", "_")):
                try:
                    pk[name] = metadata.version(alias)
                    break
                except metadata.PackageNotFoundError:
                    pk[name] = None
                except Exception as e:      # a broken install should not stop the probe
                    pk[name] = "error: " + type(e).__name__
                    break
    except ImportError:
        out["packages_error"] = "importlib.metadata missing (Python < 3.8)"
    out["packages"] = pk
    if pk.get("torch"):
        rc, so, se = sh([sys.executable, "-c", TORCH_SNIPPET], timeout=300)
        line = next((ln for ln in so.splitlines() if ln.startswith("PROBE")), None)
        out["torch"] = json.loads(line[5:]) if line else {"error": (se or so or "no output")[-300:]}
    rc, so, _ = sh([sys.executable, "-m", "pip", "--version"])
    out["pip"] = so.split(" from ")[0] if rc == 0 else None
    return out


# ----------------------------------------------------------------------------- network

SITES = [("huggingface.co", "https://huggingface.co/"),
         ("hf-mirror.com", "https://hf-mirror.com/"),
         ("modelscope.cn", "https://www.modelscope.cn/"),
         ("pypi.org", "https://pypi.org/simple/pip/"),
         ("pypi mirror (tuna)", "https://pypi.tuna.tsinghua.edu.cn/simple/pip/"),
         ("download.pytorch.org", "https://download.pytorch.org/whl/"),
         ("github.com", "https://github.com/")]


def reach(url, timeout=8):
    import urllib.error
    import urllib.request
    t = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "probe/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(512)
            return {"ok": True, "status": r.status, "s": round(time.time() - t, 2)}
    except urllib.error.HTTPError as e:      # the site answered, which is all we ask
        return {"ok": True, "status": e.code, "s": round(time.time() - t, 2)}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "s": round(time.time() - t, 2)}


def network():
    from concurrent.futures import ThreadPoolExecutor
    sites = list(SITES)
    if os.environ.get("HF_ENDPOINT"):
        sites.append(("HF_ENDPOINT (address not recorded)", os.environ["HF_ENDPOINT"]))
    with ThreadPoolExecutor(len(sites)) as ex:
        res = list(ex.map(lambda s: reach(s[1]), sites))
    return {name: r for (name, _), r in zip(sites, res)}


# ----------------------------------------------------------------------------- models on disk

def dir_size(path, budget=200000):
    """Bytes under path, without following links. Gives up after `budget` entries."""
    total, seen, stack = 0, 0, [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    seen += 1
                    if seen > budget:
                        return total, False
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
        except OSError:
            pass
    return total, True


def model_dirs(root, depth):
    """Folders under root, at most `depth` levels down, that hold a config.json."""
    out, stack = [], [(root, 0)]
    while stack and len(out) < 200:
        path, d = stack.pop()
        try:
            names = sorted(os.listdir(path))
        except OSError:
            continue
        if "config.json" in names:
            out.append(path)
            continue
        if d < depth:
            stack.extend((os.path.join(path, n), d + 1) for n in names
                         if not n.startswith(".") and os.path.isdir(os.path.join(path, n)))
    return out


def models(extra_dirs):
    found = []

    def add(name, path, where):
        size, complete = dir_size(path)
        found.append({"name": name, "where": where, "gib": round(size / GIB, 2), "size_complete": complete})

    hub = hf_hub_dir()
    try:
        for n in sorted(os.listdir(hub)):
            if n.startswith("models--"):
                add(n[len("models--"):].replace("--", "/"), os.path.join(hub, n, "blobs"), "huggingface cache")
    except OSError:
        pass
    ms = os.environ.get("MODELSCOPE_CACHE", os.path.join(HOME, ".cache", "modelscope"))
    for p in model_dirs(ms, 4):
        add(os.path.relpath(p, ms), p, "modelscope cache")
    for root in extra_dirs:
        for p in model_dirs(root, 3):
            add(os.path.relpath(p, root) if p != root else os.path.basename(p.rstrip("/")), p, tidy(root))
    return found


# ----------------------------------------------------------------------------- short benchmark

def bench(n=16384, copy_gib=4.0, xfer_gib=2.0):
    """A few seconds of work per idle card. Numbers are sanity checks, not simulator factors:
    matmul          dense 16-bit matrix product, in TFLOP/s (2 n^3 operations per product)
    copy            one buffer copied to another on the same card, in GB/s read plus written
    host_to_card    a host buffer copied to the card
    card_to_card    a buffer copied between two cards
    """
    import torch
    out = {"torch": torch.__version__, "n": n, "cards": [], "skipped": [], "card_to_card": []}
    if not torch.cuda.is_available():
        out["error"] = "torch sees no CUDA device"
        return out

    def sync(*devs):
        for d in devs:
            torch.cuda.synchronize(d)

    def best(fn, devs, reps):
        ts = []
        for _ in range(reps):
            sync(*devs)
            t = time.perf_counter()
            fn()
            sync(*devs)
            ts.append(time.perf_counter() - t)
        return min(ts)

    idle = []
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception as e:
            out["skipped"].append({"index": i, "why": type(e).__name__})
            continue
        if free < 0.9 * total:
            out["skipped"].append({"index": i, "why": "in use: %.0f of %.0f GiB free" % (free / GIB, total / GIB)})
        else:
            idle.append(i)

    host_buf, pinned = None, False
    try:
        host_buf = torch.empty(int(xfer_gib * GIB), dtype=torch.uint8)
        host_buf.fill_(1)
        try:
            host_buf = host_buf.pin_memory()
            pinned = True
        except Exception:
            pass
    except Exception as e:
        out["host_buffer_error"] = type(e).__name__
    out["host_buffer_pinned"] = pinned

    for i in idle:
        dev = torch.device("cuda", i)
        row = {"index": i, "name": torch.cuda.get_device_name(i),
               "mem_gib": round(torch.cuda.mem_get_info(i)[1] / GIB, 1)}
        for dt in ("float16", "bfloat16"):
            try:
                a = torch.randn(n, n, device=dev).to(getattr(torch, dt))
                b = torch.randn(n, n, device=dev).to(getattr(torch, dt))
                c = torch.empty_like(a)
                for _ in range(5):
                    torch.matmul(a, b, out=c)
                reps = 10
                t = best(lambda: [torch.matmul(a, b, out=c) for _ in range(reps)], [dev], 3)
                row["matmul_tflops_" + dt] = round(2.0 * n ** 3 * reps / t / 1e12, 1)
                del a, b, c
            except Exception as e:
                row["matmul_error_" + dt] = "%s: %s" % (type(e).__name__, str(e)[:120])
        try:
            x = torch.empty(int(copy_gib * GIB), dtype=torch.uint8, device=dev)
            x.fill_(1)
            y = torch.empty_like(x)
            y.copy_(x)
            row["copy_gbs"] = round(2 * x.numel() / best(lambda: y.copy_(x), [dev], 5) / 1e9, 1)
            if host_buf is not None:
                x[:host_buf.numel()].copy_(host_buf)
                t = best(lambda: x[:host_buf.numel()].copy_(host_buf), [dev], 3)
                row["host_to_card_gbs"] = round(host_buf.numel() / t / 1e9, 2)
            del x, y
        except Exception as e:
            row["copy_error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
        torch.cuda.empty_cache()
        out["cards"].append(row)

    pairs = []
    if len(idle) >= 2:
        pairs.append((idle[0], idle[1]))
    if len(idle) >= 3:
        pairs.append((idle[0], idle[-1]))
    for i, j in pairs:
        try:
            x = torch.empty(int(xfer_gib * GIB), dtype=torch.uint8, device="cuda:%d" % i)
            x.fill_(1)
            z = torch.empty(x.numel(), dtype=torch.uint8, device="cuda:%d" % j)
            z.copy_(x)
            t = best(lambda: z.copy_(x), [i, j], 3)
            out["card_to_card"].append({"from": i, "to": j, "gbs": round(x.numel() / t / 1e9, 2)})
            del x, z
        except Exception as e:
            out["card_to_card"].append({"from": i, "to": j, "error": "%s: %s" % (type(e).__name__, str(e)[:120])})
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- report

def summary(d):
    L = []
    g, h, e, p = d["gpus"], d["host"], d["environment"], d["python"]
    if g["error"]:
        L.append("cards      : none seen (%s)" % g["error"])
    else:
        kinds = {}
        for c in g["cards"]:
            key = "%s, %.0f GiB" % (c.get("name"), c["memory.total"] / 1024) \
                if is_num(c.get("memory.total")) else str(c.get("name"))
            kinds[key] = kinds.get(key, 0) + 1
        L.append("cards      : " + "; ".join("%d x %s" % (n, k) for k, n in kinds.items()))
        used = [c for c in g["cards"] if is_num(c.get("memory.used")) and c["memory.used"] > 1024]
        L.append("in use     : %d of %d cards hold more than 1 GiB" % (len(used), len(g["cards"])))
        L.append("driver     : %s, CUDA up to %s" % (g.get("driver"), g.get("driver_cuda")))
        speeds = sorted(set(re.findall(r"([\d.]+)\s*GB/s", g.get("nvlink", ""))))
        L.append("nvlink     : " + ("link speeds %s GB/s" % ", ".join(speeds) if speeds else "no link speeds reported"))
    L.append("host       : %s cpus (%s usable), %s GiB RAM (%s available), %s NUMA nodes" % (
        h.get("cpus"), h.get("cpus_usable"), h.get("ram_total_gib"), h.get("ram_available_gib"), h.get("numa_nodes")))
    for k, v in h["disks"].items():
        if "error" not in v:
            L.append("disk %-6s: %7.0f GiB free of %7.0f, %s, %s" % (k[:6], v["free_gib"], v["total_gib"], v["fs"], v["path"]))
    L.append("setting    : %s%s, tools: %s" % ("container" if e["container"] else "bare host",
                                                 ", root" if e["root"] else "", " ".join(e["tools"]) or "none"))
    L.append("python     : %s%s, %s" % (p["version"], " (venv)" if p["venv"] else "", p.get("pip") or "no pip"))
    have = ["%s %s" % (k, v) for k, v in p["packages"].items() if v]
    L.append("packages   : " + (", ".join(have) or "none of the ones looked for"))
    if "torch" in p:
        L.append("torch      : " + json.dumps(p["torch"]))
    L.append("reachable  : " + ", ".join(k for k, v in d["network"].items() if v["ok"]))
    L.append("unreachable: " + (", ".join(k for k, v in d["network"].items() if not v["ok"]) or "none"))
    ms = d["models"]
    L.append("models     : " + (", ".join("%s (%.0f GiB)" % (m["name"], m["gib"]) for m in ms[:12]) or "none found")
             + (" and %d more" % (len(ms) - 12) if len(ms) > 12 else ""))
    b = d.get("bench")
    if b:
        if b.get("error"):
            L.append("bench      : " + b["error"])
        for c in b.get("cards", []):
            L.append("bench %2d   : matmul %s / %s TFLOP/s (fp16 / bf16), copy %s GB/s, host to card %s GB/s" % (
                c["index"], c.get("matmul_tflops_float16", "-"), c.get("matmul_tflops_bfloat16", "-"),
                c.get("copy_gbs", "-"), c.get("host_to_card_gbs", "-")))
        for x in b.get("card_to_card", []):
            L.append("bench      : card %d to card %d %s" % (x["from"], x["to"], "%s GB/s" % x["gbs"] if "gbs" in x else x["error"]))
        for x in b.get("skipped", []):
            L.append("bench      : card %s skipped, %s" % (x["index"], x["why"]))
    return "\n".join(L)


def write(d):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(d, f, indent=1, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", action="store_true", help="also run a few seconds of work on each idle card")
    ap.add_argument("--models-dir", action="append", default=[], help="extra folder to search for models; repeatable")
    args = ap.parse_args()

    d = {"probe_version": 1, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for key, fn in (("gpus", gpus), ("host", host), ("environment", environment), ("python", python_info),
                    ("network", network), ("models", lambda: models(args.models_dir))):
        print("probing %s ..." % key, file=sys.stderr)
        d[key] = fn()
    write(d)        # facts are safe on disk before anything touches a card
    if args.bench:
        print("running the short benchmark ...", file=sys.stderr)
        try:
            d["bench"] = bench()
        except Exception as e:
            d["bench"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
        write(d)
    print(summary(d))
    print("\nwrote " + tidy(OUT))


if __name__ == "__main__":
    main()
