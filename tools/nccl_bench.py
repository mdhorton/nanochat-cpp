#!/usr/bin/env python3

# nccl-tests sweep: transport (P2P level), channels, protocol; optional (--tests): per-pair links, leave-one-out rings
# (slow: n*(n-1)/2 + n runs, most of the time on 8 GPUs).
# build first with tools/build_nccl_tests.sh. raw logs + summary.json go to --out.

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def gpu_count():
    return len([l for l in sh(["nvidia-smi", "-L"]).splitlines() if l.startswith("GPU")])


def busy_gpus(gpus):
    out = sh(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"])
    util = {int(i): int(u) for i, u in (l.split(",") for l in out.strip().splitlines())}
    return [g for g in gpus if util.get(g, 0) > 10]


def pcie_links(gpus, stop, peak):
    # link gen/width drop at idle, so keep the max seen while a test runs.
    q = "--query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max"
    while not stop.is_set():
        for l in sh(["nvidia-smi", q, "--format=csv,noheader"]).strip().splitlines():
            i, bus, gen, width, gmax, wmax = [x.strip() for x in l.split(",")]
            if int(i) in gpus:
                cur = peak.get(i, {"bus": bus, "gen": 0, "width": 0, "gen_max": gmax, "width_max": wmax})
                cur["gen"] = max(cur["gen"], int(gen))
                cur["width"] = max(cur["width"], int(width))
                peak[i] = cur
        stop.wait(0.3)


def parse(text, nccl_log=""):
    # size count type redop root | oop time algbw busbw #wrong | ip time algbw busbw #wrong
    half = r"\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+(\S+)"  # time (algbw) (busbw) (#wrong)
    row = re.compile(r"^\s*(\d+)\s+\d+\s+[a-z]\w*\s+\w+\s+-?\d+" + half + half + r"\s*$")
    rows = []
    for l in text.splitlines():
        if m := row.match(l):  # NCCL INFO lines can start with digits too
            size, _, busbw, w1, _, busbw_ip, w2 = m.groups()
            wrong = sum(int(w) if w.isdigit() else 0 for w in (w1, w2))
            rows.append({"bytes": int(size), "busbw": float(busbw), "busbw_ip": float(busbw_ip), "wrong": wrong})
    avg = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", text)
    oob = re.search(r"Out of bounds values\s*:\s*(\d+)", text)
    return {
        "rows": rows,
        "avg_busbw": float(avg.group(1)) if avg else None,
        "max_busbw": max((r["busbw"] for r in rows), default=None),
        "wrong": sum(r["wrong"] for r in rows) + (int(oob.group(1)) if oob else 0),
        "transport": sorted(set(re.findall(r" via (\S+)", nccl_log))),
        "nccl": (re.search(r"NCCL version (\S+)", nccl_log) or [None, None])[1],
    }


class Bench:
    def __init__(self, a):
        self.a, self.results = a, []
        self.out = Path(a.out) / time.strftime("%Y%m%d-%H%M%S")
        if not a.dry_run:
            self.out.mkdir(parents=True)

    def run(self, group, name, test, gpus, env=None, unset=(), sizes=None, links=None):
        b, e, f = sizes or (self.a.min_bytes, self.a.max_bytes, "2")
        cmd = [str(Path(self.a.build) / f"{test}_perf"), "-b", b, "-e", e, "-f", f, "-g", str(len(gpus)),
               "-n", str(self.a.iters), "-w", "5"]
        stem = f"{group}-{name}".replace("/", "_").replace(" ", "_")
        envd = {k: v for k, v in os.environ.items() if k not in unset}
        # PCI order so indices match nvidia-smi / topo -m. NCCL's log goes to its own file: on stdout it can split
        # result rows (sendrecv connects mid-test).
        envd.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "NCCL_DEBUG": "INFO",
                     "NCCL_DEBUG_SUBSYS": "INIT,GRAPH", "NCCL_DEBUG_FILE": str(self.out / f"{stem}.nccl.%p"), **(env or {})})
        tag = " ".join([f"{k}={v}" for k, v in (env or {}).items()] + [f"-{k}" for k in unset])
        print(f"[{group}] {name}: {test} gpus={','.join(map(str, gpus))} {tag}", flush=True)
        if self.a.dry_run:
            return None
        stop, t = threading.Event(), None
        if links is not None:
            t = threading.Thread(target=pcie_links, args=(gpus, stop, links), daemon=True)
            t.start()
        t0 = time.time()
        try:
            p = subprocess.run(cmd, env=envd, capture_output=True, text=True, timeout=self.a.timeout)
            text, status = p.stdout + p.stderr, "ok" if p.returncode == 0 else f"exit {p.returncode}"
        except subprocess.TimeoutExpired as ex:
            out = ex.stdout or b""  # bytes even with text=True
            text, status = out.decode() if isinstance(out, bytes) else out, "timeout"
        stop.set()
        if t:
            t.join()
        nccl_log = "".join(f.read_text(errors="replace") for f in sorted(self.out.glob(f"{stem}.nccl.*")))
        (self.out / f"{stem}.log").write_text(" ".join(cmd) + "\n" + tag + "\n\n" + text)
        r = {"group": group, "name": name, "test": test, "gpus": gpus, "env": env or {}, "unset": list(unset),
             "status": status, "secs": round(time.time() - t0, 1), **parse(text, nccl_log)}
        if not self.a.keep_rows:
            r.pop("rows")
        fmt = lambda v: f"{v:.2f}" if v is not None else "-"
        bw = f"avg {fmt(r['avg_busbw'])} max {fmt(r['max_busbw'])} GB/s"
        bad = f" WRONG={r['wrong']}" if r["wrong"] else ""
        print(f"    {status} {r['secs']}s {bw}{bad} via {','.join(r['transport']) or '?'}", flush=True)
        self.results.append(r)
        return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", default=str(ROOT / "external/nccl-tests/build"), help="nccl-tests build dir")
    p.add_argument("--gpus", help="comma list (default all)")
    p.add_argument("--p2p-levels", default="default,SYS", help="NCCL_P2P_LEVEL values; 'default' = unset")
    p.add_argument("--channels", default="2,4,8,16", help="NCCL_{MIN,MAX}_NCHANNELS sweep ('' skips)")
    p.add_argument("--min-bytes", default="4M")
    p.add_argument("--max-bytes", default="128M")
    p.add_argument("--pair-bytes", default="64M", help="sendrecv size for the per-pair test")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--timeout", type=int, default=20, help="per-test seconds (P2P can hang)")
    p.add_argument("--tests", default="transport,channels,proto",
                   help="comma list of groups: transport,pairs,subsets,channels,proto; 'all' = every group")
    p.add_argument("--out", default=str(ROOT / "cache/nccl"))
    p.add_argument("--keep-rows", action="store_true", help="keep per-size rows in summary.json")
    p.add_argument("--force", action="store_true", help="run even if the GPUs look busy")
    p.add_argument("--dry-run", action="store_true", help="print the tests only")
    a = p.parse_args()

    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else list(range(gpu_count()))
    groups = {"transport", "pairs", "subsets", "channels", "proto"}
    tests = groups if a.tests == "all" else set(filter(None, a.tests.split(",")))
    if tests - groups:
        sys.exit(f"unknown --tests: {','.join(sorted(tests - groups))}")
    if len(gpus) < 2:
        sys.exit("need >= 2 GPUs")
    if not a.dry_run:
        if not (Path(a.build) / "all_gather_perf").exists():
            sys.exit(f"no nccl-tests in {a.build}; run tools/build_nccl_tests.sh")
        if (busy := busy_gpus(gpus)) and not a.force:
            sys.exit(f"GPUs {busy} busy; --force to run anyway")

    b = Bench(a)
    if not a.dry_run:
        b.out.joinpath("topo.txt").write_text(sh(["nvidia-smi", "topo", "-m"]))
        print(sh(["nvidia-smi", "topo", "-m"]))
    print("env:", " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("NCCL_") and k != "NCCL_VERSION") or "-")

    # 1. transport: each P2P level x collective, all GPUs.
    links = {}
    levels = a.p2p_levels.split(",")
    best = {}
    if "transport" in tests:
        for lvl in levels:
            env, unset = ({}, ("NCCL_P2P_LEVEL",)) if lvl == "default" else ({"NCCL_P2P_LEVEL": lvl}, ())
            for test in ("all_gather", "reduce_scatter", "all_reduce"):
                r = b.run("transport", f"{test}-{lvl}", test, gpus, env, unset, links=links)
                if r and test == "all_gather" and not r["wrong"] and r["avg_busbw"]:
                    best[lvl] = r["avg_busbw"]
    lvl = max(best, key=best.get) if best else levels[-1]
    base_env, base_unset = ({}, ("NCCL_P2P_LEVEL",)) if lvl == "default" else ({"NCCL_P2P_LEVEL": lvl}, ())
    print(f"using P2P level {lvl} for the remaining tests")

    # 2. pairs: sendrecv between every GPU pair (finds a slow card or link).
    if "pairs" in tests:
        for i, j in itertools.combinations(gpus, 2):
            b.run("pairs", f"{i}-{j}", "sendrecv", [i, j], base_env, base_unset, sizes=(a.pair_bytes, a.pair_bytes, "2"))

    # 3. leave-one-out rings (does one GPU drag the ring?).
    if "subsets" in tests and len(gpus) >= 3:
        for g in gpus:
            sub = [x for x in gpus if x != g]
            b.run("subsets", f"without-{g}", "all_gather", sub, base_env, base_unset)

    # 4. channel count.
    if "channels" in tests:
        for ch in filter(None, a.channels.split(",")):
            env = {**base_env, "NCCL_MIN_NCHANNELS": ch, "NCCL_MAX_NCHANNELS": ch}
            b.run("channels", f"ch{ch}", "all_gather", gpus, env, base_unset)

    # 5. protocol: Simple (pixi default) vs NCCL's own choice vs LL128.
    if "proto" in tests:
        for proto in ("Simple", "default", "LL128"):
            unset = base_unset + (("NCCL_PROTO",) if proto == "default" else ())
            env = base_env if proto == "default" else {**base_env, "NCCL_PROTO": proto}
            b.run("proto", proto, "all_gather", gpus, env, unset)

    if a.dry_run:
        return

    # summary
    print(f"\n{'group':<10} {'test':<22} {'avg':>7} {'max':>7}  status  via")
    for r in b.results:
        avg = f"{r['avg_busbw']:7.2f}" if r["avg_busbw"] else "      -"
        mx = f"{r['max_busbw']:7.2f}" if r["max_busbw"] else "      -"
        st = r["status"] + (f" WRONG={r['wrong']}" if r["wrong"] else "")
        print(f"{r['group']:<10} {r['name']:<22} {avg} {mx}  {st:<6}  {','.join(r['transport'])}")
    if links:
        print("\npcie (peak during the transport tests):")
        for i, l in sorted(links.items()):
            print(f"  gpu{i} {l['bus']}: gen {l['gen']}/{l['gen_max']} x{l['width']}/{l['width_max']}")
    nccl = next((r["nccl"] for r in b.results if r["nccl"]), None)
    summary = {"nccl": nccl, "gpus": gpus, "p2p_level": lvl, "pcie": links, "results": b.results,
               "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and k != "NCCL_VERSION"}}
    (b.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nNCCL {nccl}; busbw in GB/s; logs + summary.json in {b.out}")


if __name__ == "__main__":
    main()
