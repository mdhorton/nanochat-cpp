"""
Upload a base_train metrics file (<base_dir>/metrics/<run>/metrics-<timestamp>.jsonl) to wandb, as base_train.py logs.

The header line becomes the run config; step and eval lines are logged at wandb step = training step, with "step" also
kept as a key (base_train.py's x axis). The wandb run id is kept in <file>.wandb-id, so uploading again (or following a
resumed training run) continues the same wandb run and skips the steps it already has.

  python tools/wandb_upload.py --file cache/metrics/full-d12             # newest file in the dir
  python tools/wandb_upload.py --file <file> --follow [--pid <trainer>]  # live; base_train --wandb does this
"""
import argparse
import glob
import json
import os
import sys
import time


def resolve(path):
    if not os.path.isdir(path):
        return path
    files = sorted(glob.glob(os.path.join(path, "metrics-*.jsonl")))  # timestamps sort as strings
    if not files:
        raise SystemExit(f"no metrics-*.jsonl in {path}")
    return files[-1]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def lines(path, follow, pid, poll):
    """Yields parsed lines; when following, waits for more until an end line or the trainer (pid) exits."""
    buf, pos = "", 0
    while True:
        if os.path.getsize(path) < pos:  # rewritten by a resumed training run
            buf, pos = "", 0
        with open(path) as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        buf += chunk
        *complete, buf = buf.split("\n")
        for line in complete:
            if line.strip():
                rec = json.loads(line)
                yield rec
                if rec.get("event") == "end":
                    return
        if not follow:
            return
        if not chunk:
            if pid is not None and not alive(pid):
                if os.path.getsize(path) == pos:  # drained
                    return
                continue
            time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description="Upload a base_train metrics file to wandb")
    parser.add_argument("--file", required=True, help="metrics jsonl, or a metrics/<run> dir (newest file)")
    parser.add_argument("--project", default="nanochat", help="wandb project (base_train.py: nanochat)")
    parser.add_argument("--name", help="wandb run name (default: the header's run)")
    parser.add_argument("--id", help="wandb run id to resume (default: <file>.wandb-id, else a new run)")
    parser.add_argument("--mode", choices=["online", "offline"], help="wandb mode (default: wandb's, e.g. WANDB_MODE)")
    parser.add_argument("--every", type=int, default=1, help="log every Nth training step (base_train.py: 100)")
    parser.add_argument("--follow", action="store_true", help="keep reading new lines until the end line")
    parser.add_argument("--pid", type=int, help="with --follow: also stop once this process (the trainer) exits")
    parser.add_argument("--poll", type=float, default=1.0, help="with --follow: seconds between reads")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb_upload: wandb is not installed")

    path = resolve(args.file)
    records = lines(path, args.follow, args.pid, args.poll)
    header = next(records, None)
    if header is None or "step" in header:
        raise SystemExit(f"{path}: no header line")
    config = dict(header.get("user_config", {}))
    config.update({k: v for k, v in header.items() if k not in ("user_config", "run")})

    id_path = path + ".wandb-id"
    run_id = args.id
    if run_id is None and os.path.exists(id_path):
        with open(id_path) as f:
            run_id = f.read().strip()
    run = wandb.init(project=args.project, name=args.name or header.get("run"), config=config, id=run_id,
                     resume="allow" if run_id else None, mode=args.mode)
    if run_id is None:
        with open(id_path, "w") as f:
            f.write(run.id + "\n")

    try:
        for rec in records:
            step = rec.get("step")
            if step is None or "event" in rec or step < run.step:  # run.step: next step wandb accepts
                continue
            if "val/bpb" not in rec and step % args.every != 0:
                continue
            run.log(rec, step=step)
    except KeyboardInterrupt:
        pass
    run.finish()


if __name__ == "__main__":
    sys.exit(main())
