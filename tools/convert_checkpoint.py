"""
Convert a training checkpoint between Python nanochat (.pt) and nanochat-cpp (.safetensors).

Files per step: model_<step>, optim_<step>_rank<r> (optional, one per rank), meta_<step>.json (copied as is).
Optimizer state keeps Python's state_dict structure: tensors "state.<param index>.<field>" plus the param groups as
JSON in the safetensors metadata ("param_groups").

  python tools/convert_checkpoint.py SRC DST --to safetensors [--step N] [--model-only]
"""
import argparse
import glob
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file, safe_open


def last_step(src, ext):
    steps = [int(m.group(1)) for f in glob.glob(os.path.join(src, f"model_*.{ext}"))
             if (m := re.search(r"model_(\d+)\.", os.path.basename(f)))]
    if not steps:
        raise SystemExit(f"no model_*.{ext} in {src}")
    return max(steps)


def optim_to_safetensors(data):
    tensors = {}
    for idx, state in data["state"].items():
        for key, value in state.items():
            tensors[f"state.{idx}.{key}"] = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.int64)
    return tensors, {"param_groups": json.dumps(data["param_groups"])}


def optim_from_safetensors(tensors, metadata):
    state = {}
    for name, value in tensors.items():
        _, idx, key = name.split(".", 2)
        state.setdefault(int(idx), {})[key] = value.item() if key == "step" else value
    groups = json.loads(metadata["param_groups"])
    for group in groups:
        group.pop("name", None)  # C++ only
        if "betas" in group:
            group["betas"] = tuple(group["betas"])
    return {"state": state, "param_groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--to", choices=["safetensors", "pt"], required=True)
    parser.add_argument("--step", type=int, help="default: the last step in src")
    parser.add_argument("--model-only", action="store_true", help="skip optimizer state (e.g. for evals)")
    args = parser.parse_args()
    src_ext = "pt" if args.to == "safetensors" else "safetensors"
    step = args.step if args.step is not None else last_step(args.src, src_ext)
    os.makedirs(args.dst, exist_ok=True)
    src = lambda kind: os.path.join(args.src, f"{kind}_{step:06d}")
    dst = lambda kind: os.path.join(args.dst, f"{kind}_{step:06d}")

    if args.to == "safetensors":
        model = torch.load(src("model") + ".pt", map_location="cpu")
        model = {k.removeprefix("_orig_mod."): v.contiguous() for k, v in model.items()}
        save_file(model, dst("model") + ".safetensors")
    else:
        torch.save(load_file(src("model") + ".safetensors"), dst("model") + ".pt")

    optim_paths = [] if args.model_only else sorted(glob.glob(src("optim") + f"_rank*.{src_ext}"))
    for path in optim_paths:
        rank = re.search(r"_rank(\d+)\.", path).group(1)
        if args.to == "safetensors":
            tensors, metadata = optim_to_safetensors(torch.load(path, map_location="cpu"))
            save_file({k: v.contiguous() for k, v in tensors.items()}, dst("optim") + f"_rank{rank}.safetensors",
                      metadata=metadata)
        else:
            with safe_open(path, "pt") as f:
                metadata = f.metadata()
            torch.save(optim_from_safetensors(load_file(path), metadata), dst("optim") + f"_rank{rank}.pt")

    shutil.copyfile(src("meta") + ".json", dst("meta") + ".json")
    print(f"converted step {step}: {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
