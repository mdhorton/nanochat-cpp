"""
Export golden pre-training data from Python nanochat for the C++ parity tests (tests/test_*.cpp in train_tests).

Python runs uncompiled (TORCHDYNAMO_DISABLE=1) with SDPA attention, so both sides use the same ATen kernels.
Writes to --out-dir (default <base-dir>/golden/train):
- safetensors_check.safetensors: known tensors, checks the C++ reader against a Python writer
- gpt_*: tiny model init weights, perturbed weights, a batch, and logits / loss / grads
- optim_*: MuonAdamW steps from given grads
- loader_*: batches from the BOS best-fit data loader (train, resume, rank 1 of 2, val) on the real data
- bpb*: val bits per byte of a small real-vocab model
- ckpt*: a Python checkpoint converted to safetensors (tools/convert_checkpoint.py), and the step after it
- train*: a tiny base_train run (per-step losses, val bpb, schedules, final weights)
"""
import argparse
import json
import os
import struct
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # before torch is imported
sys.dont_write_bytecode = True  # keep the nanochat tree untouched

import torch

from export_tokenizer import CACHE_DIR

NANOCHAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "nanochat")

# parity config: small, but covers S/L windows, value embeddings, backout, vocab padding and tall/wide Muon shapes
TINY = dict(sequence_len=256, vocab_size=1000, n_layer=4, n_head=4, n_kv_head=4, n_embd=256, window_pattern="SSSL")
OPTIM_ARGS = dict(unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02, weight_decay=0.2, scalar_lr=0.5)
OPTIM_STEPS = 3


def save(path, tensors, metadata=None):
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    try:
        from safetensors.torch import save_file
        save_file(tensors, path, metadata={**(metadata or {}), "writer": "safetensors"})
    except ImportError:
        save_builtin(path, tensors, {**(metadata or {}), "writer": "builtin"})


def save_builtin(path, tensors, metadata):
    # fallback until the safetensors package is installed: same format, written by hand
    dtypes = {torch.float32: "F32", torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32"}
    header, blobs, offset = {"__metadata__": metadata}, [], 0
    for name in sorted(tensors):
        t = tensors[name]
        blob = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        header[name] = {"dtype": dtypes[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for blob in blobs:
            f.write(blob)


def export_safetensors_check(out):
    # must match check_tensors() in tests/test_safetensors.cpp
    tensors = {
        "f32": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 3,
        "bf16": (torch.arange(6, dtype=torch.float32) - 2.5).to(torch.bfloat16),
        "i64": torch.arange(-3, 4, dtype=torch.int64),
        "i32": torch.arange(5, dtype=torch.int32).reshape(5, 1),
        "scalar": torch.tensor(1.25, dtype=torch.float32),
    }
    save(out("safetensors_check.safetensors"), tensors, {"source": "python"})


def build_tiny_model():
    from nanochat.gpt import GPT, GPTConfig
    with torch.device("meta"):
        model = GPT(GPTConfig(**TINY))
    model.to_empty(device="cuda")
    torch.manual_seed(42)
    model.init_weights()
    return model


def perturb(model, seed):
    # init has zero projections and tiny gates; add noise so every weight and path is exercised
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_((0.02 * torch.randn(p.shape, generator=g, device="cuda")).to(p.dtype))


def random_batch(seed, B=2):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, TINY["vocab_size"], (B, TINY["sequence_len"] + 1), generator=g)
    return ids[:, :-1].cuda(), ids[:, 1:].cuda()


def export_gpt(out):
    model = build_tiny_model()
    with open(out("gpt_config.json"), "w") as f:
        json.dump(TINY, f)
    save(out("gpt_init.safetensors"), model.state_dict())
    perturb(model, seed=1)
    save(out("gpt_perturbed.safetensors"), model.state_dict())
    idx, targets = random_batch(seed=2)
    save(out("gpt_batch.safetensors"), {"idx": idx, "targets": targets})
    with torch.no_grad():
        logits = model(idx)
    loss = model(idx, targets)
    loss.backward()
    grads = {f"grad.{name}": p.grad for name, p in model.named_parameters()}
    save(out("gpt_outputs.safetensors"), {"loss": loss, "logits": logits, **grads})
    print(f"wrote gpt goldens (loss {loss.item():.6f})")


def export_optim(out):
    # grads come from the model, but the C++ test feeds them in directly: this isolates the optimizer
    model = build_tiny_model()
    perturb(model, seed=1)
    optimizer = model.setup_optimizer(**OPTIM_ARGS)
    with open(out("optim_config.json"), "w") as f:
        json.dump({**OPTIM_ARGS, "steps": OPTIM_STEPS}, f)
    for step in range(OPTIM_STEPS):
        idx, targets = random_batch(seed=10 + step)
        model(idx, targets).backward()
        grads = {name: p.grad for name, p in model.named_parameters()}
        save(out(f"optim_grads_{step}.safetensors"), grads)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        save(out(f"optim_params_{step}.safetensors"), model.state_dict())
    print(f"wrote {OPTIM_STEPS} optimizer steps")


def loader_batches(split, B, T, num_batches, resume=None, rank=0, world_size=1):
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit as make_loader
    from nanochat.tokenizer import get_tokenizer
    env = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size)}
    if world_size > 1:
        os.environ.update(env)  # get_dist_info reads torchrun's env
    try:
        loader = make_loader(get_tokenizer(), B, T, split, device="cuda", resume_state_dict=resume)
        inputs, targets, states = [], [], []
        for _ in range(num_batches):
            x, y, state = next(loader)
            inputs.append(x.clone())
            targets.append(y.clone())
            states.append(dict(state))
    finally:
        if world_size > 1:
            for k in env:
                os.environ.pop(k, None)
    return torch.stack(inputs), torch.stack(targets), states


def export_dataloader(out):
    import pyarrow.parquet as pq
    from nanochat.dataset import list_parquet_files
    first_file_rgs = pq.ParquetFile(list_parquet_files()[0]).num_row_groups
    cases = {
        "train": dict(split="train", B=4, T=256, num_batches=4),
        "resume": dict(split="train", B=4, T=256, num_batches=3, resume=None),  # resume filled in below
        "resume_next_file": dict(split="train", B=2, T=256, num_batches=2,
                                 resume={"pq_idx": 0, "rg_idx": first_file_rgs - 1, "epoch": 1}),
        "rank1": dict(split="train", B=2, T=256, num_batches=2, rank=1, world_size=2),
        "val": dict(split="val", B=2, T=512, num_batches=2),
    }
    meta = {}
    for name, case in cases.items():
        if name == "resume":
            case["resume"] = meta["train"]["states"][1]
        inputs, targets, states = loader_batches(**case)
        save(out(f"loader_{name}.safetensors"), {"inputs": inputs, "targets": targets})
        meta[name] = {**case, "states": states}
    with open(out("loader_cases.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {len(cases)} dataloader cases")


def export_bpb(out):
    # a small model over the real vocab, evaluated on the real val shard
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit as make_loader
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    tokenizer = get_tokenizer()
    config = dict(TINY, vocab_size=tokenizer.get_vocab_size(), n_layer=2)
    with torch.device("meta"):
        model = GPT(GPTConfig(**config))
    model.to_empty(device="cuda")
    torch.manual_seed(42)
    model.init_weights()
    perturb(model, seed=3)
    B, steps = 2, 3
    loader = make_loader(tokenizer, B, config["sequence_len"], "val", device="cuda")
    bpb = evaluate_bpb(model, loader, steps, get_token_bytes(device="cuda"))
    save(out("bpb_model.safetensors"), model.state_dict())
    with open(out("bpb.json"), "w") as f:
        json.dump({"config": config, "B": B, "steps": steps, "bpb": bpb}, f)
    print(f"wrote bpb golden ({bpb:.6f})")


def export_ckpt(out):
    # a Python .pt checkpoint after 2 steps, converted with tools/convert_checkpoint.py, plus a 3rd step to check that
    # the C++ side resumes from it exactly
    try:
        import safetensors  # noqa: F401
    except ImportError:
        print("skipping ckpt: needs the safetensors package")
        return
    import shutil
    import subprocess
    from nanochat.checkpoint_manager import save_checkpoint
    model = build_tiny_model()
    perturb(model, seed=1)
    optimizer = model.setup_optimizer(**OPTIM_ARGS)

    def backward(seed):
        idx, targets = random_batch(seed)
        model(idx, targets).backward()

    for seed in (20, 21):
        backward(seed)
        optimizer.step()
        model.zero_grad(set_to_none=True)
    pt_dir, st_dir, back_dir = out("ckpt_pt"), out("ckpt"), out("ckpt_pt_back")
    save_checkpoint(pt_dir, 2, model.state_dict(), optimizer.state_dict(), {"step": 2, "model_config": TINY})
    convert = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_checkpoint.py")
    subprocess.run([sys.executable, convert, pt_dir, st_dir, "--to", "safetensors"], check=True)
    subprocess.run([sys.executable, convert, st_dir, back_dir, "--to", "pt"], check=True)
    # the round trip must reproduce the original .pt files (safetensors doesn't keep key order)
    orig = torch.load(os.path.join(pt_dir, "model_000002.pt"), map_location="cpu")
    back = torch.load(os.path.join(back_dir, "model_000002.pt"), map_location="cpu")
    assert orig.keys() == back.keys()
    assert all(torch.equal(orig[k], back[k]) and orig[k].dtype == back[k].dtype for k in orig)
    orig = torch.load(os.path.join(pt_dir, "optim_000002_rank0.pt"), map_location="cpu")
    back = torch.load(os.path.join(back_dir, "optim_000002_rank0.pt"), map_location="cpu")
    assert orig["param_groups"] == back["param_groups"]
    for idx, state in orig["state"].items():
        for key, value in state.items():
            assert (value == back["state"][idx][key]) if not torch.is_tensor(value) else torch.equal(value, back["state"][idx][key])
    shutil.rmtree(pt_dir)
    shutil.rmtree(back_dir)

    backward(22)
    save(out("ckpt_grads.safetensors"), {name: p.grad for name, p in model.named_parameters()})
    optimizer.step()
    save(out("ckpt_params.safetensors"), model.state_dict())
    print("wrote checkpoint goldens")


# tiny base_train run: d4 (d_model 256, 4 heads of 64), T=256, 2 grad accumulation steps, 20 iterations
TRAIN = dict(depth=4, aspect_ratio=64, head_dim=64, max_seq_len=256, window_pattern="SSSL", num_iterations=20,
             target_flops=-1.0, target_param_data_ratio=12, device_batch_size=8, total_batch_size=4096,
             embedding_lr=0.3, unembedding_lr=0.008, weight_decay=0.28, matrix_lr=0.02, scalar_lr=0.5, warmup_steps=2,
             warmdown_ratio=0.65, final_lr_frac=0.05, eval_every=10, eval_tokens=4 * 8 * 256)


def export_train(out):
    # scripts/base_train.py transcribed (it imports wandb/jinja2), minus compile, wandb, CORE, sampling and fp8
    import math
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit as val_loader_fn
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit as train_loader_fn
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    args = argparse.Namespace(**TRAIN)
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device="cuda")
    vocab_size = tokenizer.get_vocab_size()

    def build_model_meta(depth):
        base_dim = depth * args.aspect_ratio
        model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
        num_heads = model_dim // args.head_dim
        config = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=depth, n_head=num_heads,
                           n_kv_head=num_heads, n_embd=model_dim, window_pattern=args.window_pattern)
        with torch.device("meta"):
            return GPT(config)

    torch.manual_seed(42)  # compute_init
    model = build_model_meta(args.depth)
    model.to_empty(device="cuda")
    model.init_weights()

    def get_scaling_params(m):
        params_counts = m.num_scaling_params()
        return params_counts['transformer_matrices'] + params_counts['lm_head']
    num_scaling_params = get_scaling_params(model)
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)
    D_REF = args.target_param_data_ratio * get_scaling_params(build_model_meta(12))
    B_REF = 2**19
    total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        total_batch_size = 2 ** round(math.log2(B_REF * (target_tokens / D_REF) ** 0.383))
    batch_lr_scale = 1.0
    batch_ratio = total_batch_size / B_REF
    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio ** 0.5
    weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr * batch_lr_scale,
                                      embedding_lr=args.embedding_lr * batch_lr_scale,
                                      scalar_lr=args.scalar_lr * batch_lr_scale,
                                      matrix_lr=args.matrix_lr * batch_lr_scale, weight_decay=weight_decay_scaled)
    train_loader = train_loader_fn(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device="cuda")
    build_val_loader = lambda: val_loader_fn(tokenizer, args.device_batch_size, args.max_seq_len, split="val",
                                             device="cuda")
    x, y, _ = next(train_loader)
    num_iterations = args.num_iterations

    def get_lr_multiplier(it):
        warmup_iters = args.warmup_steps
        warmdown_iters = round(args.warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * args.final_lr_frac

    def get_muon_momentum(it):
        warmdown_iters = round(args.warmdown_ratio * num_iterations)
        warmdown_start = num_iterations - warmdown_iters
        if it < 400:
            frac = it / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif it >= warmdown_start:
            progress = (it - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97

    def get_weight_decay(it):
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

    grad_accum_steps = total_batch_size // (args.device_batch_size * args.max_seq_len)
    losses, evals, schedule = [], {}, []
    step = 0
    while True:
        last_step = step == num_iterations
        if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
            evals[step] = evaluate_bpb(model, build_val_loader(), eval_steps, token_bytes)
        if last_step:
            break
        for micro_step in range(grad_accum_steps):
            loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, _ = next(train_loader)
        lrm, muon_momentum, muon_weight_decay = get_lr_multiplier(step), get_muon_momentum(step), get_weight_decay(step)
        schedule.append([lrm, muon_momentum, muon_weight_decay])
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        model.zero_grad(set_to_none=True)
        losses.append(train_loss.item())
        step += 1
    save(out("train_final.safetensors"), model.state_dict())
    plan = dict(num_scaling_params=num_scaling_params, target_tokens=target_tokens, d_ref=D_REF,
                total_batch_size=total_batch_size, batch_lr_scale=batch_lr_scale,
                weight_decay_scaled=weight_decay_scaled, num_iterations=num_iterations,
                grad_accum_steps=grad_accum_steps)
    with open(out("train.json"), "w") as f:
        json.dump({"options": TRAIN, "plan": plan, "losses": losses, "evals": evals, "schedule": schedule}, f, indent=1)
    print(f"wrote train golden: losses {losses[0]:.4f} -> {losses[-1]:.4f}, val bpb {evals}")


SECTIONS = {"safetensors": export_safetensors_check, "gpt": export_gpt, "optim": export_optim,
            "dataloader": export_dataloader, "bpb": export_bpb, "ckpt": export_ckpt, "train": export_train}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default=CACHE_DIR, help="nanochat data directory")
    parser.add_argument("--out-dir", help="default: <base-dir>/golden/train")
    parser.add_argument("--nanochat-dir", default=NANOCHAT_DIR)
    parser.add_argument("--sections", default=",".join(SECTIONS), help="comma-separated subset of " + ",".join(SECTIONS))
    args = parser.parse_args()
    args.out_dir = args.out_dir or os.path.join(args.base_dir, "golden", "train")

    os.environ["NANOCHAT_BASE_DIR"] = args.base_dir  # nanochat reads it at import time
    sys.path.insert(0, args.nanochat_dir)
    import nanochat.flash_attention as fa
    fa._override_impl = "sdpa"
    fa.USE_FA3 = fa._resolve_use_fa3()
    from nanochat.common import COMPUTE_DTYPE
    assert COMPUTE_DTYPE == torch.bfloat16, COMPUTE_DTYPE
    torch.set_float32_matmul_precision("high")  # as compute_init

    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)
    for name in args.sections.split(","):
        SECTIONS[name](out)


if __name__ == "__main__":
    main()
