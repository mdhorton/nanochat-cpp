# porting nanochat to c++

The goal is to port nanochat (python) to c++ with and targeting sm120 GPUs.

## original nanochat

The original nanochat repo is cloned into `external/nanochat`. Do not make any changes to the original project.

## high level breakdown

nanochat has 4 high-level parts:

1. tokenizer
2. training
3. evaluation
4. inference

## porting phases

### phase 0

- create pixi.toml
- setup build
- download data

### phase 1 (complete)

port BPE tokenizer. this uses the same logic as the original tokenizer.

create tests so we can compare output with the original. they should be identical.

the rustbpe/tiktoken benchmark moved to phase 5.

### phase 2

port pre-training (`gpt.py`, `optim.py`, `dataloader.py`, `loss_eval.py`, `scripts/base_train.py`) to libtorch.

- oracle: python nanochat run uncompiled (`TORCHDYNAMO_DISABLE=1`), SDPA attention, seed 42. the dataloader must
  match exactly; everything else within bf16 tolerances.
- golden data: `tools/export_train_golden.py` -> `cache/golden/train/`.
- parity config: d4, head_dim 64, T=256, SSSL, device batch 8, total batch 4096, 20 iterations.
- target: d24 on 2 GPUs + FP8.

steps:

1. libtorch in the build (`nanochat_train` library), smoke test.
2. safetensors read/write, golden export skeleton.
3. model (`gpt`): config from depth, forward, init. check logits, loss, grads vs python.
4. optimizer (`optim`): MuonAdamW single GPU. check steps vs python.
5. dataloader: BOS best-fit packing, resume state. check batches are identical to python.
6. val bits per byte (`loss_eval`).
7. checkpoints: safetensors + meta json, `.pt` <-> safetensors converter.
8. `base_train` app: flags, scaling, schedules, grad accumulation, eval, logging. check tiny loss curve vs python,
   d6 sanity run. **stop for review before d12.**
9. memory/throughput for d24 on 1 GPU (chunked cross-entropy, activation recompute if needed).
   done: `--attention fa2` (default), PyTorch's built-in FlashAttention-2 with native sliding windows. parity tests
   use `--attention sdpa`, which is bit-identical to python on sm120.
   done: `--loss-chunk-rows N` chunked lm_head + softcap + cross-entropy with recompute (plain libtorch). saves ~7GB
   at d12 but costs ~12% speed, so it's off by default. a fused softcap + cross-entropy kernel would make it faster.
10. 2 GPUs: NCCL, ZeRO-2 MuonAdamW, launcher.
11. FP8.
