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
   done: `--loss-chunk-rows N` (default 4096) chunked lm_head + fused softcap/cross-entropy CUDA kernel
   (`model/softcap_ce_kernel.cu`); gradients are computed in forward, so nothing is recomputed. d12 batch 8: 59.5k vs 46.1k
   tok/s, 9.1 vs 16.9GB. `0` = python's unchunked path (parity tests).
   todo: d24 batch 2 fits (21.2GB, 7.3k tok/s), batch 4 OOMs: needs activation recompute or ZeRO on 2 GPUs.
10. 2 GPUs: NCCL, ZeRO-2 MuonAdamW, launcher.
    done: `base_train --nproc 2` (c10d ProcessGroupNCCL + TCPStore, `train/dist.{h,cpp}`), ZeRO-2 MuonAdamW as
    `optim.py`. tiny 2-GPU run bit-identical to python under torchrun (`train_ddp2` golden). d12: 122k vs 61.7k tok/s.
    d24 batch 2: 22.7k tok/s at 18.8GB/GPU (~3 days at ratio 8); batch 4 OOMs.
11. FP8.
    done: `--fp8`, port of `fp8.py` (tensorwise `_scaled_mm`, e4m3/e5m2, eval in bf16). quantization is fused into CUDA
    kernels (`model/fp8_kernel.cu`), which also write the transposed copies backward needs; bit-identical to python
    (`fp8`, `train_fp8` goldens). the chunked loss keeps lm_head in bf16. d12 1 GPU: 67.7k vs 60.0k tok/s. d24 2 GPUs:
    30.0k (dbs 2) / 31.1k (dbs 4, 22.9GB) vs 22.7k tok/s. d6/d12 300-step val bpb same as bf16.
12. fused elementwise kernels (`--fused`, default on; off = python's op-by-op path, used by the parity tests).
    done: rotary + QK norm (`model/rotary_norm_kernel.cu`), fp32 inside, closer to fp64 than the bf16 ops. d12 FP8 1 GPU:
    74.5k vs 66.6k tok/s.
    todo: resid/x0 lambda blend, residual add + rms_norm, value-embedding gate.
