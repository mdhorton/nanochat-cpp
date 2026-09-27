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
    (`fp8`, `train_fp8` goldens). the chunked loss ran lm_head in bf16 until step 12. d12 1 GPU: 67.7k vs 60.0k tok/s. d24 2 GPUs:
    30.0k (dbs 2) / 31.1k (dbs 4, 22.9GB) vs 22.7k tok/s. d6/d12 300-step val bpb same as bf16.
12. fused elementwise kernels (`--fused`, default on; off = python's op-by-op path, used by the parity tests).
    done: rotary + QK norm (`model/rotary_norm_kernel.cu`), fp32 inside, closer to fp64 than the bf16 ops. d12 FP8 1 GPU:
    74.5k vs 66.6k tok/s.
    done: resid/x0 lambda blend, lambda grads summed in fp32, deterministic. d12 FP8 1 GPU: 77.3k tok/s (+16% total).
    the blend is bit-identical to python's ops, which round the fp32 lambdas to bf16 (type promotion) and each
    product: full-precision math trained worse (d12 300 steps: val bpb 1.052 vs 1.028).
    done: relu² (`model/relu_square_kernel.cu`), bit-identical. with FP8 it's folded into the quantize kernels
    (c_proj's input quantized straight from h; dh's amax in the backward kernel). quick-d12 2 GPUs: 181.9k vs 159.8k
    tok/s (unfused 139.3k).
    done: residual add + lambda blend + rms_norm in one row kernel (`model/residual_norm_kernel.cu`, replaces the
    blend kernel); backward folds in autograd's sum of the residual's two gradients. add/blend bit-identical, norm fp32
    inside. quick-d12 2 GPUs: 183.8k tok/s (+1%).
    done: merged q/k/v (fused path): one GEMM each for forward and weight grads. with FP8, x is quantized once and
    row-wise `_scaled_mm` scales (the fp32 product of each part's two scales, the other operand's scales 1) keep python's
    per-tensor scales: forward bit-identical, weight grads up to rare 1-ulp flips; the input grad stays 3 GEMMs, summed
    in one kernel with the value-embedding gate's input grad. quick-d12 2 GPUs: 189.8k tok/s (+3%), peak mem 7.86 GB.
    done: embedding grads straight into .grad (`model/embedding_kernel.cu`): wte and the value embeddings in one lookup,
    one sort, one kernel summing each touched row in fp32 into .grad (no per-call zero fill and full-size add; ZeRO-2
    reduces in the optimizer step, after the micro-steps). quick-d12 2 GPUs: 193.7k tok/s (+2%).
    done: FP8 lm_head in the chunked loss (python quantizes it too): x and the weight quantized once, the softcap kernel
    reports its gradient's amax, each chunk's gradient gets its own e5m2 scale (python: one for all rows), grad_w
    accumulates in fp32. vs the unchunked fp8 path: grad_x within 2e-5, grad_w 0.2% (python rounds it to bf16).
    quick-d12 2 GPUs: 208.3k tok/s (+7.5%; --fused=false 145.2k), bpb 1.7791 vs 1.7784 with bf16 lm_head (30 steps).
    done: FP8 weights cached between optimizer steps (`Fp8WeightCache`, fused path): reused while the weight keeps its
    storage and version counter, bit-identical. quick-d12 2 GPUs: 210.8k tok/s (+1.2%), peak mem 8.32 GB.
    done: MXFP8 (`--fp8-recipe=mxfp8`, not in Python; default tensorwise): every operand e4m3 with an e8m0 scale per
    32 values along K (2^ceil(log2(amax/448))), written in cuBLAS's swizzled layout by one kernel that also writes
    the transpose (`model/fp8_kernel.cu`). No amax pass; merged q/k/v dgrad becomes one GEMM. Needs dims % 128, else
    tensorwise. GEMMs as fast as tensorwise (bench: 67.9 vs 68.3 ms per micro-step). quick-d12 2 GPUs: 217.3k tok/s
    (+3.1%), bpb 1.7783 vs 1.7792 tensorwise (30 steps).
    done: MX relu^2 backward: dh quantized straight from g and h (`quantize_mx_relu_square_bwd`), never written in
    bf16; bit-identical. quick-d12 2 GPUs: 223.8k tok/s (+3.0%).
    done: MX lm_head gradient: the loss kernel also writes each row's lse, then a 32x64-tile kernel recomputes the
    gradient from the logits and quantizes it (`softcap_ce_grad_mx`); bit-identical. lm_head kernels 9.3 -> 6.7 ms
    per micro-step. quick-d12 2 GPUs: 227.4k tok/s (+1.6%).
    todo: MX quantization in residual_norm and rotary_norm bwd. The MX tile kernel (`mx_kernel.cuh`) runs at
    ~500-530 GB/s, 90-96% of a bf16 copy (555 GB/s): little left to gain in the kernel itself.
