# run history

Most of the sm120 tuning work was done using `--depth=12` to help speedup development iterations.

# medium runs @ d12

These runs are designed to finish in under 5 minutes. It's easier to spot bpb regressions than quick runs.

```bash
--depth=12 --device-batch-size=8 --num-iterations=100 --eval-tokens=4194304 --eval-every=-1 --save=false
```

Runs 1-3 are from the original nanochat port (python → c++). No tuning yet.

Runs 4+ are our tuning and customisation for sm120. Much of the tuning is fusing and other optimisations that would
normally be done by inductor and triton.

| run | toks/sec |     loss |      bpb | memory |   time | notes                                          |
|----:|---------:|---------:|---------:|-------:|-------:|------------------------------------------------|
|   1 |   36,192 | 4.854269 | 1.439428 |  17.0g | 24.11m | --nproc=1                                      |
|   2 |   72,305 | 4.800026 | 1.434599 |  16.4g | 12.08m | --nproc=2 (ZeRO-2, MuonAdamW)                  |
|   3 |   80,129 | 4.820226 | 1.441378 |  16.0g | 10.91m | --nproc=2 --fp8                                |
|   4 |  107,091 | 4.820269 | 1.441330 |  16.0g |  8.15m | --nproc=2 --fp8 --attention=fa2                |
|   5 |  138,679 | 4.804933 | 1.436173 |   8.2g |  6.29m | --loss-chunk-rows=4096                         |
|   6 |  155,024 | 4.808020 | 1.437057 |   8.2g |  5.63m | fused rotary + QK-norm                         |
|   7 |  158,808 | 4.807032 | 1.436761 |   8.2g |  5.49m | fused resid/x0 + λ-blend                       |
|   8 |  181,216 | 4.803973 | 1.435802 |   8.2g |  4.82m | fused relu² + amax                             |
|   9 |  183,137 | 4.806090 | 1.436472 |   8.2g |  4.77m | fused residual add + rms_norm                  |
|  10 |  188,796 | 4.807217 | 1.436816 |   7.9g |  4.62m | merged q/k/v                                   |
|  11 |  192,034 | 4.809634 | 1.437552 |   7.9g |  4.54m | embed grads into .grad                         |
|  12 |  206,686 | 4.821239 | 1.441635 |   8.2g |  4.22m | lm_head fp8                                    |
|  13 |  209,310 | 4.822021 | 1.441975 |   8.3g |  4.17m | weight caching                                 |
|  14 |  215,449 | 4.802828 | 1.435484 |   8.4g |  4.04m | --fp8-recipe=mxfp8                             |
|  15 |  221,792 | 4.805797 | 1.436430 |   8.4g |  3.93m | --fp8-recipe=mxfp8 (relu² backward)            |
|  16 |  225,049 | 4.804839 | 1.436043 |   8.4g |  3.86m | --fp8-recipe=mxfp8 (softcap CE grads)          |
|  17 |  230,934 | 4.802351 | 1.435362 |   8.4g |  3.77m | --fp8-recipe=mxfp8 (qkv + rotary + val embed)  |
|  18 |  234,497 | 4.806611 | 1.436663 |   8.7g |  3.71m | --fp8-recipe=mxfp8 (lm_head grad_w)            |
|  19 |  237,122 | 4.802950 | 1.435503 |   8.7g |  3.68m | x0 gradient fold                               |
|  20 |  242,702 | 4.805256 | 1.436230 |   8.7g |  3.60m | cuBLASLt workspace 32 MB (split-K)             |
|  21 |  247,254 | 4.802301 | 1.435336 |   8.5g |  3.52m | fused smear/backout, dW into .grad             |
|  22 |  249,284 | 4.802814 | 1.435525 |   8.5g |  3.50m | NCCL_PROTO = "Simple" NCCL_MIN_NCHANNELS = "8" |
|  23 |          |          |          |        |        |                                                |

Run 4: sm120 can't use fa3. however, fa2 works pretty good. `--nproc=2 --fp8 --attention=fa2` enabled by default going
forward.

Run 5: python nanochat keeps several GB of logits alive at once. this triggers OOM at `--depth=24` on my setup. chunked
and fused CE loss to reduce vram usage. `--loss-chunk-rows=4096` enabled by default going forward.

Runs 6-9: sm120 have lower memory bandwidth than datacenter GPUs. for example, RTX Pro 4000 has ~6x slower dram
bandwidth vs H100. fusing memory bound kernels is usually worth it. this is also the area normally covered by pytorch
inductor.

Runs 14-18: mxfp8 support. this reduced memory traffic around quantization. this worked because several kernels are
memory bound. `--fp8-recipe=mxfp8` enabled by default going forward.

Runs 22: NCCL optimisations. `NCCL_PROTO = "Simple" NCCL_MIN_NCHANNELS = "8"` enabled by default going forward.

# medium runs @ d24

These runs started at d12 run 5 because otherwise it would OOM.

```bash
--fp8 --nproc=2 --depth=24 --device-batch-size=2 --num-iterations=30 --eval-tokens=4194304 --eval-every=-1 --save=false
```

| toks/sec |     loss |      bpb | memory |   time | git tag and flags (if any) |
|---------:|---------:|---------:|-------:|-------:|----------------------------|
|          |          |          |        |        | run5                       |
|   43,449 | 6.149978 | 1.720151 |  20.3g | 12.05m | run14 --fp8-recipe=mxfp8   |
|          |          |          |        |        |                            |

# full run @d12

```bash
--fp8 --nproc=2 --depth=12 --device-batch-size=8
```

| steps | loss     | bpb      | time    |
|-------|----------|----------|---------|
| 2520  | 2.814477 | 0.847031 | 178.26m |
|       |          |          |         |