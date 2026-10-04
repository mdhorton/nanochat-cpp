# run history

To speedup dev iterations, initial sm120 tuning was done at `--depth=12`. Later, `--depth=24` became fast enough to
iterate with.

Most of the runs have an associated git tag (eg, run27).

# medium runs @ d12

`--depth=12 --device-batch-size=8 --num-iterations=100 --eval-tokens=4194304 --eval-every=-1 --save=false`

Runs 1-3 are from the original nanochat port (python → c++). No tuning yet.

Runs 4+ are the tuning and customisations for sm120. Probably more than half the tuning is fusing and other
optimisations that would normally be handled by inductor and triton.

| run | toks/sec |     loss |      bpb | memory |   time | notes                                          |
|----:|---------:|---------:|---------:|-------:|-------:|------------------------------------------------|
|   1 |   36,192 | 4.854269 | 1.439428 |  17.0g | 24.11m | --nproc=1                                      |
|   2 |   72,305 | 4.800026 | 1.434599 |  16.4g | 12.08m | --nproc=2 (ZeRO-2, MuonAdamW)                  |
|   3 |   80,129 | 4.820226 | 1.441378 |  16.0g | 10.91m | --nproc=2 --fp8=true                           |
|   4 |  107,091 | 4.820269 | 1.441330 |  16.0g |  8.15m | --nproc=2 --fp8=true --attention=fa2           |
|   5 |  138,679 | 4.804933 | 1.436173 |   8.2g |  6.29m | --loss-chunk-rows=4096                         |
|   6 |  155,024 | 4.808020 | 1.437057 |   8.2g |  5.63m | fused rotary + QK-norm                         |
|   7 |  158,808 | 4.807032 | 1.436761 |   8.2g |  5.49m | fused resid/x0 + λ-blend                       |
|   8 |  181,216 | 4.803973 | 1.435802 |   8.2g |  4.82m | fused relu² + amax                             |
|   9 |  183,137 | 4.806090 | 1.436472 |   8.2g |  4.77m | fused residual add + rms_norm                  |
|  10 |  188,796 | 4.807217 | 1.436816 |   7.9g |  4.62m | merged q/k/v                                   |
|  11 |  192,034 | 4.809634 | 1.437552 |   7.9g |  4.54m | embed grads into .grad                         |
|  12 |  206,686 | 4.821239 | 1.441635 |   8.2g |  4.22m | fp8 lm_head                                    |
|  13 |  209,310 | 4.822021 | 1.441975 |   8.3g |  4.17m | added weight caching                           |
|  14 |  215,449 | 4.802828 | 1.435484 |   8.4g |  4.04m | --fp8-recipe=mxfp8                             |
|  15 |  221,792 | 4.805797 | 1.436430 |   8.4g |  3.93m | --fp8-recipe=mxfp8 (relu² backward)            |
|  16 |  225,049 | 4.804839 | 1.436043 |   8.4g |  3.86m | --fp8-recipe=mxfp8 (softcap CE grads)          |
|  17 |  230,934 | 4.802351 | 1.435362 |   8.4g |  3.77m | --fp8-recipe=mxfp8 (qkv + rotary + val embed)  |
|  18 |  234,497 | 4.806611 | 1.436663 |   8.7g |  3.71m | --fp8-recipe=mxfp8 (lm_head grad_w)            |
|  19 |  237,122 | 4.802950 | 1.435503 |   8.7g |  3.68m | x0 gradient fold                               |
|  20 |  242,702 | 4.805256 | 1.436230 |   8.7g |  3.60m | cuBLASLt workspace 32 MB (split-K)             |
|  21 |  247,254 | 4.802301 | 1.435336 |   8.5g |  3.52m | fused smear/backout, dW into .grad             |
|  22 |  249,284 | 4.802814 | 1.435525 |   8.5g |  3.50m | NCCL_PROTO = "Simple" NCCL_MIN_NCHANNELS = "8" |
|  23 |  250,734 | 4.803105 | 1.435576 |   8.5g |  3.48m | fused AdamW kernel, Muon grad stacks           |
|  24 |  252,656 | 4.800344 | 1.434727 |   8.5g |  3.45m | fused MX quantize into residual_norm           |
|  25 |  257,917 | 4.799418 | 1.434437 |   8.5g |  3.38m | --attention=bf16 (FA bf16 forward)             |
|  26 |  267,715 | 4.803928 | 1.435875 |   8.6g |  3.26m | custom FA with mxfp8                           |
|  27 |  273,814 | 4.804844 | 1.436141 |   8.0g |  3.17m | cutlass mxfp8 gemms                            |

Run 4: sm120 can't use fa3. however, fa2 works pretty good. `--nproc=2 --fp8=true --attention=fa2` enabled by default
going forward.

Run 5: python nanochat keeps several GB of logits alive at once. this triggers OOM at `--depth=24` on my setup. chunked
and fused CE loss to reduce vram usage. `--loss-chunk-rows=4096` enabled by default going forward.

Runs 6-9: sm120 have lower memory bandwidth than datacenter GPUs. for example, RTX Pro 4000 has ~6x slower dram
bandwidth vs H100. fusing memory bound kernels is usually worth it. this is also the area normally covered by pytorch
inductor.

Runs 14-18: mxfp8 support. this reduced memory traffic around quantisation. this worked because several kernels are
memory bound. `--fp8-recipe=mxfp8` enabled by default going forward.

Run 22: `NCCL_PROTO = "Simple" NCCL_MIN_NCHANNELS = "8"` enabled by default going forward.

Run 26: `--attention=mx` enabled by default going forward.

# quick runs @ d24

These start at run 5 because otherwise it would OOM.

`--depth=24 --device-batch-size=2 --num-iterations=10 --eval-tokens=4194304 --eval-every=-1 --save=false`

|    | toks/sec |     loss |      bpb | memory |  time | notes                      |
|----|---------:|---------:|---------:|-------:|------:|----------------------------|
| 5  |          |          |          |        |       |                            |
| 10 |          |          |          |        |       |                            |
| 15 |          |          |          |        |       |                            |
| 20 |          |          |          |        |       |                            |
| 25 |          |          |          |        |       |                            |
| 28 |   57,183 | 8.882510 | 2.109818 |  17.7g | 3.04m | faster dgrad epilogue      |
| 29 |   58,006 | 8.882510 | 2.109818 |  18.0g | 2.98m | undervolt                  |
| 30 |   59,161 | 8.834047 | 2.109956 |  18.0g | 2.93m | --rank-micro-steps=126,130 |
| 25 |          |          |          |        |       |                            |

Run 29: The GPU allows minor adjustments to its clock via undervolting. This raises the clock about 200-300 MHz.
`tools/adjust_voltage.sh` enabled by default going forward.

Run 30: GPU0 is about 5-10% slower. This gives a few extra micro-steps to GPU1 so that it's waiting less.
`--rank-micro-steps=126,130` enabled by default going forward.

# quick runs @ d24, 100 iterations

`--depth=24 --device-batch-size=2 --rank-micro-steps=126,130 --num-iterations=100 --eval-tokens=4194304 --eval-every=-1 --save=false`

| run | toks/sec |     loss |      bpb | memory |   time | notes                                                              |
|-----|---------:|---------:|---------:|-------:|-------:|--------------------------------------------------------------------|
| 32  |   58,627 | 4.471248 | 1.328176 |  17.6g | 26.43m | mxfp8 baseline                                                     |
| 33  |   71,226 | 4.482675 | 1.331916 |  17.0g | 21.80m | --nvfp4-wgrad --nvfp4-dgrad                                        |
| 34  |   70,307 | 4.477348 | 1.330228 |  17.0g | 22.12m | --nvfp4-wgrad --nvfp4-dgrad --nvfp4-eden=dgrad                     |
| 35  |   80,441 | 4.488634 | 1.333021 |  16.7g | 19.30m | --nvfp4-wgrad --nvfp4-dgrad --nvfp4-fwd                            |
| 36  |   71,108 | 4.484975 | 1.332725 |  17.0g | 21.18m | --nvfp4-wgrad --nvfp4-dgrad --nvfp4-fwd --nvfp4-fwd-until=warmdown |
| 37  |   71,049 | 4.413067 | 1.308297 |  17.0g | 21.86m | run 33 with --seed=43                                              |
| 38  |   70,238 | 4.411165 | 1.307771 |  17.0g | 22.13m | run 34 with --seed=43                                              |
| 39  |   71,441 | 4.424313 | 1.311929 |  16.8g | 21.69m | run 33 with --muon-fused                                           |

Run 36: NVFP4 forward for steps 0-34 (80,457 tok/s), MXFP8 forward from the warmdown at step 35 (71,160 tok/s).

Runs 37-38: the seed (init + stochastic rounding, same data order) moves bpb by 0.024 at 100 steps, so only same-seed
pairs compare. EDEN vs SR dgrad: -0.0017 (seed 42), -0.0005 (seed 43) for -1.2% tok/s.

Run 39: fused Muon, +0.3% tok/s, -0.020 bpb vs run 33 at the same seed. Not bit-identical, so at d24 the trajectory
diverges like a seed change: medium-d12 seed pairs (cache/metrics/muon) show the fused update +0.0028 (seed 42) and
+0.0023 (seed 43) worse.

# full run @d12

`--depth=12 --device-batch-size=8`

|    | toks/sec |     loss |      bpb | memory |   time | notes                                   |
|----|---------:|---------:|---------:|-------:|-------:|-----------------------------------------|
| 5  |          |          |          |        |        |                                         |
| 10 |          |          |          |        |        |                                         |
| 15 |          |          |          |        |        |                                         |
| 20 |          |          |          |        |        |                                         |
| 31 |  351,321 | 2.860658 | 0.879813 |   8.0g | 41.75m | --nvfp4-wgrad --nvfp4-dgrad --nvfp4-fwd |

# full run @d24

python baseline

- CORE: 0.256525
- bpb: 0.71800

`--depth=24 --device-batch-size=2 --save-every=250 --eval-every=250`

|    | toks/sec |    loss |     bpb | CORE   | memory |     time | notes                                                              |
|----|---------:|--------:|--------:|--------|-------:|---------:|--------------------------------------------------------------------|
|    |   81,303 | 2.36295 | 0.74283 | 0.2270 |  17.1g | 1194.49m | --nvfp4-wgrad --nvfp4-dgrad --nvfp4-fwd --rank-micro-steps=126,130 |
|    |   59,978 | 2.41296 | 0.71934 | 0.2616 |  18.0g | 1622.59m | --rank-micro-steps=126,130                                         |
|    |          |         |         |        |        |          |                                                                    |
