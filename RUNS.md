# medium runs @ d12 (initial port)

Runs based on the initial nanochat port to c++. No tuning yet.

```
--depth=12 --device-batch-size=8 --num-iterations=100 --eval-tokens=4194304 --eval-every=-1 --save=false
```

|   | toks/sec |     loss |      bpb | memory |   time | notes                         |
|--:|---------:|---------:|---------:|-------:|-------:|-------------------------------|
| 1 |   36,192 | 4.854269 | 1.439428 |  17.0g | 24.11m | --nproc=1                     |
| 2 |   72,305 | 4.800026 | 1.434599 |  16.4g | 12.08m | --nproc=2 (ZeRO-2, MuonAdamW) |
| 3 |   80,129 | 4.820226 | 1.441378 |  16.0g | 10.91m | --nproc=2 --fp8               |

# medium runs @ d12 (tuned)

Runs based on the sm120 customizations and tuning.

```
--fp8 --nproc=2 --depth=12 --device-batch-size=8 --num-iterations=100 --eval-tokens=4194304 --eval-every=-1 --save=false
```

|    | toks/sec |     loss |      bpb | memory |  time | notes                         |
|---:|---------:|---------:|---------:|-------:|------:|-------------------------------|
|  4 |  107,091 | 4.820269 | 1.441330 |  16.0g | 8.15m | --attention=fa2               |
|  5 |  138,679 | 4.804933 | 1.436173 |   8.2g | 6.29m | --loss-chunk-rows=4096        |
|  6 |  155,024 | 4.808020 | 1.437057 |   8.2g | 5.63m | fused rotary + QK-norm        |
|  7 |  158,808 | 4.807032 | 1.436761 |   8.2g | 5.49m | fused resid/x0 + λ-blend      |
|  8 |  181,216 | 4.803973 | 1.435802 |   8.2g | 4.82m | fused relu² with amax         |
|  9 |  183,137 | 4.806090 | 1.436472 |   8.2g | 4.77m | fused residual add + rms_norm |
| 10 |  188,796 | 4.807217 | 1.436816 |   7.9g | 4.62m |                               |
| 10 |          |          |          |        |       |                               |

Run 4: sm120 can't use fa3. however, fa2 works pretty good. --attention=fa2 enabled by default going forward.

Run 5: python nanochat keeps several GB of logits alive at once. this triggers OOM at --depth=24 on my setup. chunk and
fuse CE loss to reduce vram usage. --loss-chunk-rows=4096 enabled by default going forward.

Runs 6-9: sm120 have lower memory bandwidth than datacenter GPUs. for example, my RTX Pro 4000 has ~6x slower dram
bandwidth vs H100. fusing memory bound kernels is usually worth it. this is also the parts normally covered by pytorch
inductor.

# medium runs @ d24

These runs didn't start until chunked CE loss was added (run 5 with d12).

```
--fp8 --nproc=2 --depth=24 --device-batch-size=2 --num-iterations=30 --eval-tokens=4194304 --eval-every=-1 --save=false
```

|   | toks/sec |     loss |      bpb | memory |  time | extra args or overrides |
|--:|---------:|---------:|---------:|-------:|------:|-------------------------|
| 1 |   29,736 | 8.841636 | 2.107628 |  18.8g | 5.85m | based on d12 run 5      |
| 2 |          |          |          |        |       |                         |

# full run @d12

```
--fp8 --nproc=2 --depth=12 --device-batch-size=8
```

| steps | loss     | bpb      | time    |
|-------|----------|----------|---------|
| 2520  | 2.814477 | 0.847031 | 178.26m |
|       |          |          |         |