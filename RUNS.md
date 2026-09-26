# quick runs @ d12 (initial port)

Runs based on the initial port from python to c++. No tuning yet.

```
--depth=12 --device-batch-size=8 --num-iterations=30 --eval-tokens=4194304 --eval-every=-1 --save=false
```

|   | toks/sec |     loss |      bpb | memory |  time | extra args or overrides       |
|--:|---------:|---------:|---------:|-------:|------:|-------------------------------|
| 1 |   36,013 | 6.415645 | 1.780948 |  17.0g | 7.27m | --nproc=1                     |
| 2 |   71,778 | 6.423695 | 1.778126 |  16.4g | 3.65m | --nproc=2 (ZeRO-2, MuonAdamW) |
| 3 |   79,374 | 6.428883 | 1.779265 |  16.0g | 3.30m | --nproc=2 --fp8               |

# quick runs @ d12

Runs based on the sm120 customizations.

```
--fp8 --nproc=2 --depth=12 --device-batch-size=8 --num-iterations=30 --eval-tokens=4194304 --eval-every=-1 --save=false
```

|   | toks/sec |     loss |      bpb | memory |  time | extra args or overrides |
|--:|---------:|---------:|---------:|-------:|------:|-------------------------|
| 4 |  106,614 | 6.428749 | 1.779237 |  16.0g | 2.46m | --attention=fa2         |
| 5 |  137,748 | 6.423904 | 1.778217 |   8.2g | 1.90m | --loss-chunk-rows=4096  |
| 6 |  153,782 | 6.424578 | 1.778360 |   8.2g | 1.70m | --fused=true            |

Run 4: sm120 can't use fa3. however, fa2 works pretty good. --attention=fa2 enabled by default going forward.

Run 5: the python version keeps several GB of logits alive at once. depth=24 triggers OOM on my setup. this run chunks
and fuses the CE loss, which reduces vram. --loss-chunk-rows=4096 enabled by default going forward.

# quick runs @ d24

These runs didn't start until chunked CE loss was added (run 5 with d12).

```
--fp8 --nproc=2 --depth=24 --device-batch-size=2 --num-iterations=10 --eval-tokens=4194304 --eval-every=-1 --save=false
```

|   | toks/sec |     loss |      bpb | memory |  time | extra args or overrides |
|--:|---------:|---------:|---------:|-------:|------:|-------------------------|
| 1 |   29,736 | 8.841636 | 2.107628 |  18.8g | 5.85m |                         |
| 2 |          |          |          |        |       |                         |

# full run @d12

```
--fp8 --nproc=2 --depth=12 --device-batch-size=8
```

| steps | loss     | bpb      | time    |
|-------|----------|----------|---------|
| 2520  | 2.814477 | 0.847031 | 178.26m |
|       |          |          |         |