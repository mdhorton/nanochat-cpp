## introduction

This is a **python → c++** port of Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat). With the added
twist that it targets Nvidia sm120 GPUs.

I'd classify this port as more of a performance engineering project. I own 2x RTX Pro 4000 Blackwell and my initial goal
was to see how fast it could run locally.

## sm120 GPUs

These are Blackwell non-datacenter GPUs such as RTX Pro 6000, RTX Pro 4000, RTX 5090, etc... They lack important
features compared to datacenter GPUs. For example:

- no nvlink
- no wgmma
- 50% less shared memory
- lower power cap (RTX Pro 4000 == 145 watts)
- lower vram (RTX Pro 4000 == 24GB)

FA3+ requires datacenter GPUs. Thus, sm120 GPUs are limited to FA2. Also, the RTX non-Pro line (eg, RTX 5090) does not
support P2P.

However, sm120 has MXFP8 and NVFP4, which the H100 does not have. So we'll see if this helps.

## TLDR results

| GPU            | GPU Count |     bpb |   CORE |                  time | device-batch-size |       
|----------------|----------:|--------:|-------:|----------------------:|------------------:|
| RTX Pro 6000 S |         8 | 0.71773 | 0.2592 |  139.96m ( 2h:19.96m) |                 8 |
| RTX 5090       |         8 | 0.71769 | 0.2615 |  148.85m ( 2h:28.85m) |                 4 |
| RTX Pro 4000   |         2 | 0.71934 | 0.2616 | 1622.59m (27h:02.59m) |                 2 |

These runs used `--depth=24 --target-param-data-ratio=8`. They kept the same model config as the python version. CORE
was calculated using the original python code.

## performance history change summary

There were more changes than this, but these were the keepers. Each can be run via `pixi run medium-d24` along with the
flags from the table.

Row #10 is the current default. Technically those flags could be omitted. I laid out all the flags for each row to make
it clear what was being enabled or not.

| row |      bpb | memory |     time | notes                        | flags                                                                                                                                                                                  |
|----:|---------:|-------:|---------:|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   1 | 1.335750 |  39.3g | 04:31.38 | baseline                     | `--muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=tensorwise --fused=false --loss-chunk-rows=0 --attention=sdpa --window-pattern=L` |
|   2 |          |        |          | enable torch FA2 and SSSL    | `--muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=tensorwise --fused=false --loss-chunk-rows=0 --attention=fa2`                     |
|   3 |          |        |          | fused & chunked lm_head loss | `--muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=tensorwise --fused=false --loss-chunk-rows=-1 --attention=fa2`                    |
|   4 |          |        |          | fused many more kernels      | `--muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=tensorwise --fused=true --loss-chunk-rows=-1 --attention=fa2`                     |
|   5 |          |        |          | mxfp8                        | `--muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=fa2`                          |
|   6 |          |        |          | mxfp8 attention              | `--muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx`                           |
|   7 |          |        |          | mxfp8 gemm backend           | `--muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cutlass --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx`                          |
|   8 |          |        |          | overlap gathers with forward | `--muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=true --gemm=cutlass --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx`                           |
|   9 |          |        |          | bf16 muon gradient reduce    | `--muon-bf16-gather=false --muon-bf16-reduce=true --gather-overlap=true --gemm=cutlass --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx`                            |
|  10 | 1.318522 |  33.1g |  2:10.37 | bf16 muon update gather      | `--muon-bf16-gather=true --muon-bf16-reduce=true --gather-overlap=true --gemm=cutlass --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx`                             |

## initial port

Opus 5.5 (high effort) ported the nanochat pre-training code from python to c++ in under 40 minutes. Couple reason why
this was so quick. The original code is well-designed. Also, pytorch uses libtorch, which is a c++ library. This made
some of the translation to c++ straightforward.

However, libtorch does not have `torch.compile` or inductor. So the baseline performance is not great. This starts the
performance engineering work to make it faster with sm120.

## tuning

Most of the sm120 tuning fell into the following categories:

- kernel fusing
- overlapping compute with collectives
- mxfp8
- flash attention
- memory tuning

## setup

### requirements

- sm120 capability GPU (RTX Pro 6000/4000, RTX 5090, etc...)
- nvidia driver supporting cuda 13+
- [pixi](https://prefix.dev/) package manager

I've only tested on Linux Ubuntu 24.04.

### initial setup

```bash
# install pixi if needed
curl -fsSL https://pixi.sh/install.sh | bash

# download 10 parquet files
pixi run dataset 10

# train the BPE tokenizer
pixi run tok-train
```

### execute some runs

```bash
# execute a short (30 step) --depth=12 training run
pixi run quick-d12

# execute a slightly longer (100 step) --depth=12 training run
pixi run medium-d12

# execute a full --depth=24 training run and use wandb (requires WANDB_API_KEY or wandb login)
pixi run full-d24 --run=full-d24-8x6000 --wandb

# execute a customized training run
pixi run base-train --depth=26 --device-batch-size=16 --eval-every=1000 --save-every=1000 --run=full-d26 --wandb
```

The `quick-, medium-, full-` prefixes use preset flags for convenience (see `pixi.toml` for the presets). All preset
flags can be overridden on the command line. Or use `pixi run based-train` for no preset flags.

NOTE: `--fp8` is enabled by default. To disable use `--fp8=false`. This will also disable mxfp8 related flags.

## why c++?

Honestly, mostly curiosity. The core code runs on the GPU, so there's little if any performance benefit from c++.

At the outset I was curious how long it would take Claude to port nanochat to c++. Then I just kept going. I didn't hit
any significant roadblocks and saw no reason to stop.

## notes

A few more [notes here](NOTES.md).

## License

MIT
