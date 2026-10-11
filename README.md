## Intro

This is a **c++** port of Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat). With the added
twist that it targets nvidia sm120 GPUs.

I'd classify this port as more of a performance engineering project. I own 2x RTX Pro 4000 Blackwell and my goal was to
see how fast it could run locally.

## sm120 GPUs

These are Blackwell non-datacenter GPUs such as RTX Pro 6000, RTX Pro 4000, RTX 5090, etc... They lack important
features compared to datacenter GPUs. For example:

- no nvlink
- no wgmma
- 50% less shared memory
- lower power cap (RTX Pro 4000 = 145 watts)
- lower vram (RTX Pro 4000 = 24GB)

FA3+ requires datacenter GPUs. Thus, sm120 GPUs are limited to FA2. Also, the RTX non-Pro line (eg, RTX 5090) does not
support P2P.

However, sm120 has MXFP8 and NVFP4, which the H100 does not have. So we'll see if this helps.

## TLDR

### Pre-training results

| GPU            | GPUs | device batch size |     bpb |   CORE |     time |    
|----------------|-----:|------------------:|--------:|-------:|---------:|
| RTX Pro 6000 S |    8 |                 8 | 0.71773 | 0.2592 |  135.07m |
| RTX 5090       |    8 |                 4 | 0.71769 | 0.2615 |  148.85m |
| RTX Pro 4000   |    2 |                 2 | 0.71934 | 0.2616 | 1622.59m |

These runs used `--depth=24 --target-param-data-ratio=8`. The same model config was used as the python version. CORE
metric was calculated using the original python code.

## Performance history change summary

These were the keepers. These can be run with `pixi run medium-d24` along with the row flags.

These were 100-steps @ depth 24 run on rented 8x RTX Pro 6000 S.

| row |      bpb | memory |  time | notes                        | flags                                                                                                                                                             |
|----:|---------:|-------:|------:|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   1 | 1.333600 |  39.3g | 4.37m | baseline                     | --muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=tensorwise --fused=false --loss-chunk-rows=0 --attention=fa2  |
|   2 | 1.333295 |  33.5g | 4.01m | fused & chunked lm_head loss | --muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=tensorwise --fused=false --loss-chunk-rows=-1 --attention=fa2 |
|   3 | 1.329447 |  33.0g | 2.80m | fused many more kernels      | --muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=tensorwise --fused=true --loss-chunk-rows=-1 --attention=fa2  |
|   4 | 1.326894 |  33.0g | 2.58m | mxfp8                        | --muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=fa2       |
|   5 | 1.327575 |  33.1g | 2.48m | mxfp8 attention              | --muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cublas --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx        |
|   6 | 1.327575 |  33.1g | 2.31m | mxfp8 backend cutlass gemms  | --muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=false --gemm=cutlass --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx       |
|   7 | 1.326927 |  33.1g | 2.24m | overlap gathers with forward | --muon-bf16-gather=false --muon-bf16-reduce=false --gather-overlap=true --gemm=cutlass --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx        |
|   8 | 1.328304 |  33.1g | 2.19m | bf16 muon gradient reduce    | --muon-bf16-gather=false --muon-bf16-reduce=true --gather-overlap=true --gemm=cutlass --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx         |
|   9 | 1.328021 |  33.1g | 2.17m | bf16 muon update gather      | --muon-bf16-gather=true --muon-bf16-reduce=true --gather-overlap=true --gemm=cutlass --fp8-recipe=mxfp8 --fused=true --loss-chunk-rows=-1 --attention=mx          |

Row 2 was required to get `--depth=24` working on RTX Pro 4000.

Row 3 was needed because c++ doesn't have Inductor. (However, we have an LLM agent that can fuse kernels.) This was the
bulk of the speedup.

Rows 4-6 utilises sm120 features.

Rows 7-9 target the lack of nvlink. Move less data and overlap it with compute.

Row 9 is the current default. Technically those flags can be omitted. I laid out all the flags just to make it clear
what was enabled/disabled.

## Initial port

Opus 5.5 (high effort) ported the nanochat pre-training code from python to c++ in under 40 minutes. Couple reasons why
this was so quick. The original code is well-designed. Also, pytorch uses libtorch, which is a c++ library. This made
some of the translation to c++ straightforward.

However, libtorch does not have `torch.compile` or Inductor. So the baseline performance is not great. This is where the
performance engineering work starts to make it faster with sm120.

## Tuning

Most of the sm120 tuning fell into the following categories:

- memory
- kernel fusing
- flash attention
- mxfp8
- overlapping compute with collectives

## Setup

Everything runs as the current user. root is not needed.

### Requirements

- sm120 capability GPU (RTX Pro 6000/4000, RTX 5090, etc...)
- nvidia driver supporting cuda 12.8+ (12.8+ uses torch 2.11; 13+ uses torch 2.14)
- pixi

I've only tested with Linux Ubuntu 24.04.

### pixi

The [pixi package manager](https://pixi.prefix.dev/latest/) is similar to `uv` but works with c++. The following command
will install pixi:

`curl -fsSL https://pixi.sh/install.sh | bash`

Now one of the following is needed to add pixi to PATH:

- logout and login again
- restart your terminal or shell
- `export PATH="$HOME/.pixi/bin:$PATH"`

### Initial 1-time setup

```bash
# setup dependencies (similar to `uv sync`).
pixi install

# build the executables (takes 1-2 minutes)
pixi run build

# download 10 parquet files
pixi run dataset 10

# train the BPE tokenizer
pixi run tok-train

# optional tests (takes 2-3 minutes)
pixi run test
```

### Execute some runs

```bash
# execute a short (30 step) --depth=12 training run
pixi run quick-d12

# execute a short (30 step) --depth=24 training run
pixi run quick-d12

# execute a full --depth=24 training run and use wandb (requires WANDB_API_KEY or wandb login)
# run `pixi run dataset 170` if needed to fetch enough parquet files.
pixi run full-d24 --run=full-d24-8x6000 --wandb

# execute a customized training run
pixi run base-train --depth=26 --device-batch-size=16 --eval-every=1000 --save-every=1000 --run=full-d26 --wandb
```

The `quick-, medium-, full-` prefixes use preset flags for convenience (see `pixi.toml` for the presets). All preset
flags can be overridden on the command line. Or use `pixi run based-train` for no preset flags.

NOTE: `--fp8` is enabled by default. Use `--fp8=false` to disable. This will also disable mxfp8 related flags.

## Why c++?

Honestly, mostly curiosity. The core code runs on the GPU, so there's little if any performance benefit from c++.

At the outset I was curious how long it would take Claude to port nanochat to c++. Then I just kept going. I didn't hit
any significant roadblocks and saw no reason to stop.

## Notes

A few more [notes here](NOTES.md).

## License

MIT
