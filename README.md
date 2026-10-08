## introduction

This is a **python → c++** port of Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat). With the added
twist that it targets Nvidia sm120 GPUs.

I'd classify this as more of a performance engineering project. I own 2x RTX Pro 4000 and my initial goal was to see how
fast it would run locally.

## sm120 GPUs

These are Blackwell non-datacenter GPUs such as RTX Pro 6000, RTX Pro 4000, RTX 5090, etc... They lack important
features compared to datacenter GPUs such as H100. For example:

- no nvlink
- no wgmma
- 50% less shared memory
- lower power cap (RTX Pro 4000 == 145 watts)
- lower vram (RTX Pro 4000 == 24GB)

FA3+ focuses on datacenter GPUs. Thus, sm120 GPUs are limited to FA2.

Also, the RTX non-Pro line (eg, RTX 5090) does not support P2P and does not have ECC. Not having ECC is more of a
quality issue.

However, sm120 has MXFP8 and NVFP4, which H100 does not have.

## initial port

Opus 5.5 (high effort) ported the nanochat pre-training code from python to c++ in ~40 minutes. I believe one of the
reasons for the speed is the fact that pytorch uses libtorch, which is a c++ library.

However, libtorch does not have torch.compile or inductor. So the baseline performance is not great. This starts the
performance engineering work on making it faster with sm120.

## TLDR

8x RTX 5090

## setup

### requirements

- sm120 capability GPU (RTX Pro 6000/4000, RTX 5090, etc...)
- nvidia driver supporting cuda 13+
- [pixi](https://prefix.dev/) package manager

I've only tested on Linux Ubuntu 24.04.

### 1-time initial setup

```bash
# install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# download the parquet files (downloads 10 files by default)
pixi run dataset 

# or download more files (downloads 170 parquet files)
# pixi run dataset 170

# train the BPE tokenizer
pixi run tok-train
```

### execute some runs

```bash
# execute a 30 iteration --depth=12 pre-training run
pixi run quick-d12

# execute a 100 iteration --depth=12 pre-training run
pixi run medium-d12

# execute a full --depth=24 pre-training run and use wandb (requires WANDB_API_KEY or wandb login)
pixi run full-d24 --run=full-d24-super --wandb
```

### optional

A NCCL benchmark can help determine if the host is correctly configured and has sufficient GPU communication hardware
for pre-training. sm120 does not support nvlink. I've found that some rented hosts have poor PCI bus bandwidth, which
significantly impacts performance.

```bash
pixi run nccl-bench
```

# pytorch vs LLM agent CUDA kernels

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) asked: Can LLMs Write GPU Kernels?

The development time saving are enormous. Handwritten kernels can take weeks to write and tune. An LLM can generate a
dozen high performance kernels in a few hours. This includes tests, benchmarks, and tuning with nsys/ncu.

So the answer to the question is yes, but problem is more of
a [verification issue](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels).

For this project, I'm going to side-step this and treat the final result as verification. This is after all an
experimental learning project. Does the trained model's GPT-2 CORE score pass the threshold?

# porting nanochat to c++

My GPUs, 2x RTX Pro 4000, are sm120. Triton does not optimise these as blackwell. Rightly so because they lack important
blackwell datacenter features.

However, sm120 does have a couple of relevant features that could improve performance and increase model quality.

- mxfp8
- nvfp4

One doesn't need to port a python project to c++ just to access these features. But I was curious all the same.

The original nanochat is pytorch without custom triton kernels.

Claude Opus 5.5 (high) took less than **40 minutes** to translate the bulk of the pre-training code from python to c++.
This is probably because pytorch uses libtorch, which is a c++ library. The main exception is `torch.compile`, which
includes inductor.

This means the base translation will work, but perform poorly.

# goal with 2x RTX Pro 4000 blackwell GPUs (sm120)

nanochat (python) trains ~5.84B tokens in 99 minutes using 8x H100. We have to assume nanochat is properly tuned. The
current record has not been improved upon in 6+ months.

On paper an H100 is roughly 6x faster than a RTX Pro 4000. Thus, 8x H100 are roughly 24x faster than 2x RTX Pro 4000.
Using that ratio we get the following target:

```~5.84B tokens in 2376 minutes (39.6 hours) = ~41k toks/sec```

The same GPT-2 CORE threshold will be used: 0.256525

# RTX Pro 4000 Blackwell == 145 watts

The workload is heavily power-bound and these cards have a low power cap (145 watts). They boost to 3000 GHz, but most
large dense GEMMs run at ~1700 GHz due to the power cap. Water cooling wouldn't help much because they aren't thermally
throttled.

Improved kernel efficiency can mean the same work in less time. Same energy in less time. This translates to a lower GPU
clock because power is already at the cap. A 3% isolated step improvement is sometimes hard to distinguish from noise
due to the power cap eating a chunk of the gain.

On the other hand, kernel fusing is a double win. 1) Fewer bytes are transferred, and 2) less time. Less energy in less
time. This doesn’t have the power cap clock tax that improved kernel efficiency does.
