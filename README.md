# introduction

This continues the saga of exploring sm120 GPUs. Specifically 2x RTX Pro 4000, which is my local setup. With the added
twist of c++.

# pytorch vs LLM agent CUDA kernels

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) asked: Can LLMs Write GPU Kernels?

The answer seems to be yes. The problem becomes more of
a [verification issue](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels).

For this project, I'm going to side-step this and treat the final result as verification. (This is after all an
experimental learning project.) Does the trained model's GPT-2 CORE score pass the threshold?

# porting nanochat to c++

My GPUs, 2x RTX Pro 4000, are sm120. Triton does not optimise these as blackwell. Rightly so because they lack important
blackwell datacenter features.

However, sm120 does have a couple of relevant features that could improve performance and increase model quality.

- mxfp8
- nvfp4

One doesn't need to port a python project to c++ just to access these features. But I was curious all the same.

The original nanochat is pytorch without custom triton kernels.

I was impressed to find that Opus 5.5 (high) took less than **40 minutes** to translate the bulk of the pre-training
code from python to c++.

This is probably because pytorch uses libtorch, the c++ library. The main exception is `torch.compile`, which includes
inductor.

# goal with 2x RTX Pro 4000 blackwell GPUs (sm120)

nanochat (python) trains ~5.84B tokens in 99 minutes using 8x H100. We have to assume nanochat is properly tuned. The
current record has not been improved upon in several months.

On paper an H100 is roughly 6x faster than a RTX Pro 4000.
Thus, 8x H100 are roughly 24x faster than 2x RTX Pro 4000. Using that ratio we get the following target:

|    |     |
|----|-----|
| 6x | 41k |
| 5x | 49k |
| 4x | 61k |

```~5.84B tokens in 2376 minutes (39.6 hours) = ~41k toks/sec```

The same GPT-2 CORE threshold will be used: 0.256525

A secondary goal would be to train within 24 hours (1440 minutes). This is an arbitrary goal and will be tough to
achieve.

```~5.84B tokens in 1440 minutes = ~68k toks/sec```

# RTX Pro 4000 Blackwell == 145 watts

The workload is heavily power-bound and these cards have a low power cap. They boost to 3000 GHz, but most kernels run
below 2000 GHz due to the power cap. A chiller wouldn't help much because they usually don't thermally throttle.

Improved kernel efficiency can mean the same work (energy) in less time. This translates to a lower GPU clock because
power is usually already at the cap. A 5% isolated benchmark gain might realise an actual 2% gain.

In contrast, fusing is a double win: fewer bytes are transferred (lower energy) in less time. Less energy in less
time does not tax the clocks like kernel efficiency (same energy in less time).
