# introduction

This continues the saga of exploring sm120 GPUs. Specifically 2x RTX Pro 4000, which is my local setup. With the added
exciting twist of c++.

# pytorch inductor generated triton kernels vs LLM agent CUDA kernels

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) asked: Can LLMs Write GPU Kernels?

The answer seems to be yes. Not only can it write them, it can write performant kernels in a fraction of the time it
takes a human.

The problem becomes more of a [verification issue](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels).
For this project, I'm going to side-step this and treat the final result as verification. (This is after all an
experimental learning project.) Does the trained model's CORE score pass the threshold?

# porting nanochat to c++

My GPUs, 2x RTX Pro 4000, are sm120. Triton does not optimize these as blackwell. Rightly so because they lack important
datacenter blackwell features.

However, sm120 does have a couple of relevant features that can improve performance and model quality.

- mxfp8
- nvfp4 (inference)

One doesn't need to port a python project to c++ just to access these features. But I was curious all the same.

I was stunned to find that Opus 5.5 (high) took less than **40 minutes** to translate the bulk of the pre-training code
from python to c++.

This is probably because libtorch, the c++ library, is what pytorch uses. The main exception is `torch.compile`, which
includes inductor fusion.

# goal with 2x RTX Pro 4000 blackwell GPUs (sm120)

nanochat (python) trains ~5.84B tokens in 99 minutes using 8x H100. An H100 is roughly 6x faster than a RTX Pro 4000.
Thus, 8x H100 are roughly 24x faster than 2x RTX Pro 4000. Using that ratio we get the following target:

```~5.84B tokens in 2376 minutes (39.6 hours) = ~41k toks/sec```

The same GPT-2 CORE threshold will be used: 0.256525

A secondary goal would be to train within 24 hours (1440 minutes). This is an arbitrary goal and will be tough to
achieve.

```~5.84B tokens in 1440 minutes = ~68k toks/sec```
