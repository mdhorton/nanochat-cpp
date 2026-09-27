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

This is probably because libtorch, the c++ library, is what pytorch uses. The main exceptions are torch.compile and
InductorFusion.

If the agent

This starts to blur the benefit of python vs c++. Thi

I love python and have used it for 10+ years. This isn't a knock on python. This project is just an experiment.

# goals

5.84B total tokens
99 minutes

After thinking about it, I decided on the following goals (or questions really):

1. how fast can claude port nanochat to c++?
2. how fast can we train the model using sm120?

# why c++ then?

To be honest, curiosity. This is first and foremost a learning project.

c++ has nothing to do with making nanochat work with sm120 GPUs. By itself, c++ won't speed things up.

Does the language even matter? With powerful LLMs, these porting exercises become almost trivial. Translation from one
language to another is one of LLMs strongest features.
