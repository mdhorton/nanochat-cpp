## LLM agent CUDA kernels

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) asked: Can LLMs Write GPU Kernels?

The development time saving are enormous. Handwritten kernels take weeks to write and tune. An LLM can generate a dozen
high performance kernels in a few hours. This includes tests, benchmarks, and tuning with nsys/ncu.

So the answer to the question is yes, but problem is more of
a [verification issue](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels).

For this project, I'm going to side-step this and treat the final result as verification. After all, this is an
experimental learning project. Does the trained model's GPT-2 CORE score pass the threshold?

## goal with 2x RTX Pro 4000 blackwell GPUs (sm120)

nanochat python trains ~5.84B tokens in 99 minutes using 8x H100.

On paper an H100 is roughly 6x faster than a RTX Pro 4000. Thus, 8x H100 are roughly 24x faster than 2x RTX Pro 4000.
Using that ratio we get the following target:

```~5.84B tokens in 2376 minutes (39.6 hours) = ~41k toks/sec```

The same GPT-2 CORE threshold will be used: 0.256525

## RTX Pro 4000 Blackwell == 145 watts

Pre-training workload is heavily power-bound and these cards have a low power cap (145 watts). They boost to 3000 GHz,
but most large dense GEMMs run at ~1700 GHz due to the power cap.

Improved kernel efficiency can mean the same work in less time. (Same energy in less time.) This translates to a lower
GPU clock because power is already at the cap. A 2-3% isolated step improvement is sometimes hard to distinguish from
noise due to the power cap eating a chunk of the gain.

On the other hand, kernel fusing is a double win. 1) Fewer bytes are transferred, and 2) less time. Less energy in less
time. This doesn’t have the power cap clock tax that improved kernel efficiency does.
