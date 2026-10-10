## LLM agent CUDA kernels

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) asked: Can LLMs Write GPU Kernels?

The development time savings are enormous. Handwritten kernels take weeks to write and tune. An LLM can generate a dozen
high performance kernels in a few hours. This includes tests, benchmarks, and tuning with nsys/ncu.

So the answer is yes, but problem is more of
a [verification issue](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels).

For this project, I'm going to side-step this and treat the final result as verification. After all, this is an
experimental learning project. Does the trained model's GPT-2 CORE score pass the `0.256525` threshold?

## rental performance

The following are worth checking out on rented hosts:

- collectives performance (PCIe)
- power cap
- cooling

Run `pixi run nccl-bench` to get an idea of collectives performance. Substandard PCIe can easily cut performance in
half. For PCIe 5x16, a solid host will report ~25+ GB/sec. ~20 GB/sec is usually acceptable. ~10-15 GB/sec is probably
the bare minimum.

Some hosts intentionally lower the power cap. Sometimes for thermal reasons. This is worth a quick check via
`nvidia-smi`. Also, if temps regularly exceed 80C then thermal throttling is a risk.

If any of these 3 items is subpar, then the performance will suffer.

## goal with 2x RTX Pro 4000 blackwell GPUs (sm120)

2x RTX Pro 4000 is my local setup and where most of the tuning will be done.

nanochat python trains ~5.84B tokens in 99 minutes using 8x H100. On paper an H100 is roughly 6x faster than an RTX
Pro 4000. Thus, 8x H100 are roughly 24x faster than 2x RTX Pro 4000. Using that ratio we get the following minimum
target:

```~5.84B tokens in 2376 minutes (39.6 hours) = ~41k toks/sec```

## RTX Pro 4000 Blackwell = 145 watts

Pre-training workload is heavily power-bound and these cards have a low power cap (145 watts). They boost to 3000 GHz,
but most large dense GEMMs run at ~1700 GHz due to the power cap.

Improved kernel efficiency can mean the same work in less time. (Same energy in less time.) This translates to a lower
GPU clock on these cards because power is already at the cap. A 2-3% isolated step improvement is sometimes hard to
distinguish from noise due to the power cap eating a chunk of the gain.

On the other hand, kernel fusing is a double win. 1) Fewer bytes are transferred, and 2) less time. Less energy in less
time. This doesn’t have the power cap clock tax that improved kernel efficiency does on these cards.
