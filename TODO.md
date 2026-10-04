# todo

- ensure minimal instruction cache
- get rid of unused flags/code
- we don't need to be bit-identical to python
- take a close look at each hot kernel code
- run --attention=mx|fa2|bf16|bf16mx full d12 equal-time pairs
- FA warp specialization

# done

- remove nvfp4 code
- add hours to eta: 1h 34.2m
- only use full tokens for final bpb validation
- why does it rebuild without changes sometimes?
- reduce compile noise
- speed up builds (skip building tests)
