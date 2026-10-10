# todo

- ensure minimal instruction cache
- get rid of unused flags/code
- we don't need to be bit-identical to python
- take a close look at each hot kernel code
- run --attention=mx|fa2|bf16|bf16mx full d12 equal-time pairs
- FA warp specialization
- fix mfu for rtx pro 6000 and rtx 5090
- try mxfp8 with muon gather/reduce
- try sm120f (GB10 compatibility)

# done

- remove nvfp4 code
- add hours to eta: 1h 34.2m
- why does it sometimes rebuild even with no code changes?
- reduce compile noise
- speed up builds (skip building tests)
