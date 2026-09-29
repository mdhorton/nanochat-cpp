# model config for RTX Pro 4000 Blackwell

vocabulary 32k

| n_layer (depth)                      |   12 |   24 |
|--------------------------------------|-----:|-----:|
| n_embd (auto computed: depth x 64)   |  768 | 1536 |
| n_head (auto computed: n_embd / 128) |    6 |   12 |
| --device-batch-size                  |    8 |    2 |
| --max-seq-len                        | 2048 | 2048 |
| --target-param-data-ratio            |    8 |    8 |

# notes

At `--depth=24` setting `--device-batch-size=4` increased performance ~2%. However, it pushed vram usage over 98%. And
this was after tuning memory usage. I decided it wasn't worth the change on my GPUs. I might revisit it later after
development stabilises.
