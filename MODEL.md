# model config

| n_layer (depth)       |  12 |   24 |
|-----------------------|----:|-----:|
| n_embd (depth x 64)   | 768 | 1536 |
| n_head (n_embd / 128) |   6 |   12 |
| --device-batch-size   |   8 |    4 |

vocab: 32k

--max-seq-len 2k
--target-param-data-ratio 8
