// Fused relu(h).square(): one CUDA kernel each way instead of 2 and ~4, saving h instead of relu(h).
#pragma once

#include <torch/torch.h>

namespace nanochat {

// h: contiguous bf16. Bit-identical to torch::relu(h).square(), forward and backward.
torch::Tensor relu_square(const torch::Tensor& h);

} // namespace nanochat
