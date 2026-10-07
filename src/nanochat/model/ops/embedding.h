// Embedding lookups of several tables with the same token ids; backward accumulates straight into each table's .grad.
#pragma once

#include <torch/torch.h>

namespace nanochat {

// Returns weights[i][idx] for each table (bf16, CUDA). Backward sorts idx once and adds each touched row's gradient
// into weights[i].grad() in place (zeros first when undefined), summed in fp32 and rounded once, deterministic; it
// returns no gradient, so the tables' autograd accumulation (a full-size zero fill and add per call) is skipped.
std::vector<torch::Tensor> embeddings(const torch::Tensor& idx, const std::vector<torch::Tensor>& weights);

} // namespace nanochat
