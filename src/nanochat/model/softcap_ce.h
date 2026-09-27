// lm_head + logit softcap + cross-entropy, computed a chunk of rows at a time so the full (N, vocab) logits never
// exist. A fused CUDA kernel does softcap, loss and the logits' gradient in one pass over each bf16 chunk.
#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace nanochat {

enum class LossReduction { Mean, Sum, None };

// x: (N, C) in the compute dtype; weight: (V_padded, C) lm_head weight (fp32, cast to x's dtype for the matmul);
// targets: (N) int64, -1 = ignored. Logits are cropped to vocab_size, softcapped as softcap * tanh(z / softcap)
// in fp32. Mean divides by the number of non-ignored targets, like F.cross_entropy. fp8: the three matmuls as
// fp8_matmul (fp8.h), but each chunk's logits gradient gets its own scale (row counts and dims % 16, else bf16).
torch::Tensor softcap_cross_entropy(
      const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& targets, int64_t vocab_size,
      double softcap, int64_t chunk_rows, LossReduction reduction = LossReduction::Mean, bool fp8 = false);

} // namespace nanochat
