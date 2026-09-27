// Embedding backward straight into .grad (no torch headers, so nvcc stays out of libtorch).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// Tables that share the token ids: upstream gradients g (n, cols) and the accumulated gradients grad (vocab, cols),
// bf16, contiguous, 16-byte aligned, cols % 8 == 0.
struct EmbeddingGrads {
  static constexpr int kMaxTables = 16;
  static constexpr int kMaxCols = 2048;
  const void* g[kMaxTables];
  void* grad[kMaxTables];
  int cols[kMaxTables];
  int tables;
};

// grad[id] += sum of g's rows with that id, per table. ids: the n token ids sorted (stably), order: their positions.
// Each touched row sums in fp32 in position order and is rounded once; untouched rows are not read or written.
void embedding_grad_accumulate(
      const int64_t* ids, const int64_t* order, int64_t n, const EmbeddingGrads& a, cudaStream_t stream);

} // namespace nanochat::kernels
