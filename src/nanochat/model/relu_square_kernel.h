// Fused relu(h).square() and its backward (no torch headers, so nvcc stays out of libtorch).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// h, a: n bf16 values, 16-byte aligned. a = relu(h)^2, rounded as the op path.
void relu_square_fwd(const void* h, void* a, int64_t n, cudaStream_t stream);

// dh = h > 0 ? 2 * h * g : 0 (bf16, bit-identical to the op path). amax: device float or null; when given, set to
// max|dh| for its FP8 quantization.
void relu_square_bwd(const void* g, const void* h, void* dh, float* amax, int64_t n, cudaStream_t stream);

} // namespace nanochat::kernels
