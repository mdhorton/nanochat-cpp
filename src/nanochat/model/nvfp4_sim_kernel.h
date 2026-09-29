// NVFP4 fake quantization (quantize, then dequantize to bf16) for simulating NVFP4 training numerics with bf16 GEMMs
// (no torch headers, so nvcc stays out of libtorch).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// x (rows, cols) row-major, bf16 or fp32 -> out (rows, cols) bf16: each 16 values along a row become e2m1 times a ue4m3
// block scale times the fp32 tensor scale (amax / (6 * 448)), then dequantized. The values are exact in bf16, so a bf16
// GEMM with fp32 accumulation computes the NVFP4 product. amax: device float, max|x| (or of the tensor the blocks come
// from). block_amax (optional): (rows, cols / 16) device floats used as the blocks' amax (2D blocks). stochastic:
// stochastic rounding to e2m1 from seed, else round to nearest even. cols % 16, pointers 16-byte aligned.
void nvfp4_fake_quant(
      const void* x, bool bf16, int64_t rows, int64_t cols, const float* amax, const float* block_amax, bool stochastic,
      uint64_t seed, void* out, cudaStream_t stream);

} // namespace nanochat::kernels
