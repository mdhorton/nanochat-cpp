// Fused tensorwise FP8 quantization (no torch headers). Same math as fp8.py's _to_fp8, bit for bit:
// amax = max|x|, scale = float(fp8_max * (1 / max(double(amax), 1e-12))), q = fp8(clamp(float(x) * scale)), inv = 1 /
// scale.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

enum class Fp8Format { E4M3, E5M2 };

// x: (rows, cols) contiguous bf16 (x_bf16 = true) or fp32. amax: device float scratch. Writes *inv_scale (device
// float), and out (rows, cols) and/or out_t (cols, rows, the transpose) when non-null.
void quantize_fp8(
      const void* x, bool x_bf16, int64_t rows, int64_t cols, Fp8Format format, void* out, void* out_t, float* amax,
      float* inv_scale, cudaStream_t stream);

} // namespace nanochat::kernels
