#include "nanochat/model/qkv_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>

namespace nanochat {

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8; // bf16 per 16-byte load

using bf16 = __nv_bfloat16;

__device__ void add8(const bf16* p, float (&acc)[kVec]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const auto* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k) {
    const float2 f = __bfloat1622float2(b[k]);
    acc[2 * k] += f.x, acc[2 * k + 1] += f.y;
  }
}

__device__ __forceinline__ void qkv_grad_sum_body(
      const bf16* dq, const bf16* dk, const bf16* dv, const bf16* gate, int gate_cols, bf16* dx, int64_t rows,
      int cols) {
  const int64_t n_vec = rows * cols / kVec, stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t v = blockIdx.x * kThreads + threadIdx.x; v < n_vec; v += stride) {
    const int64_t i = v * kVec;
    float acc[kVec] = {};
    add8(dq + i, acc);
    add8(dk + i, acc);
    add8(dv + i, acc);
    const int col = static_cast<int>(i % cols);
    if (gate != nullptr && col < gate_cols) {
      const bf16* g = gate + i / cols * gate_cols;
      for (int k = 0; k < kVec && col + k < gate_cols; ++k)
        acc[k] += __bfloat162float(g[col + k]);
    }
    uint4 raw;
    auto* b = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int k = 0; k < kVec / 2; ++k)
      b[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
    *reinterpret_cast<uint4*>(dx + i) = raw;
  }
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kThreads) nanochat_qkv_grad_sum(
      const __nv_bfloat16* dq, const __nv_bfloat16* dk, const __nv_bfloat16* dv, const __nv_bfloat16* gate,
      int gate_cols, __nv_bfloat16* dx, int64_t rows, int cols) {
  nanochat::qkv_grad_sum_body(dq, dk, dv, gate, gate_cols, dx, rows, cols);
}

namespace nanochat::kernels {

void qkv_grad_sum(
      const void* dq, const void* dk, const void* dv, const void* gate, int gate_cols, void* dx, int64_t rows, int cols,
      cudaStream_t stream) {
  static const int sms = [] {
    int device = 0, n = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    return n;
  }();
  const int64_t needed = (rows * cols / kVec + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sms)));
  nanochat_qkv_grad_sum<<<blocks, kThreads, 0, stream>>>(
        static_cast<const bf16*>(dq), static_cast<const bf16*>(dk), static_cast<const bf16*>(dv),
        static_cast<const bf16*>(gate), gate_cols, static_cast<bf16*>(dx), rows, cols);
}

} // namespace nanochat::kernels
