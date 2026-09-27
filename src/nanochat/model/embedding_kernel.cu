#include "nanochat/model/embedding_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>

namespace nanochat {

namespace {

constexpr int kVec = 8;    // bf16 per 16-byte load
constexpr int kUnroll = 4; // rows in flight per thread

using bf16 = __nv_bfloat16;

__device__ void add8(uint4 raw, float (&acc)[kVec]) {
  const auto* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k) {
    const float2 f = __bfloat1622float2(b[k]);
    acc[2 * k] += f.x, acc[2 * k + 1] += f.y;
  }
}

// Block (j, table): if position j starts a run of equal ids, sums the run's rows into grad[id]. Thread t owns columns
// [8t, 8t + 8), so the sum needs no cross-thread reduction.
__device__ __forceinline__ void embedding_grad_body(
      const int64_t* ids, const int64_t* order, int64_t n, const kernels::EmbeddingGrads& a) {
  const int64_t j = blockIdx.x;
  const int64_t id = ids[j];
  if (j > 0 && ids[j - 1] == id)
    return; // not a run start (uniform per block)
  // run end: the first k > j with another id (ids are sorted)
  __shared__ unsigned long long end;
  if (threadIdx.x == 0)
    end = static_cast<unsigned long long>(n);
  __syncthreads();
  for (int64_t base = j + 1; base < n; base += blockDim.x) {
    const int64_t k = base + threadIdx.x;
    if (k < n && ids[k] != id)
      atomicMin(&end, static_cast<unsigned long long>(k));
    __syncthreads();
    const bool done = end < static_cast<unsigned long long>(n);
    __syncthreads();
    if (done)
      break;
  }
  const int t = static_cast<int>(blockIdx.y), cols = a.cols[t], c = static_cast<int>(threadIdx.x) * kVec;
  if (c >= cols)
    return;
  const auto* g = static_cast<const bf16*>(a.g[t]);
  const auto stop = static_cast<int64_t>(end);
  float acc[kVec] = {};
  int64_t k = j;
  for (; k + kUnroll <= stop; k += kUnroll) {
    uint4 raw[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
      raw[u] = *reinterpret_cast<const uint4*>(g + order[k + u] * cols + c);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
      add8(raw[u], acc);
  }
  for (; k < stop; ++k)
    add8(*reinterpret_cast<const uint4*>(g + order[k] * cols + c), acc);
  auto* row = static_cast<bf16*>(a.grad[t]) + id * cols + c;
  add8(*reinterpret_cast<const uint4*>(row), acc);
  uint4 out;
  auto* b = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int k2 = 0; k2 < kVec / 2; ++k2)
    b[k2] = __floats2bfloat162_rn(acc[2 * k2], acc[2 * k2 + 1]);
  *reinterpret_cast<uint4*>(row) = out;
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kernels::EmbeddingGrads::kMaxCols / nanochat::kVec) nanochat_embedding_grad(
      const int64_t* ids, const int64_t* order, int64_t n, const nanochat::kernels::EmbeddingGrads a) {
  nanochat::embedding_grad_body(ids, order, n, a);
}

namespace nanochat::kernels {

void embedding_grad_accumulate(
      const int64_t* ids, const int64_t* order, int64_t n, const EmbeddingGrads& a, cudaStream_t stream) {
  if (n == 0 || a.tables == 0)
    return;
  const int max_cols = *std::max_element(a.cols, a.cols + a.tables);
  const int threads = (max_cols / kVec + 31) / 32 * 32;
  nanochat_embedding_grad<<<dim3(static_cast<unsigned>(n), static_cast<unsigned>(a.tables)), threads, 0, stream>>>(
        ids, order, n, a);
}

} // namespace nanochat::kernels
