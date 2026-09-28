#include "nanochat/model/smear_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>

namespace nanochat {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32; // rows per block pass, one warp each
constexpr int kVec = 8;               // bf16 per 16-byte load
constexpr int kGateIters = kernels::kSmearMaxGateCols / 32;

using bf16 = __nv_bfloat16;

struct Vec8 {
  float v[kVec];
};

__device__ Vec8 load8(const bf16* p) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const auto* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
  Vec8 r;
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k) {
    const float2 f = __bfloat1622float2(b[k]);
    r.v[2 * k] = f.x, r.v[2 * k + 1] = f.y;
  }
  return r;
}

__device__ void store8(bf16* p, const Vec8& r) {
  uint4 raw;
  auto* b = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k)
    b[k] = __floats2bfloat162_rn(r.v[2 * k], r.v[2 * k + 1]);
  *reinterpret_cast<uint4*>(p) = raw;
}

__device__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// Lane l holds gate columns l, l + 32, ... (the same order in the gate logit, its input gradient and dw).
template <int kIters>
__device__ __forceinline__ void smear_fwd_body(
      const bf16* x, const float* w, const float* lambda, bf16* out, float* sig, int64_t rows, int64_t seq_len,
      int cols, int gate_cols) {
  const int lane = static_cast<int>(threadIdx.x % 32);
  const float lam = round_bf16(*lambda);
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32; row < rows;
       row += static_cast<int64_t>(gridDim.x) * kWarps) {
    const bf16* xr = x + row * cols;
    bf16* o = out + row * cols;
    if (row % seq_len == 0) { // the first token: unchanged
      if (lane == 0)
        sig[row] = 0.f;
#pragma unroll
      for (int it = 0; it < kIters; ++it) {
        const int col = (it * 32 + lane) * kVec;
        if (col < cols)
          *reinterpret_cast<uint4*>(o + col) = *reinterpret_cast<const uint4*>(xr + col);
      }
      continue;
    }
    // gate = bf16(lambda) * sigmoid(bf16(x[:gate_cols] . bf16(w))), rounded per op
    float z = 0.f;
    for (int c = lane; c < gate_cols; c += 32)
      z += __bfloat162float(xr[c]) * round_bf16(w[c]);
    z = round_bf16(warp_sum(z));
    const float s = round_bf16(1.f / (1.f + expf(-z)));
    const float gate = round_bf16(lam * s);
    if (lane == 0)
      sig[row] = s;
    const bf16* xp = xr - cols; // the previous token
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int col = (it * 32 + lane) * kVec;
      if (col >= cols)
        break;
      Vec8 v = load8(xr + col);
      const Vec8 p = load8(xp + col);
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        v.v[k] = round_bf16(v.v[k] + round_bf16(gate * p.v[k]));
      store8(o + col, v);
    }
  }
}

// dx[t] = g[t] + gate[t + 1] * g[t + 1] + dz[t] * w (gate columns), dz = dgate * lambda * s * (1 - s) with dgate =
// g[t] . x[t - 1]; dlambda += dgate * s, dw += dz * x[t, :gate_cols]. Block partials of dw and dlambda.
template <int kIters>
__device__ __forceinline__ void smear_bwd_body(
      const bf16* g, const bf16* x, const float* sig, const float* w, const float* lambda, bf16* dx, float* partials,
      int64_t rows, int64_t seq_len, int cols, int gate_cols) {
  __shared__ float smem[kWarps][kernels::kSmearMaxGateCols + 1];
  const int lane = static_cast<int>(threadIdx.x % 32), warp = static_cast<int>(threadIdx.x / 32);
  const float lam = round_bf16(*lambda);
  float dw[kGateIters] = {}, dl = 0.f;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp; row < rows;
       row += static_cast<int64_t>(gridDim.x) * kWarps) {
    const int64_t t = row % seq_len;
    const bf16* gr = g + row * cols;
    const bf16* xr = x + row * cols;
    const bool has_next = t + 1 < seq_len;
    const float gate_next = has_next ? round_bf16(lam * sig[row + 1]) : 0.f;
    float dz = 0.f;
    if (t > 0) {
      const bf16* xp = xr - cols;
      float dot = 0.f;
#pragma unroll
      for (int it = 0; it < kIters; ++it) {
        const int col = (it * 32 + lane) * kVec;
        if (col >= cols)
          break;
        const Vec8 gv = load8(gr + col), pv = load8(xp + col);
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          dot += gv.v[k] * pv.v[k];
      }
      const float dgate = warp_sum(dot), s = sig[row];
      dz = dgate * lam * s * (1.f - s);
      dl += dgate * s;
      for (int c = lane, i = 0; c < gate_cols; c += 32, ++i)
        dw[i] += dz * __bfloat162float(xr[c]);
    }
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int col = (it * 32 + lane) * kVec;
      if (col >= cols)
        break;
      Vec8 d = load8(gr + col);
      if (has_next) {
        const Vec8 n = load8(gr + cols + col);
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          d.v[k] += gate_next * n.v[k];
      }
      if (t > 0 && col < gate_cols) // whole 8-runs: gate_cols % 8 == 0
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          d.v[k] += dz * round_bf16(w[col + k]);
      store8(dx + row * cols + col, d);
    }
  }
  // warps' partials -> the block's, summed in a fixed order
  for (int c = lane, i = 0; c < gate_cols; c += 32, ++i)
    smem[warp][c] = dw[i];
  if (lane == 0)
    smem[warp][gate_cols] = dl;
  __syncthreads();
  for (int c = static_cast<int>(threadIdx.x); c <= gate_cols; c += kThreads) {
    float s = 0.f;
    for (int wp = 0; wp < kWarps; ++wp)
      s += smem[wp][c];
    partials[blockIdx.x * (gate_cols + 1) + c] = s;
  }
}

// One block: sums the partials over blocks in a fixed order, writes dw and dlambda.
__device__ __forceinline__ void smear_bwd_finalize_body(
      const float* partials, int blocks, int gate_cols, float* dw, float* dlambda) {
  for (int c = static_cast<int>(threadIdx.x); c <= gate_cols; c += kThreads) {
    float s = 0.f;
    for (int b = 0; b < blocks; ++b)
      s += partials[b * (gate_cols + 1) + c];
    if (c < gate_cols)
      dw[c] = s;
    else
      *dlambda = s;
  }
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu); the
// suffix is the largest cols each handles (row values stay in registers).
#define NANOCHAT_SMEAR(cols)                                                                                           \
  __global__ void __launch_bounds__(nanochat::kThreads) nanochat_smear_fwd_c##cols(                                    \
        const __nv_bfloat16* x, const float* w, const float* lambda, __nv_bfloat16* out, float* sig, int64_t rows,     \
        int64_t seq_len, int c, int gate_cols) {                                                                       \
    nanochat::smear_fwd_body<(cols) / 256>(x, w, lambda, out, sig, rows, seq_len, c, gate_cols);                       \
  }                                                                                                                    \
  __global__ void __launch_bounds__(nanochat::kThreads) nanochat_smear_bwd_c##cols(                                    \
        const __nv_bfloat16* g, const __nv_bfloat16* x, const float* sig, const float* w, const float* lambda,         \
        __nv_bfloat16* dx, float* partials, int64_t rows, int64_t seq_len, int c, int gate_cols) {                     \
    nanochat::smear_bwd_body<(cols) / 256>(g, x, sig, w, lambda, dx, partials, rows, seq_len, c, gate_cols);           \
  }

NANOCHAT_SMEAR(256)
NANOCHAT_SMEAR(512)
NANOCHAT_SMEAR(768)
NANOCHAT_SMEAR(1024)
NANOCHAT_SMEAR(1280)
NANOCHAT_SMEAR(1536)
NANOCHAT_SMEAR(1792)
NANOCHAT_SMEAR(2048)

__global__ void __launch_bounds__(nanochat::kThreads)
      nanochat_smear_bwd_finalize(const float* partials, int blocks, int gate_cols, float* dw, float* dlambda) {
  nanochat::smear_bwd_finalize_body(partials, blocks, gate_cols, dw, dlambda);
}

namespace nanochat::kernels {

namespace {

int blocks_for(int64_t rows) {
  static const int sms = [] {
    int device = 0, n = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    return n;
  }();
  const int64_t needed = (rows + kWarps - 1) / kWarps;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sms)));
}

using FwdKernel = void (*)(const bf16*, const float*, const float*, bf16*, float*, int64_t, int64_t, int, int);
using BwdKernel = void (*)(
      const bf16*, const bf16*, const float*, const float*, const float*, bf16*, float*, int64_t, int64_t, int, int);

template <typename K>
K select(int cols, const K (&kernels)[kSmearMaxCols / 256]) {
  return kernels[(cols + 255) / 256 - 1];
}

} // namespace

void smear_fwd(
      const void* x, const float* w, const float* lambda, void* out, float* sig, int64_t rows, int64_t seq_len,
      int cols, int gate_cols, cudaStream_t stream) {
  static constexpr FwdKernel kernels[] = {nanochat_smear_fwd_c256,  nanochat_smear_fwd_c512,  nanochat_smear_fwd_c768,
                                          nanochat_smear_fwd_c1024, nanochat_smear_fwd_c1280, nanochat_smear_fwd_c1536,
                                          nanochat_smear_fwd_c1792, nanochat_smear_fwd_c2048};
  select(cols, kernels)<<<blocks_for(rows), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x), w, lambda, static_cast<bf16*>(out), sig, rows, seq_len, cols, gate_cols);
}

int smear_bwd_blocks(int64_t rows) {
  return blocks_for(rows);
}

void smear_bwd(
      const void* g, const void* x, const float* sig, const float* w, const float* lambda, void* dx, float* partials,
      float* dw, float* dlambda, int64_t rows, int64_t seq_len, int cols, int gate_cols, cudaStream_t stream) {
  static constexpr BwdKernel kernels[] = {nanochat_smear_bwd_c256,  nanochat_smear_bwd_c512,  nanochat_smear_bwd_c768,
                                          nanochat_smear_bwd_c1024, nanochat_smear_bwd_c1280, nanochat_smear_bwd_c1536,
                                          nanochat_smear_bwd_c1792, nanochat_smear_bwd_c2048};
  const int blocks = blocks_for(rows);
  select(cols, kernels)<<<blocks, kThreads, 0, stream>>>(
        static_cast<const bf16*>(g), static_cast<const bf16*>(x), sig, w, lambda, static_cast<bf16*>(dx), partials,
        rows, seq_len, cols, gate_cols);
  nanochat_smear_bwd_finalize<<<1, kThreads, 0, stream>>>(partials, blocks, gate_cols, dw, dlambda);
}

} // namespace nanochat::kernels
