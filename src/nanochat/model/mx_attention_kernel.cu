#include "nanochat/model/mx_attention_kernel.h"

#include <cuda_bf16.h>

#include "nanochat/model/flash_mx.cuh"
#include "nanochat/model/mx_kernel.cuh"
#include "nanochat/model/rotary_norm.cuh"

namespace nanochat {

namespace {

using bf16 = __nv_bfloat16;
using kernels::kMxHeadDim;

__device__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

// gpt.py's gate: bf16(3 * bf16(sigmoid(z))), sigmoid as torch's (1 / (1 + exp(-z)) in float)
__device__ float sigmoid_bf16(float z) {
  return round_bf16(1.f / (1.f + expf(-z)));
}

// One block per 32 tokens of one head (a kMxRows x kMxHeadDim tile), a warp per row, 4 rows each.
constexpr int kRowsPerWarp = kMxRows / (kMxThreads / 32);

// grid: (heads, tokens / 32)
__device__ __forceinline__ void rotary_norm_bwd_mx_body(
      const bf16* dout, const bf16* x, const bf16* cos, const bf16* sin, const float* rstd, int heads, int64_t seq_len,
      int64_t x_stride, float scale, MxOutDev out, MxOutDev out_t) {
  __shared__ float tile[kMxRows][kMxHeadDim + 1];
  constexpr int half = kMxHeadDim / 2;
  const int h = static_cast<int>(blockIdx.x), warp = static_cast<int>(threadIdx.x / 32);
  const int64_t token0 = static_cast<int64_t>(blockIdx.y) * kMxRows;
  for (int i = 0; i < kRowsPerWarp; ++i) {
    const int r = warp + i * (kMxThreads / 32);
    const int64_t row = (token0 + r) * heads + h, t = (token0 + r) % seq_len;
    rotary::bwd_row(
          dout + row * kMxHeadDim, rotary::x_row(x, row, heads, kMxHeadDim, x_stride), cos + t * half, sin + t * half,
          rstd[row], kMxHeadDim, scale, [&](int j, float a, float b) {
            tile[r][j] = round_bf16(a);
            tile[r][j + 1] = round_bf16(b);
          });
  }
  __syncthreads();
  mx_store_tile<kMxHeadDim>(tile, token0, static_cast<int64_t>(h) * kMxHeadDim, out, out_t);
}

// 8 values per thread
__device__ __forceinline__ void value_mix_fwd_body(
      const bf16* v, int64_t v_stride, const bf16* z, const bf16* ve, bf16* out, int64_t tokens, int heads) {
  const int64_t width = static_cast<int64_t>(heads) * kMxHeadDim;
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (i >= tokens * width)
    return;
  const int64_t token = i / width, col = i % width;
  const float gate = round_bf16(3.f * sigmoid_bf16(__bfloat162float(z[token * heads + col / kMxHeadDim])));
  float a[8], e[8];
  load8(v + token * v_stride + col, a);
  load8(ve + i, e);
  __align__(16) bf16 o[8];
#pragma unroll
  for (int k = 0; k < 8; ++k)
    o[k] = __float2bfloat16(a[k] + round_bf16(gate * e[k]));
  *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
}

// value_mix_fwd_body's values on a 64-token tile, then MX-quantized both ways. grid: (seq_len / 64, heads, B)
__device__ __forceinline__ void value_mix_mx_body(
      const bf16* v, int64_t v_stride, const bf16* z, const bf16* ve, bf16* out, uint8_t* v8, uint32_t* v8_scale,
      uint8_t* vt, uint8_t* vt_scale, int64_t seq_len, int heads) {
  __shared__ __align__(16) flash::TTile tile;
  const int n = static_cast<int>(blockIdx.x), h = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t width = static_cast<int64_t>(heads) * kMxHeadDim;
#pragma unroll
  for (int i = 0; i < flash::kTTokens * kMxHeadDim / 8 / flash::kTThreads; ++i) {
    const int idx = tid + i * flash::kTThreads, t = idx / (kMxHeadDim / 8), c = idx % (kMxHeadDim / 8);
    const int64_t token = static_cast<int64_t>(b) * seq_len + static_cast<int64_t>(n) * flash::kTTokens + t;
    const int64_t col = static_cast<int64_t>(h) * kMxHeadDim + c * 8;
    float a[8];
    load8(v + token * v_stride + col, a);
    __align__(16) bf16 o[8];
    if (ve != nullptr) {
      const float gate = round_bf16(3.f * sigmoid_bf16(__bfloat162float(z[token * heads + h])));
      float e[8];
      load8(ve + token * width + col, e);
#pragma unroll
      for (int k = 0; k < 8; ++k)
        o[k] = __float2bfloat16(a[k] + round_bf16(gate * e[k]));
      if (out != nullptr)
        *reinterpret_cast<uint4*>(out + token * width + col) = *reinterpret_cast<const uint4*>(o);
    }
    else {
#pragma unroll
      for (int k = 0; k < 8; ++k)
        o[k] = __float2bfloat16(a[k]);
    }
    *reinterpret_cast<uint4*>(&tile[t][c * 8]) = *reinterpret_cast<const uint4*>(o);
  }
  __syncthreads();
  flash::store_t_tile(tile, vt, vt_scale, static_cast<int64_t>(b) * heads + h, n, seq_len);
  flash::store_rows_tile(tile, v8, v8_scale, b, static_cast<int64_t>(n) * flash::kTTokens, h, heads, seq_len);
}

// The ops' backward: dve = bf16(dout * gate); d gate = bf16(sum(bf16(dout * ve))), times 3, then sigmoid's
// (d * (1 - s)) * s. Lane: 4 values of a row. grid: (heads, tokens / 32)
__device__ __forceinline__ void value_mix_bwd_mx_body(
      const bf16* dout, const bf16* z, const bf16* ve, bf16* dve, bf16* dz, int heads, MxOutDev out, MxOutDev out_t) {
  __shared__ float tile[kMxRows][kMxHeadDim + 1];
  const int h = static_cast<int>(blockIdx.x), warp = static_cast<int>(threadIdx.x / 32);
  const int lane = static_cast<int>(threadIdx.x % 32), c = lane * 4;
  const int64_t token0 = static_cast<int64_t>(blockIdx.y) * kMxRows;
  for (int i = 0; i < kRowsPerWarp; ++i) {
    const int r = warp + i * (kMxThreads / 32);
    const int64_t row = (token0 + r) * heads + h, at = row * kMxHeadDim + c;
    const float s = sigmoid_bf16(__bfloat162float(z[row])), gate = round_bf16(3.f * s);
    const uint2 graw = *reinterpret_cast<const uint2*>(dout + at), eraw = *reinterpret_cast<const uint2*>(ve + at);
    const auto *g = reinterpret_cast<const bf16*>(&graw), *e = reinterpret_cast<const bf16*>(&eraw);
    __align__(8) bf16 d[4];
    float dot = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float gk = __bfloat162float(g[k]);
      dot += round_bf16(gk * __bfloat162float(e[k]));
      d[k] = __float2bfloat16(gk * gate);
      tile[r][c + k] = gk;
    }
    *reinterpret_cast<uint2*>(dve + at) = *reinterpret_cast<const uint2*>(d);
    const float dg = round_bf16(3.f * round_bf16(rotary::warp_sum(dot)));
    if (lane == 0)
      dz[row] = __float2bfloat16(dg * (1.f - s) * s);
  }
  __syncthreads();
  mx_store_tile<kMxHeadDim>(tile, token0, static_cast<int64_t>(h) * kMxHeadDim, out, out_t);
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kMxThreads) nanochat_rotary_norm_bwd_mx(
      const __nv_bfloat16* dout, const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
      const float* rstd, int heads, int64_t seq_len, int64_t x_stride, float scale, nanochat::MxOutDev out,
      nanochat::MxOutDev out_t) {
  nanochat::rotary_norm_bwd_mx_body(dout, x, cos, sin, rstd, heads, seq_len, x_stride, scale, out, out_t);
}

__global__ void __launch_bounds__(256) nanochat_value_mix_fwd(
      const __nv_bfloat16* v, int64_t v_stride, const __nv_bfloat16* z, const __nv_bfloat16* ve, __nv_bfloat16* out,
      int64_t tokens, int heads) {
  nanochat::value_mix_fwd_body(v, v_stride, z, ve, out, tokens, heads);
}

__global__ void __launch_bounds__(nanochat::flash::kTThreads) nanochat_value_mix_mx(
      const __nv_bfloat16* v, int64_t v_stride, const __nv_bfloat16* z, const __nv_bfloat16* ve, __nv_bfloat16* out,
      uint8_t* v8, uint32_t* v8_scale, uint8_t* vt, uint8_t* vt_scale, int64_t seq_len, int heads) {
  nanochat::value_mix_mx_body(v, v_stride, z, ve, out, v8, v8_scale, vt, vt_scale, seq_len, heads);
}

__global__ void __launch_bounds__(nanochat::kMxThreads) nanochat_value_mix_bwd_mx(
      const __nv_bfloat16* dout, const __nv_bfloat16* z, const __nv_bfloat16* ve, __nv_bfloat16* dve, __nv_bfloat16* dz,
      int heads, nanochat::MxOutDev out, nanochat::MxOutDev out_t) {
  nanochat::value_mix_bwd_mx_body(dout, z, ve, dve, dz, heads, out, out_t);
}

namespace nanochat::kernels {

namespace {

using bf16 = __nv_bfloat16;

dim3 head_tiles(int64_t tokens, int heads) {
  return {static_cast<unsigned>(heads), static_cast<unsigned>(tokens / kMxRows)};
}

} // namespace

void rotary_norm_bwd_mx(
      const void* dout, const void* x, const void* cos, const void* sin, const float* rstd, int64_t tokens, int heads,
      int64_t seq_len, int64_t x_stride, float scale, MxOut out, MxOut out_t, cudaStream_t stream) {
  nanochat_rotary_norm_bwd_mx<<<head_tiles(tokens, heads), kMxThreads, 0, stream>>>(
        static_cast<const bf16*>(dout), static_cast<const bf16*>(x), static_cast<const bf16*>(cos),
        static_cast<const bf16*>(sin), rstd, heads, seq_len, x_stride, scale, mx_dev(out), mx_dev(out_t));
}

void value_mix_fwd(
      const void* v, int64_t v_stride, const void* z, const void* ve, void* out, int64_t tokens, int heads,
      cudaStream_t stream) {
  const int64_t n = tokens * heads * kMxHeadDim / 8;
  nanochat_value_mix_fwd<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<const bf16*>(v), v_stride, static_cast<const bf16*>(z), static_cast<const bf16*>(ve),
        static_cast<bf16*>(out), tokens, heads);
}

void value_mix_mx(
      const void* v, int64_t v_stride, const void* z, const void* ve, void* out, void* v8, uint32_t* v8_scale, void* vt,
      uint8_t* vt_scale, int B, int64_t seq_len, int heads, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(seq_len / flash::kTTokens), heads, B);
  nanochat_value_mix_mx<<<grid, flash::kTThreads, 0, stream>>>(
        static_cast<const bf16*>(v), v_stride, static_cast<const bf16*>(z), static_cast<const bf16*>(ve),
        static_cast<bf16*>(out), static_cast<uint8_t*>(v8), v8_scale, static_cast<uint8_t*>(vt), vt_scale, seq_len,
        heads);
}

void value_mix_bwd_mx(
      const void* dout, const void* z, const void* ve, void* dve, void* dz, int64_t tokens, int heads, MxOut out,
      MxOut out_t, cudaStream_t stream) {
  nanochat_value_mix_bwd_mx<<<head_tiles(tokens, heads), kMxThreads, 0, stream>>>(
        static_cast<const bf16*>(dout), static_cast<const bf16*>(z), static_cast<const bf16*>(ve),
        static_cast<bf16*>(dve), static_cast<bf16*>(dz), heads, mx_dev(out), mx_dev(out_t));
}

} // namespace nanochat::kernels
