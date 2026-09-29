#include "nanochat/model/nvfp4_sim_kernel.h"

#include <cuda_bf16.h>

#include "nanochat/model/nvfp4.cuh"

namespace nanochat {

namespace {

constexpr int kThreads = 256;
constexpr int kBlock = 16;

using bf16 = __nv_bfloat16;

template <class T>
__device__ void load16(const T* p, float* v);

template <>
__device__ void load16(const bf16* p, float* v) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint4 raw = q[h];
    const auto* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __bfloat1622float2(b[k]);
      v[h * 8 + 2 * k] = f.x, v[h * 8 + 2 * k + 1] = f.y;
    }
  }
}

template <>
__device__ void load16(const float* p, float* v) {
  const float4* q = reinterpret_cast<const float4*>(p);
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    const float4 f = q[h];
    v[4 * h] = f.x, v[4 * h + 1] = f.y, v[4 * h + 2] = f.z, v[4 * h + 3] = f.w;
  }
}

// one thread per 16-value block
template <class T>
__global__ void fake_quant_kernel(
      const T* x, int64_t blocks, const float* amax, const float* block_amax, bool stochastic, uint64_t seed,
      bf16* out) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (b >= blocks)
    return;
  float v[kBlock];
  load16(x + b * kBlock, v);
  const float tensor_amax = *amax;
  float bmax = 0.f;
  if (block_amax != nullptr)
    bmax = block_amax[b];
  else
#pragma unroll
    for (int i = 0; i < kBlock; ++i)
      bmax = fmaxf(bmax, fabsf(v[i]));
  const Nvfp4Scale sc = nvfp4_scale(bmax, nvfp4_encode(tensor_amax));
  const uint32_t codes[2] = {
        nvfp4_codes8(v, sc.to_q, stochastic, seed, b * kBlock),
        nvfp4_codes8(v + 8, sc.to_q, stochastic, seed, b * kBlock + 8)};
  uint4 packed[2];
  auto* o = reinterpret_cast<__nv_bfloat162*>(packed);
#pragma unroll
  for (int i = 0; i < kBlock; i += 2) {
    const uint32_t c = codes[i / 8] >> (4 * (i % 8));
    o[i / 2] = __floats2bfloat162_rn(nvfp4_value(c & 15, sc.from_q), nvfp4_value((c >> 4) & 15, sc.from_q));
  }
  uint4* dst = reinterpret_cast<uint4*>(out + b * kBlock);
  dst[0] = packed[0];
  dst[1] = packed[1];
}

} // namespace

namespace kernels {

void nvfp4_fake_quant(
      const void* x, bool is_bf16, int64_t rows, int64_t cols, const float* amax, const float* block_amax,
      bool stochastic, uint64_t seed, void* out, cudaStream_t stream) {
  const int64_t blocks = rows * cols / kBlock;
  if (blocks == 0)
    return;
  const auto grid = static_cast<unsigned>((blocks + kThreads - 1) / kThreads);
  auto* o = static_cast<bf16*>(out);
  if (is_bf16)
    fake_quant_kernel<<<grid, kThreads, 0, stream>>>(
          static_cast<const bf16*>(x), blocks, amax, block_amax, stochastic, seed, o);
  else
    fake_quant_kernel<<<grid, kThreads, 0, stream>>>(
          static_cast<const float*>(x), blocks, amax, block_amax, stochastic, seed, o);
}

} // namespace kernels

} // namespace nanochat
