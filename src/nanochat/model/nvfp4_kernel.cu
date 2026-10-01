#include "nanochat/model/nvfp4_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>

#include "nanochat/model/mx_kernel.cuh"
#include "nanochat/model/nvfp4.cuh"

namespace nanochat {

namespace {

constexpr int kThreads = 256;
constexpr int kGroup = 16;

using bf16 = __nv_bfloat16;

// Each thread: one 16-value group, g = row * (cols / 16) + col / 16, so a warp reads 512 contiguous bytes.
struct Group {
  int64_t row, col, g;
};

__device__ __forceinline__ bool group_of(int64_t rows, int64_t cols, Group& grp) {
  grp.g = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (grp.g >= rows * cols / kGroup)
    return false;
  grp.row = grp.g / (cols / kGroup);
  grp.col = grp.g % (cols / kGroup) * kGroup;
  return true;
}

// 16 MX values, dequantized (exact in fp32), then times H if given
__device__ __forceinline__ void load_mx16(
      const uint8_t* data, const uint8_t* scale, int64_t cols, const Group& grp, const float* hadamard, float* v) {
  const uint4 raw = *reinterpret_cast<const uint4*>(data + grp.row * cols + grp.col);
  const auto* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
  const float mul = ldexpf(1.f, static_cast<int>(scale[mx_scale_index(grp.row, grp.col / 32, cols / 128)]) - 127);
#pragma unroll
  for (int k = 0; k < kGroup / 2; ++k) {
    const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(p[k], __NV_E4M3)));
    v[2 * k] = f.x * mul, v[2 * k + 1] = f.y * mul;
  }
  if (hadamard == nullptr)
    return;
  // x . H with H = diag(signs) . Walsh-Hadamard / 4: H's first column holds signs / 4 (exact scaling), then the fast
  // transform (Sylvester order, symmetric)
#pragma unroll
  for (int i = 0; i < kGroup; ++i)
    v[i] *= hadamard[i * kGroup];
#pragma unroll
  for (int h = 1; h < kGroup; h *= 2)
#pragma unroll
    for (int i = 0; i < kGroup; i += 2 * h)
#pragma unroll
      for (int j = i; j < i + h; ++j) {
        const float a = v[j], b = v[j + h];
        v[j] = a + b, v[j + h] = a - b;
      }
}

// H in shared memory
__device__ __forceinline__ const float* stage_hadamard(const float* hadamard, float* smem) {
  if (hadamard == nullptr)
    return nullptr;
  smem[threadIdx.x] = hadamard[threadIdx.x]; // kThreads == 256 == 16 * 16
  __syncthreads();
  return smem;
}

__global__ void amax_kernel(
      const uint8_t* data, const uint8_t* scale, int64_t rows, int64_t cols, const float* hadamard, float* amax) {
  __shared__ float h[kGroup * kGroup];
  const float* H = stage_hadamard(hadamard, h);
  Group grp;
  float m = 0.f;
  if (group_of(rows, cols, grp)) {
    float v[kGroup];
    load_mx16(data, scale, cols, grp, H, v);
#pragma unroll
    for (int i = 0; i < kGroup; ++i)
      m = fmaxf(m, fabsf(v[i]));
  }
#pragma unroll
  for (int o = 16; o > 0; o /= 2)
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  if (threadIdx.x % 32 == 0 && m > 0.f)
    atomicMax(reinterpret_cast<int*>(amax), __float_as_int(m)); // non-negative floats order as ints
}

__global__ void quantize_kernel(
      const uint8_t* data, const uint8_t* scale, int64_t rows, int64_t cols, const float* hadamard, bool stochastic,
      uint64_t seed, uint8_t* out, uint8_t* out_scale, const float* amax) {
  __shared__ float h[kGroup * kGroup];
  const float* H = stage_hadamard(hadamard, h);
  Group grp;
  if (!group_of(rows, cols, grp))
    return;
  float v[kGroup];
  load_mx16(data, scale, cols, grp, H, v);
  float bmax = 0.f;
#pragma unroll
  for (int i = 0; i < kGroup; ++i)
    bmax = fmaxf(bmax, fabsf(v[i]));
  const Nvfp4Scale sc = nvfp4_scale(bmax, nvfp4_encode(*amax));
  const int64_t i = grp.g * kGroup;
  *reinterpret_cast<uint2*>(out + grp.row * (cols / 2) + grp.col / 2) = make_uint2(
        nvfp4_codes8(v, sc.to_q, stochastic, seed, i), nvfp4_codes8(v + 8, sc.to_q, stochastic, seed, i + 8));
  out_scale[mx_scale_index(grp.row, grp.col / kGroup, cols / 64)] = sc.s8;
}

__global__ void nvfp4_to_bf16_kernel(
      const uint8_t* data, const uint8_t* scale, const float* amax, int64_t rows, int64_t cols, bf16* out) {
  Group grp;
  if (!group_of(rows, cols, grp))
    return;
  const uint2 raw = *reinterpret_cast<const uint2*>(data + grp.row * (cols / 2) + grp.col / 2);
  const uint8_t s8 = scale[mx_scale_index(grp.row, grp.col / kGroup, cols / 64)];
  const float s = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3)));
  const float enc = nvfp4_encode(*amax), from_q = s > 0.f ? s / enc : 0.f;
  bf16* o = out + grp.row * cols + grp.col;
#pragma unroll
  for (int i = 0; i < kGroup; ++i)
    o[i] = __float2bfloat16_rn(nvfp4_value(((i < 8 ? raw.x : raw.y) >> (4 * (i % 8))) & 15, from_q));
}

__global__ void mx_to_bf16_kernel(const uint8_t* data, const uint8_t* scale, int64_t rows, int64_t cols, bf16* out) {
  Group grp;
  if (!group_of(rows, cols, grp))
    return;
  float v[kGroup];
  load_mx16(data, scale, cols, grp, nullptr, v);
  bf16* o = out + grp.row * cols + grp.col;
#pragma unroll
  for (int i = 0; i < kGroup; ++i)
    o[i] = __float2bfloat16_rn(v[i]);
}

// 16 threads per 16x16 block: thread i reads the block's row i and column i, and writes them to out and out_t
template <class T>
__global__ void quantize_2d_kernel(
      const T* x, int64_t rows, int64_t cols, const kernels::Nvfp4Out out, const kernels::Nvfp4Out out_t) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x, b = t / kGroup;
  const bool active = b < rows * cols / (kGroup * kGroup);
  const int i = static_cast<int>(threadIdx.x % kGroup);
  const int64_t r0 = active ? b / (cols / kGroup) * kGroup : 0, c0 = active ? b % (cols / kGroup) * kGroup : 0;
  float row[kGroup], col[kGroup], m = 0.f;
#pragma unroll
  for (int k = 0; k < kGroup; ++k) {
    row[k] = active ? static_cast<float>(x[(r0 + i) * cols + c0 + k]) : 0.f;
    col[k] = active ? static_cast<float>(x[(r0 + k) * cols + c0 + i]) : 0.f;
    m = fmaxf(m, fabsf(row[k]));
  }
#pragma unroll
  for (int o = 1; o < kGroup; o *= 2)
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  float s = nvfp4_round_scale(nvfp4_div6(m)), to_q = s > 0.f ? nvfp4_rcp(s) : 0.f;
  if (out.four_six) { // the block's errors over its 16 rows
    float cs[2], ct[2], e[2];
    nvfp4_four_six_scales(m, cs, ct);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      e[k] = nvfp4_err8(row, ct[k]) + nvfp4_err8(row + 8, ct[k]);
#pragma unroll
      for (int o = 1; o < kGroup; o *= 2)
        e[k] += __shfl_xor_sync(0xffffffff, e[k], o);
    }
    const bool four = nvfp4_pick_four(cs, e);
    s = four ? cs[1] : cs[0], to_q = four ? ct[1] : ct[0];
  }
  if (active) {
    const auto store = [&](const kernels::Nvfp4Out& o, const float* v, int64_t r, int64_t c) {
      const int64_t j = r * o.ld + c;
      *reinterpret_cast<uint2*>(static_cast<uint8_t*>(o.data) + j / 2) = make_uint2(
            nvfp4_pack8(v, to_q, o, o.index0 + j), nvfp4_pack8(v + 8, to_q, o, o.index0 + j + 8));
      nvfp4_store_scale(o, r, c, s);
    };
    store(out, row, r0 + i, c0);
    if (out_t.data != nullptr)
      store(out_t, col, c0 + i, r0);
  }
  nvfp4_smax_block(out, s);
  if (out_t.data != nullptr)
    nvfp4_smax_block(out_t, s);
}

// The tensor scale's exponent: the largest k with smax * 2^k <= 448 (smax has 3 mantissa bits; 448 = 1.75 * 2^8)
__device__ __forceinline__ int finish_shift(unsigned smax) {
  if (smax == 0)
    return 0;
  const int e = static_cast<int>(smax >> 23) - 127, mantissa = static_cast<int>(smax >> 20) & 7;
  return 8 - e - (mantissa == 7 ? 1 : 0);
}

int64_t finish_blocks(int64_t rows, int64_t cols) {
  return std::max<int64_t>((rows * cols / kGroup / 8 + kThreads - 1) / kThreads, 1);
}

// up to two tensors, blocks [0, blocks0) the first's
struct FinishJobs {
  kernels::Nvfp4Finish job[2];
  unsigned blocks0;
};

// elementwise over the swizzled scales, 8 per thread
__global__ void finish_kernel(const FinishJobs jobs) {
  const bool second = blockIdx.x >= jobs.blocks0;
  const kernels::Nvfp4Finish j = second ? jobs.job[1] : jobs.job[0];
  const auto* smax = j.smax;
  __shared__ int shift;
  if (threadIdx.x < 32) {
    const unsigned long long v = threadIdx.x < kernels::kNvfp4Slots ? smax[threadIdx.x] : 0ull;
    const unsigned m = __reduce_max_sync(0xffffffff, v >> 32 == j.epoch ? static_cast<unsigned>(v) : 0u);
    if (threadIdx.x == 0)
      shift = finish_shift(m);
  }
  __syncthreads();
  const int k = shift;
  const int64_t i = static_cast<int64_t>(second ? blockIdx.x - jobs.blocks0 : blockIdx.x) * kThreads + threadIdx.x;
  if (i == 0)
    *j.amax = ldexpf(kE2m1Max * kE4m3Max, -k);
  if (i >= j.rows * j.cols / kGroup / 8)
    return;
  const uint4 raw = static_cast<const uint4*>(j.scale16)[i];
  const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
  uint32_t out[2] = {0, 0};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float s = ldexpf(__uint_as_float(((w[j / 2] >> (16 * (j % 2))) & 0xffffu) << 16), k);
    out[j / 4] |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3)) << (8 * (j % 4));
  }
  static_cast<uint2*>(j.out_scale)[i] = make_uint2(out[0], out[1]);
}

unsigned grid_for(int64_t rows, int64_t cols) {
  return static_cast<unsigned>((rows * cols / kGroup + kThreads - 1) / kThreads);
}

} // namespace

namespace kernels {

void mx_to_nvfp4(
      const void* data, const void* scale, int64_t rows, int64_t cols, const float* hadamard, bool stochastic,
      uint64_t seed, void* out, void* out_scale, float* amax, cudaStream_t stream) {
  if (rows * cols == 0)
    return;
  const auto* d = static_cast<const uint8_t*>(data);
  const auto* s = static_cast<const uint8_t*>(scale);
  amax_kernel<<<grid_for(rows, cols), kThreads, 0, stream>>>(d, s, rows, cols, hadamard, amax);
  quantize_kernel<<<grid_for(rows, cols), kThreads, 0, stream>>>(
        d, s, rows, cols, hadamard, stochastic, seed, static_cast<uint8_t*>(out), static_cast<uint8_t*>(out_scale),
        amax);
}

void quantize_nvfp4_2d(
      const void* x, bool x_f32, int64_t rows, int64_t cols, const Nvfp4Out& out, const Nvfp4Out& out_t,
      cudaStream_t stream) {
  if (rows * cols == 0)
    return;
  if (x_f32)
    quantize_2d_kernel<<<grid_for(rows, cols), kThreads, 0, stream>>>(
          static_cast<const float*>(x), rows, cols, out, out_t);
  else
    quantize_2d_kernel<<<grid_for(rows, cols), kThreads, 0, stream>>>(
          static_cast<const bf16*>(x), rows, cols, out, out_t);
}

void nvfp4_finish(const Nvfp4Finish& a, const Nvfp4Finish* b, cudaStream_t stream) {
  const auto blocks0 = static_cast<unsigned>(finish_blocks(a.rows, a.cols));
  const auto blocks = blocks0 + (b != nullptr ? static_cast<unsigned>(finish_blocks(b->rows, b->cols)) : 0u);
  finish_kernel<<<blocks, kThreads, 0, stream>>>({{a, b != nullptr ? *b : a}, blocks0});
}

void nvfp4_to_bf16(
      const void* data, const void* scale, const float* amax, int64_t rows, int64_t cols, void* out,
      cudaStream_t stream) {
  if (rows * cols == 0)
    return;
  nvfp4_to_bf16_kernel<<<grid_for(rows, cols), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(scale), amax, rows, cols,
        static_cast<bf16*>(out));
}

void mx_to_bf16(const void* data, const void* scale, int64_t rows, int64_t cols, void* out, cudaStream_t stream) {
  if (rows * cols == 0)
    return;
  mx_to_bf16_kernel<<<grid_for(rows, cols), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(scale), rows, cols, static_cast<bf16*>(out));
}

} // namespace kernels

} // namespace nanochat

namespace nanochat::kernels {

namespace {

__global__ void alpha_kernel(const float* a_amax, const float* b_amax, const float* alpha, float* out) {
  constexpr float kNorm = kE2m1Max * kE4m3Max;
  float s = a_amax != nullptr ? *a_amax / kNorm * (*b_amax / kNorm) : 1.f;
  if (alpha != nullptr)
    s = s * *alpha;
  *out = s;
}

} // namespace

void nvfp4_alpha(const float* a_amax, const float* b_amax, const float* alpha, float* out, cudaStream_t stream) {
  alpha_kernel<<<1, 1, 0, stream>>>(a_amax, b_amax, alpha, out);
}

} // namespace nanochat::kernels
