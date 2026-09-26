#include "nanochat/model/fp8_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace nanochat::kernels {

namespace {

constexpr int kThreads = 256;

template <typename T>
struct Vec; // 16-byte loads

template <>
struct Vec<__nv_bfloat16> {
  static constexpr int kSize = 8;
};

template <>
struct Vec<float> {
  static constexpr int kSize = 4;
};

__device__ float to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ float to_float(float v) {
  return v;
}

// |x| >= 0, so float order is int order; NaN (0x7fc...) wins, as torch's max propagates it
__device__ void atomic_max_abs(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

template <typename T>
__global__ void __launch_bounds__(kThreads) amax_kernel(const T* x, int64_t n, float* amax) {
  constexpr int V = Vec<T>::kSize;
  float m = 0.f;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  const int64_t n_vec = n / V;
  for (int64_t i = blockIdx.x * kThreads + threadIdx.x; i < n_vec; i += stride) {
    const uint4 raw = reinterpret_cast<const uint4*>(x)[i];
    const auto* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int k = 0; k < V; ++k)
      m = fmaxf(m, fabsf(to_float(v[k])));
  }
  for (int64_t i = n_vec * V + blockIdx.x * kThreads + threadIdx.x; i < n; i += stride)
    m = fmaxf(m, fabsf(to_float(x[i])));
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  __shared__ float smem[kThreads / 32];
  if (threadIdx.x % 32 == 0)
    smem[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < kThreads / 32; ++w)
      m = fmaxf(m, smem[w]);
    atomic_max_abs(amax, m);
  }
}

__device__ float fp8_scale(float amax, float fp8_max) {
  return static_cast<float>(fp8_max * (1.0 / fmax(static_cast<double>(amax), 1e-12)));
}

// 64x64 tiles, 256 threads. Each thread casts two runs of 8 values of a row (16-byte loads, 8-byte stores),
// then writes 16 bytes of one transposed row. Needs cols % 8 == 0.
constexpr int kTile = 64, kTileThreads = 256, kPad = 16;

template <typename T>
__device__ void load8(const T* p, float (&v)[8]) {
  if constexpr (sizeof(T) == 2) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const auto* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      v[k] = __bfloat162float(b[k]);
  }
  else {
    const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
  }
}

template <typename T, __nv_fp8_interpretation_t kFormat>
__global__ void __launch_bounds__(kTileThreads) cast_kernel(
      const T* x, int64_t rows, int64_t cols, float fp8_max, const float* amax, __nv_fp8_storage_t* out,
      __nv_fp8_storage_t* out_t, float* inv_scale) {
  __shared__ __align__(16) __nv_fp8_storage_t tile[kTile][kTile + kPad];
  const float scale = fp8_scale(*amax, fp8_max);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    *inv_scale = 1.f / scale;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * kTile;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kTile;
#pragma unroll
  for (int pass = 0; pass < kTile * kTile / 8 / kTileThreads; ++pass) {
    const int idx = pass * kTileThreads + static_cast<int>(threadIdx.x);
    const int r = idx / (kTile / 8), c = idx % (kTile / 8) * 8;
    const int64_t gr = row0 + r, gc = col0 + c;
    if (gr >= rows || gc >= cols)
      continue;
    float v[8];
    load8(x + gr * cols + gc, v);
    __align__(8) __nv_fp8_storage_t q[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
      q[k] = __nv_cvt_float_to_fp8(fminf(fmaxf(v[k] * scale, -fp8_max), fp8_max), __NV_SATFINITE, kFormat);
    if (out != nullptr)
      *reinterpret_cast<uint2*>(out + gr * cols + gc) = *reinterpret_cast<const uint2*>(q);
    *reinterpret_cast<uint2*>(&tile[r][c]) = *reinterpret_cast<const uint2*>(q);
  }
  if (out_t == nullptr)
    return;
  __syncthreads();
  // out_t row = tile column tc, bytes for tile rows [r, r + 16)
  const int tc = static_cast<int>(threadIdx.x) / (kTile / 16), r = static_cast<int>(threadIdx.x) % (kTile / 16) * 16;
  const int64_t gtc = col0 + tc, gr = row0 + r;
  if (gtc >= cols || gr >= rows)
    return;
  __align__(16) __nv_fp8_storage_t q[16];
#pragma unroll
  for (int k = 0; k < 16; ++k)
    q[k] = tile[r + k][tc];
  auto* dst = out_t + gtc * rows + gr;
  if (gr + 16 <= rows && reinterpret_cast<uintptr_t>(dst) % 16 == 0)
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(q);
  else
    for (int k = 0; k < 16 && gr + k < rows; ++k)
      dst[k] = q[k];
}

template <typename T>
void launch(
      const T* x, int64_t rows, int64_t cols, Fp8Format format, void* out, void* out_t, float* amax, float* inv_scale,
      cudaStream_t stream) {
  int sms = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  const int64_t n = rows * cols;
  const int64_t needed = (n / Vec<T>::kSize + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sms)));
  cudaMemsetAsync(amax, 0, sizeof(float), stream);
  amax_kernel<T><<<blocks, kThreads, 0, stream>>>(x, n, amax);
  const dim3 grid(static_cast<unsigned>((cols + kTile - 1) / kTile), static_cast<unsigned>((rows + kTile - 1) / kTile));
  const dim3 block(kTileThreads);
  auto* q = static_cast<__nv_fp8_storage_t*>(out);
  auto* q_t = static_cast<__nv_fp8_storage_t*>(out_t);
  if (format == Fp8Format::E4M3)
    cast_kernel<T, __NV_E4M3><<<grid, block, 0, stream>>>(x, rows, cols, 448.f, amax, q, q_t, inv_scale);
  else
    cast_kernel<T, __NV_E5M2><<<grid, block, 0, stream>>>(x, rows, cols, 57344.f, amax, q, q_t, inv_scale);
}

} // namespace

void quantize_fp8(
      const void* x, bool x_bf16, int64_t rows, int64_t cols, Fp8Format format, void* out, void* out_t, float* amax,
      float* inv_scale, cudaStream_t stream) {
  if (x_bf16)
    launch(static_cast<const __nv_bfloat16*>(x), rows, cols, format, out, out_t, amax, inv_scale, stream);
  else
    launch(static_cast<const float*>(x), rows, cols, format, out, out_t, amax, inv_scale, stream);
}

} // namespace nanochat::kernels
