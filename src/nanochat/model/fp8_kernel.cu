#include "nanochat/model/fp8_kernel.h"

#include <algorithm>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "nanochat/model/mx_kernel.cuh"

namespace nanochat {

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

// Applied to each value before quantizing.
struct Identity {
  __device__ float operator()(float v) const {
    return v;
  }
};

// bf16(relu(v)^2), as relu(h).square() (relu_square_kernel.cu): quantizes the MLP activation without writing it
struct ReluSquare {
  __device__ float operator()(float v) const {
    const float r = v <= 0.f ? 0.f : v;
    return __bfloat162float(__float2bfloat16(r * r));
  }
};

// |x| >= 0, so float order is int order; NaN (0x7fc...) wins, as torch's max propagates it
__device__ void atomic_max_abs(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

template <typename T, typename P>
__device__ __forceinline__ void amax_body(const T* x, int64_t n, float* amax) {
  constexpr int V = Vec<T>::kSize;
  float m = 0.f;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  const int64_t n_vec = n / V;
  for (int64_t i = blockIdx.x * kThreads + threadIdx.x; i < n_vec; i += stride) {
    const uint4 raw = reinterpret_cast<const uint4*>(x)[i];
    const auto* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int k = 0; k < V; ++k)
      m = fmaxf(m, fabsf(P{}(to_float(v[k]))));
  }
  for (int64_t i = n_vec * V + blockIdx.x * kThreads + threadIdx.x; i < n; i += stride)
    m = fmaxf(m, fabsf(P{}(to_float(x[i]))));
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

template <typename T, __nv_fp8_interpretation_t kFormat, typename P>
__device__ __forceinline__ void cast_body(
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
      q[k] = __nv_cvt_float_to_fp8(fminf(fmaxf(P{}(v[k]) * scale, -fp8_max), fp8_max), __NV_SATFINITE, kFormat);
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

// mx_body's input: P of x (rows, cols)
template <typename T, typename P>
struct MxLoad {
  const T* x;
  int64_t cols;

  __device__ void operator()(int64_t row, int64_t col, float (&v)[8]) const {
    load8(x + row * cols + col, v);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      v[k] = P{}(v[k]);
  }
};

// dh = bf16(h > 0 ? g * 2h : 0), as relu_square_kernel.cu's backward
struct MxLoadReluSquareGrad {
  const __nv_bfloat16 *g, *h;
  int64_t cols;

  __device__ void operator()(int64_t row, int64_t col, float (&v)[8]) const {
    const int64_t i = row * cols + col;
    float hv[8];
    load8(g + i, v);
    load8(h + i, hv);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      v[k] = __bfloat162float(__float2bfloat16(hv[k] <= 0.f ? 0.f : v[k] * (2.f * hv[k])));
  }
};

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (ncu drops the innermost namespace).
// Not extern "C": ncu's --filter-mode per-launch-config then confuses kernels of equal launch shape.
#define NANOCHAT_FP8_AMAX(name, T, P)                                                                                  \
  __global__ void __launch_bounds__(nanochat::kThreads) name(const T* x, int64_t n, float* amax) {                     \
    nanochat::amax_body<T, P>(x, n, amax);                                                                             \
  }

#define NANOCHAT_FP8_CAST(name, T, format, P)                                                                          \
  __global__ void __launch_bounds__(nanochat::kTileThreads)                                                            \
        name(const T* x, int64_t rows, int64_t cols, float fp8_max, const float* amax, __nv_fp8_storage_t* out,        \
             __nv_fp8_storage_t* out_t, float* inv_scale) {                                                            \
    nanochat::cast_body<T, format, P>(x, rows, cols, fp8_max, amax, out, out_t, inv_scale);                            \
  }

NANOCHAT_FP8_AMAX(nanochat_fp8_amax_bf16, __nv_bfloat16, nanochat::Identity)
NANOCHAT_FP8_AMAX(nanochat_fp8_amax_f32, float, nanochat::Identity)
NANOCHAT_FP8_AMAX(nanochat_fp8_amax_relu_square_bf16, __nv_bfloat16, nanochat::ReluSquare)
NANOCHAT_FP8_CAST(nanochat_fp8_cast_e4m3_bf16, __nv_bfloat16, __NV_E4M3, nanochat::Identity)
NANOCHAT_FP8_CAST(nanochat_fp8_cast_e5m2_bf16, __nv_bfloat16, __NV_E5M2, nanochat::Identity)
NANOCHAT_FP8_CAST(nanochat_fp8_cast_e4m3_f32, float, __NV_E4M3, nanochat::Identity)
NANOCHAT_FP8_CAST(nanochat_fp8_cast_e5m2_f32, float, __NV_E5M2, nanochat::Identity)
NANOCHAT_FP8_CAST(nanochat_fp8_cast_e4m3_relu_square_bf16, __nv_bfloat16, __NV_E4M3, nanochat::ReluSquare)

#define NANOCHAT_MX(name, T, P)                                                                                        \
  __global__ void __launch_bounds__(nanochat::kMxThreads)                                                              \
        name(const T* x, int64_t cols, nanochat::MxOutDev out, nanochat::MxOutDev out_t) {                             \
    nanochat::mx_body(nanochat::MxLoad<T, P>{x, cols}, out, out_t);                                                    \
  }

NANOCHAT_MX(nanochat_mx_quantize_bf16, __nv_bfloat16, nanochat::Identity)
NANOCHAT_MX(nanochat_mx_quantize_f32, float, nanochat::Identity)
NANOCHAT_MX(nanochat_mx_quantize_relu_square_bf16, __nv_bfloat16, nanochat::ReluSquare)

__global__ void __launch_bounds__(nanochat::kMxThreads) nanochat_mx_quantize_relu_square_bwd_bf16(
      const __nv_bfloat16* g, const __nv_bfloat16* h, int64_t cols, nanochat::MxOutDev out, nanochat::MxOutDev out_t) {
  nanochat::mx_body(nanochat::MxLoadReluSquareGrad{g, h, cols}, out, out_t);
}

namespace nanochat::kernels {

namespace {

template <typename T>
using AmaxKernel = void (*)(const T*, int64_t, float*);

template <typename T>
using CastKernel = void (*)(
      const T*, int64_t, int64_t, float, const float*, __nv_fp8_storage_t*, __nv_fp8_storage_t*, float*);

// amax_kernel null: *amax is already set
template <typename T>
void launch(
      const T* x, int64_t rows, int64_t cols, float fp8_max, AmaxKernel<T> amax_kernel, CastKernel<T> cast_kernel,
      void* out, void* out_t, float* amax, float* inv_scale, cudaStream_t stream) {
  if (amax_kernel != nullptr) {
    int sms = 0, device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int64_t n = rows * cols;
    const int64_t needed = (n / Vec<T>::kSize + kThreads - 1) / kThreads;
    const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sms)));
    cudaMemsetAsync(amax, 0, sizeof(float), stream);
    amax_kernel<<<blocks, kThreads, 0, stream>>>(x, n, amax);
  }
  const dim3 grid(static_cast<unsigned>((cols + kTile - 1) / kTile), static_cast<unsigned>((rows + kTile - 1) / kTile));
  cast_kernel<<<grid, kTileThreads, 0, stream>>>(
        x, rows, cols, fp8_max, amax, static_cast<__nv_fp8_storage_t*>(out), static_cast<__nv_fp8_storage_t*>(out_t),
        inv_scale);
}

} // namespace

void quantize_fp8(
      const void* x, bool x_bf16, int64_t rows, int64_t cols, Fp8Format format, void* out, void* out_t, float* amax,
      float* inv_scale, cudaStream_t stream, bool amax_ready) {
  const bool e4m3 = format == Fp8Format::E4M3;
  const float fp8_max = e4m3 ? 448.f : 57344.f;
  if (x_bf16) {
    const auto* xb = static_cast<const __nv_bfloat16*>(x);
    launch(
          xb, rows, cols, fp8_max, amax_ready ? nullptr : nanochat_fp8_amax_bf16,
          e4m3 ? nanochat_fp8_cast_e4m3_bf16 : nanochat_fp8_cast_e5m2_bf16, out, out_t, amax, inv_scale, stream);
  }
  else {
    const auto* xf = static_cast<const float*>(x);
    launch(
          xf, rows, cols, fp8_max, amax_ready ? nullptr : nanochat_fp8_amax_f32,
          e4m3 ? nanochat_fp8_cast_e4m3_f32 : nanochat_fp8_cast_e5m2_f32, out, out_t, amax, inv_scale, stream);
  }
}

void quantize_fp8_relu_square(
      const void* h, int64_t rows, int64_t cols, void* out, void* out_t, float* amax, float* inv_scale,
      cudaStream_t stream) {
  launch(
        static_cast<const __nv_bfloat16*>(h), rows, cols, 448.f, nanochat_fp8_amax_relu_square_bf16,
        nanochat_fp8_cast_e4m3_relu_square_bf16, out, out_t, amax, inv_scale, stream);
}

void quantize_mx(
      const void* x, bool x_bf16, int64_t rows, int64_t cols, MxOut out, MxOut out_t, bool relu_square,
      cudaStream_t stream) {
  const dim3 grid = mx_grid(rows, cols);
  if (!x_bf16)
    nanochat_mx_quantize_f32<<<grid, kMxThreads, 0, stream>>>(
          static_cast<const float*>(x), cols, mx_dev(out), mx_dev(out_t));
  else if (relu_square)
    nanochat_mx_quantize_relu_square_bf16<<<grid, kMxThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x), cols, mx_dev(out), mx_dev(out_t));
  else
    nanochat_mx_quantize_bf16<<<grid, kMxThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x), cols, mx_dev(out), mx_dev(out_t));
}

void quantize_mx_relu_square_bwd(
      const void* g, const void* h, int64_t rows, int64_t cols, MxOut out, MxOut out_t, cudaStream_t stream) {
  nanochat_mx_quantize_relu_square_bwd_bf16<<<mx_grid(rows, cols), kMxThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(g), static_cast<const __nv_bfloat16*>(h), cols, mx_dev(out), mx_dev(out_t));
}

} // namespace nanochat::kernels
