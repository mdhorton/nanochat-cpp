// MXFP8 quantization shared by the CUDA kernels that write MX directly (included by .cu files only).
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "nanochat/model/fp8/fp8_kernel.h"

namespace nanochat {

// 8 bf16 or fp32 values from 16-byte aligned p
template <typename T>
__device__ __forceinline__ void load8(const T* p, float (&v)[8]) {
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

// MXFP8: biased e8m0 exponent of 2^ceil(log2(amax / 448)) (448 = 1.75 * 2^8), exact integer math
__device__ __forceinline__ int mx_exponent(float amax) {
  const uint32_t bits = __float_as_uint(amax);
  const int e = static_cast<int>(bits >> 23) - 8 + ((bits & 0x7fffffu) > 0x600000u ? 1 : 0);
  return min(max(e, 0), 253);
}

// 2^-(e - 127), exact
__device__ __forceinline__ float mx_multiplier(int e) {
  return __uint_as_float(static_cast<uint32_t>(254 - e) << 23);
}

// cuBLAS's swizzled scale layout: 128x4 tiles of (row, block), 512 bytes each, row-major over tiles
__device__ __forceinline__ int64_t mx_scale_index(int64_t row, int64_t block, int64_t tiles) {
  return ((row >> 7) * tiles + (block >> 2)) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (block & 3);
}

struct MxOutDev {
  __nv_fp8_storage_t* data;
  int64_t ld;
  uint8_t* scale;
  int64_t scale_tiles;
};

__device__ __forceinline__ void mx_store8(
      const float (&v)[8], float amax, const MxOutDev& o, int64_t row, int64_t col, int64_t scale_row,
      bool write_scale) {
  const int e = mx_exponent(amax);
  const float mul = mx_multiplier(e);
  __align__(8) __nv_fp8_storage_t q[8];
#pragma unroll
  for (int k = 0; k < 8; ++k)
    q[k] = __nv_cvt_float_to_fp8(v[k] * mul, __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint2*>(o.data + row * o.ld + col) = *reinterpret_cast<const uint2*>(q);
  if (write_scale)
    o.scale[mx_scale_index(scale_row, col / 32, o.scale_tiles)] = static_cast<uint8_t>(e);
}

// 32x64 tiles, 256 threads. Rows: each thread quantizes 8 values, 4 threads per 32-value block. Columns: from the
// tile in shared memory, each thread 8 rows of a column, 4 threads per block.
constexpr int kMxRows = 32, kMxCols = 64, kMxThreads = 256;

// load(row, col, v): the 8 input values at (row, col..col + 7). Grid: (cols / kMxCols, rows / kMxRows).
template <typename L>
__device__ __forceinline__ void mx_body(L load, MxOutDev out, MxOutDev out_t) {
  __shared__ float tile[kMxRows][kMxCols + 1];
  const int t = static_cast<int>(threadIdx.x);
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kMxRows, col0 = static_cast<int64_t>(blockIdx.x) * kMxCols;
  {
    const int r = t / 8, c = t % 8 * 8;
    float v[8];
    load(row0 + r, col0 + c, v);
    float m = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      m = fmaxf(m, fabsf(v[k]));
      tile[r][c + k] = v[k];
    }
    if (out.data != nullptr) {
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      mx_store8(v, m, out, row0 + r, col0 + c, row0 + r, t % 4 == 0);
    }
  }
  if (out_t.data == nullptr)
    return;
  __syncthreads();
  const int c = t / 4, r = t % 4 * 8;
  float v[8];
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    v[k] = tile[r + k][c];
    m = fmaxf(m, fabsf(v[k]));
  }
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
  mx_store8(v, m, out_t, col0 + c, row0 + r, col0 + c, t % 4 == 0);
}

// Quantizes a kMxRows x kCols tile of shared memory both ways, as mx_body: out from (row0, col0), out_t from (col0,
// row0). All kMxThreads threads, after a __syncthreads().
template <int kCols>
__device__ __forceinline__ void mx_store_tile(
      const float (*tile)[kCols + 1], int64_t row0, int64_t col0, const MxOutDev& out, const MxOutDev& out_t) {
  static_assert(kCols % 64 == 0, "whole 8-value runs for every thread");
  const int t = static_cast<int>(threadIdx.x);
  for (int q = t; q < kMxRows * kCols / 8; q += kMxThreads) {
    const int r = q / (kCols / 8), c = q % (kCols / 8) * 8;
    float v[8], m = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      v[k] = tile[r][c + k];
      m = fmaxf(m, fabsf(v[k]));
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    mx_store8(v, m, out, row0 + r, col0 + c, row0 + r, q % 4 == 0);
  }
  for (int q = t; q < kCols * kMxRows / 8; q += kMxThreads) {
    const int c = q / 4, r = q % 4 * 8;
    float v[8], m = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      v[k] = tile[r + k][c];
      m = fmaxf(m, fabsf(v[k]));
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    mx_store8(v, m, out_t, col0 + c, row0 + r, col0 + c, q % 4 == 0);
  }
}

inline MxOutDev mx_dev(const kernels::MxOut& o) {
  return {static_cast<__nv_fp8_storage_t*>(o.data), o.ld, static_cast<uint8_t*>(o.scale), o.scale_tiles};
}

inline dim3 mx_grid(int64_t rows, int64_t cols) {
  return {static_cast<unsigned>(cols / kMxCols), static_cast<unsigned>(rows / kMxRows)};
}

} // namespace nanochat
