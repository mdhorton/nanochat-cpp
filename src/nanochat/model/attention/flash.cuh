// bf16 flash attention's device helpers, shared by the forward and backward kernels (included by .cu files only).
#pragma once

#include <cstdint>

#include <cuda_bf16.h>

#include "nanochat/model/attention/flash_kernel.h"

namespace nanochat::flash {

constexpr int kD = kernels::kFlashHeadDim;
constexpr int kRowBytes = kD * 2; // a bf16 row of Q, K, V, O or their gradients: 16 chunks of 16 bytes

// chunk c of row r (256-byte rows): conflict-free ldmatrix, ldmatrix.trans and 4-byte stores
__device__ __forceinline__ uint32_t swz(int r, int c) {
  return r * kRowBytes + ((c ^ (r & 7)) << 4);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// d += a b, bf16 m16n8k16, fp32 accumulate
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// rows x 128 bf16 from swizzled shared memory to global (ld elements apart)
template <int Threads>
__device__ __forceinline__ void store_rows(const uint8_t* src, __nv_bfloat16* dst, int64_t ld, int rows) {
  for (int idx = static_cast<int>(threadIdx.x); idx < rows * 16; idx += Threads) {
    const int r = idx / 16, c = idx % 16;
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(dst + r * ld) + c * 16) = *reinterpret_cast<const uint4*>(
          src + swz(r, c));
  }
}

// an m16 x 128 fp32 accumulator (16 n8 tiles), times scale, as bf16 into rows row0.. of swizzled shared memory
__device__ __forceinline__ void stage_acc(uint8_t* smem, int row0, const float (&acc)[kD / 8][4], float scale) {
  const int lane = static_cast<int>(threadIdx.x % 32);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = row0 + lane / 4 + 8 * r;
#pragma unroll
    for (int dt = 0; dt < kD / 8; ++dt)
      *reinterpret_cast<uint32_t*>(smem + swz(row, dt) + (lane % 4) * 4) = pack_bf16(
            acc[dt][2 * r] * scale, acc[dt][2 * r + 1] * scale);
  }
}

} // namespace nanochat::flash
