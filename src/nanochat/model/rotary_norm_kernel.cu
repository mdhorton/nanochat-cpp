#include "nanochat/model/rotary_norm_kernel.h"

#include <cuda_bf16.h>

#include "nanochat/model/mx_flash.cuh"
#include "nanochat/model/rotary_norm.cuh"

namespace nanochat {

namespace {

using namespace rotary;

constexpr int kWarps = 8; // rows per block, one warp each

__device__ __forceinline__ void rotary_norm_fwd_body(
      const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin, __nv_bfloat16* out, float* rstd,
      int64_t rows, int heads, int64_t seq_len, int head_dim, int64_t x_stride, float scale, float eps,
      uint8_t* mx_data, uint32_t* mx_scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int lane = static_cast<int>(threadIdx.x % 32), half = head_dim / 2;
  const int64_t t = row / heads % seq_len;
  Rotated y;
  const float ss = warp_sum(rotate(x_row(x, row, heads, head_dim, x_stride), cos + t * half, sin + t * half, half, y));
  const float r = rsqrtf(ss / static_cast<float>(head_dim) + eps);
  if (lane == 0)
    rstd[row] = r;
  auto* o = out + row * head_dim;
  const float k = r * scale;
#pragma unroll
  for (int i = 0; i < kMaxIters; ++i) {
    const int j = 2 * lane + 64 * i;
    if (j >= half)
      break;
    store2(o + j, y.y1[i][0] * k, y.y1[i][1] * k);
    store2(o + half + j, y.y2[i][0] * k, y.y2[i][1] * k);
  }
  if (mx_data != nullptr) { // head_dim 128: one column pair per half, quantized as rounded to bf16 in out
    const auto rb = [](float v) {
      return __bfloat162float(__float2bfloat16(v));
    };
    const int64_t b = row / heads / seq_len;
    mx_flash::store_row_halves(
          rb(y.y1[0][0] * k), rb(y.y1[0][1] * k), rb(y.y2[0][0] * k), rb(y.y2[0][1] * k), mx_data + row * head_dim,
          mx_scale + (b * heads + row % heads) * seq_len + t);
  }
}

__device__ __forceinline__ void rotary_norm_bwd_body(
      const __nv_bfloat16* dout, const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
      const float* rstd, __nv_bfloat16* dx, int64_t rows, int heads, int64_t seq_len, int head_dim, int64_t x_stride,
      float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int half = head_dim / 2;
  const int64_t t = row / heads % seq_len;
  auto* d = dx + row * head_dim;
  bwd_row(
        dout + row * head_dim, x_row(x, row, heads, head_dim, x_stride), cos + t * half, sin + t * half, rstd[row],
        head_dim, scale, [&](int j, float a, float b) {
          store2(d + j, a, b);
        });
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kWarps * 32) nanochat_rotary_norm_fwd(
      const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin, __nv_bfloat16* out, float* rstd,
      int64_t rows, int heads, int64_t seq_len, int head_dim, int64_t x_stride, float scale, float eps,
      uint8_t* mx_data, uint32_t* mx_scale) {
  nanochat::rotary_norm_fwd_body(
        x, cos, sin, out, rstd, rows, heads, seq_len, head_dim, x_stride, scale, eps, mx_data, mx_scale);
}

__global__ void __launch_bounds__(nanochat::kWarps * 32) nanochat_rotary_norm_bwd(
      const __nv_bfloat16* dout, const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
      const float* rstd, __nv_bfloat16* dx, int64_t rows, int heads, int64_t seq_len, int head_dim, int64_t x_stride,
      float scale) {
  nanochat::rotary_norm_bwd_body(dout, x, cos, sin, rstd, dx, rows, heads, seq_len, head_dim, x_stride, scale);
}

namespace nanochat::kernels {

namespace {

using bf16 = __nv_bfloat16;

unsigned blocks(int64_t rows) {
  return static_cast<unsigned>((rows + kWarps - 1) / kWarps);
}

} // namespace

void rotary_norm_fwd(
      const void* x, const void* cos, const void* sin, void* out, float* rstd, int64_t rows, int heads, int64_t seq_len,
      int head_dim, int64_t x_stride, float scale, float eps, cudaStream_t stream, void* mx_data, uint32_t* mx_scale) {
  nanochat_rotary_norm_fwd<<<blocks(rows), kWarps * 32, 0, stream>>>(
        static_cast<const bf16*>(x), static_cast<const bf16*>(cos), static_cast<const bf16*>(sin),
        static_cast<bf16*>(out), rstd, rows, heads, seq_len, head_dim, x_stride, scale, eps,
        static_cast<uint8_t*>(mx_data), mx_scale);
}

void rotary_norm_bwd(
      const void* dout, const void* x, const void* cos, const void* sin, const float* rstd, void* dx, int64_t rows,
      int heads, int64_t seq_len, int head_dim, int64_t x_stride, float scale, cudaStream_t stream) {
  nanochat_rotary_norm_bwd<<<blocks(rows), kWarps * 32, 0, stream>>>(
        static_cast<const bf16*>(dout), static_cast<const bf16*>(x), static_cast<const bf16*>(cos),
        static_cast<const bf16*>(sin), rstd, static_cast<bf16*>(dx), rows, heads, seq_len, head_dim, x_stride, scale);
}

} // namespace nanochat::kernels
