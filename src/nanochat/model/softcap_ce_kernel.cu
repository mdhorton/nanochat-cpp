#include "nanochat/model/softcap_ce_kernel.h"

#include <cuda_bf16.h>

#include "nanochat/model/mx_kernel.cuh"

namespace nanochat {

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8; // bf16 per 16-byte load

__device__ float block_sum(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (lane == 0)
    smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < kThreads / 32 ? smem[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0)
      smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

// max over the block, then one atomic on the float bits (v >= 0, as fp8_kernel.cu's amax)
__device__ void block_max_to(float v, float* smem, float* out) {
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  __syncthreads(); // smem reuse after block_sum
  if (threadIdx.x % 32 == 0)
    smem[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < kThreads / 32; ++w)
      v = fmaxf(v, smem[w]);
    atomicMax(reinterpret_cast<int*>(out), __float_as_int(v));
  }
}

// d loss_row / d z for one logit z (th = tanh(z / softcap)), before rounding
__device__ __forceinline__ float softcap_ce_grad(float z, bool is_target, float lse, float softcap, float scale) {
  const float th = tanhf(z / softcap);
  const float p = expf(softcap * th - lse);
  return (p - (is_target ? 1.f : 0.f)) * (1.f - th * th) * scale;
}

__device__ __forceinline__ float softcap_ce_grad_scale(
      int64_t row, const float* grad_scale, int64_t grad_scale_stride, const int64_t* num_valid) {
  float scale = grad_scale != nullptr ? grad_scale[row * grad_scale_stride] : 1.f;
  if (num_valid != nullptr)
    scale /= static_cast<float>(*num_valid);
  return scale;
}

// One block per row. |capped| <= softcap, so exp(capped - softcap) needs no running max.
template <bool kGrad>
__device__ __forceinline__ void softcap_ce_body(
      __nv_bfloat16* logits, int64_t ld, const int64_t* targets, int vocab, int padded, float softcap, float* loss,
      const float* grad_scale, int64_t grad_scale_stride, const int64_t* num_valid, float* grad_amax, float* lse_out) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  auto* z = logits + row * ld;
  const int64_t t = targets[row];
  if (t < 0) {
    if (loss != nullptr && threadIdx.x == 0)
      loss[row] = 0.f;
    if constexpr (kGrad)
      for (int j = threadIdx.x * kVec; j < padded; j += kThreads * kVec)
        *reinterpret_cast<uint4*>(z + j) = make_uint4(0, 0, 0, 0);
    return;
  }
  const float z_t = __bfloat162float(z[t]); // read before any thread overwrites the row

  float sum = 0.f;
  for (int j = threadIdx.x * kVec; j < padded; j += kThreads * kVec) {
    const uint4 raw = *reinterpret_cast<const uint4*>(z + j);
    const auto* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int k = 0; k < kVec; ++k)
      if (j + k < vocab)
        sum += expf(softcap * tanhf(__bfloat162float(v[k]) / softcap) - softcap);
  }
  const float lse = softcap + logf(block_sum(sum, smem));
  if (threadIdx.x == 0) {
    if (loss != nullptr)
      loss[row] = lse - softcap * tanhf(z_t / softcap);
    if (lse_out != nullptr)
      lse_out[row] = lse;
  }

  if constexpr (kGrad) {
    const float scale = softcap_ce_grad_scale(row, grad_scale, grad_scale_stride, num_valid);
    float m = 0.f;
    for (int j = threadIdx.x * kVec; j < padded; j += kThreads * kVec) {
      uint4 raw = *reinterpret_cast<const uint4*>(z + j);
      auto* v = reinterpret_cast<__nv_bfloat16*>(&raw);
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        const float g = j + k < vocab ? softcap_ce_grad(__bfloat162float(v[k]), j + k == t, lse, softcap, scale) : 0.f;
        v[k] = __float2bfloat16(g);
        m = fmaxf(m, fabsf(__bfloat162float(v[k])));
      }
      *reinterpret_cast<uint4*>(z + j) = raw;
    }
    if (grad_amax != nullptr)
      block_max_to(m, smem, grad_amax);
  }
}

// mx_body's input: the gradient as the grad kernel writes it (bf16), from the logits and each row's lse
struct MxLoadSoftcapGrad {
  const __nv_bfloat16* logits;
  int64_t ld;
  const int64_t* targets;
  const float* lse;
  int vocab;
  float softcap;
  const float* grad_scale;
  int64_t grad_scale_stride;
  const int64_t* num_valid;

  __device__ void operator()(int64_t row, int64_t col, float (&v)[8]) const {
    const int64_t t = targets[row];
    if (t < 0) {
#pragma unroll
      for (int k = 0; k < 8; ++k)
        v[k] = 0.f;
      return;
    }
    load8(logits + row * ld + col, v);
    const float l = lse[row], scale = softcap_ce_grad_scale(row, grad_scale, grad_scale_stride, num_valid);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float g = col + k < vocab ? softcap_ce_grad(v[k], col + k == t, l, softcap, scale) : 0.f;
      v[k] = __bfloat162float(__float2bfloat16(g));
    }
  }
};

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (ncu drops the innermost namespace).
// Not extern "C": ncu's --filter-mode per-launch-config then confuses kernels of equal launch shape.
#define NANOCHAT_SOFTCAP_CE(name, grad)                                                                                \
  __global__ void __launch_bounds__(nanochat::kThreads) name(                                                          \
        __nv_bfloat16* logits, int64_t ld, const int64_t* targets, int vocab, int padded, float softcap, float* loss,  \
        const float* grad_scale, int64_t grad_scale_stride, const int64_t* num_valid, float* grad_amax, float* lse) {  \
    nanochat::softcap_ce_body<grad>(                                                                                   \
          logits, ld, targets, vocab, padded, softcap, loss, grad_scale, grad_scale_stride, num_valid, grad_amax,      \
          lse);                                                                                                        \
  }

NANOCHAT_SOFTCAP_CE(nanochat_softcap_ce_loss, false)
NANOCHAT_SOFTCAP_CE(nanochat_softcap_ce_loss_grad, true)

__global__ void __launch_bounds__(nanochat::kMxThreads)
      nanochat_softcap_ce_grad_mx(nanochat::MxLoadSoftcapGrad load, nanochat::MxOutDev out, nanochat::MxOutDev out_t) {
  nanochat::mx_body(load, out, out_t);
}

namespace nanochat::kernels {

void softcap_ce(
      void* logits, int64_t rows, int64_t ld, const int64_t* targets, int vocab, int padded, float softcap, float* loss,
      const float* grad_scale, int64_t grad_scale_stride, const int64_t* num_valid, bool grad, float* grad_amax,
      float* lse, cudaStream_t stream) {
  auto* z = static_cast<__nv_bfloat16*>(logits);
  if (grad) {
    if (grad_amax != nullptr)
      cudaMemsetAsync(grad_amax, 0, sizeof(float), stream);
    nanochat_softcap_ce_loss_grad<<<rows, kThreads, 0, stream>>>(
          z, ld, targets, vocab, padded, softcap, loss, grad_scale, grad_scale_stride, num_valid, grad_amax, lse);
  }
  else
    nanochat_softcap_ce_loss<<<rows, kThreads, 0, stream>>>(
          z, ld, targets, vocab, padded, softcap, loss, grad_scale, grad_scale_stride, num_valid, nullptr, lse);
}

void softcap_ce_grad_mx(
      const void* logits, int64_t rows, int64_t ld, const int64_t* targets, const float* lse, int vocab, int padded,
      float softcap, const float* grad_scale, int64_t grad_scale_stride, const int64_t* num_valid, MxOut out,
      MxOut out_t, cudaStream_t stream) {
  const MxLoadSoftcapGrad load{static_cast<const __nv_bfloat16*>(logits),
                               ld,
                               targets,
                               lse,
                               vocab,
                               softcap,
                               grad_scale,
                               grad_scale_stride,
                               num_valid};
  nanochat_softcap_ce_grad_mx<<<mx_grid(rows, padded), kMxThreads, 0, stream>>>(load, mx_dev(out), mx_dev(out_t));
}

} // namespace nanochat::kernels
