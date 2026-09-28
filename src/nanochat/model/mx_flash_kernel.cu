// MXFP8 flash attention forward on sm_120a. Layout and tricks after SageAttention2's sm89 FP8 kernel
// (external/SageAttention/csrc/qattn, Apache-2.0): 4 warps x 32 query rows, 64-key tiles, cp.async, Vᵀ tokens
// permuted so P's accumulator is the next mma's A operand without shuffles. The P scale is LP-FA4's: exp2 offset
// by +8 (P stored as 256 p in e4m3), a constant 2^-8 ue8m0 scale in the mma.
#include "nanochat/model/mx_flash_kernel.h"

#include <type_traits>

#include <cuda_bf16.h>

#include "nanochat/model/mx_flash.cuh"

namespace nanochat {

namespace {

constexpr int kD = kernels::kMxFlashHeadDim;
constexpr int kBlockM = kernels::kMxFlashBlockM;
constexpr int kBlockN = mx_flash::kVtTokens;
constexpr int kWarps = 4; // 32 query rows each
constexpr int kThreads = kWarps * 32;
constexpr int kMTiles = 2;           // m16 tiles per warp
constexpr int kNTiles = kBlockN / 8; // n8 tiles of S
constexpr int kDTiles = kD / 8;      // n8 tiles of O
constexpr int kKSteps = kD / 32;     // k32 steps of Q·Kᵀ, one MX block each
constexpr int kPSteps = kBlockN / 32;
constexpr uint32_t kPScale = 0x77777777u;      // ue8m0 2^-8 in every byte
constexpr float kRescaleThreshold = 0.807355f; // log2(448 / 256)
constexpr float kNegBig = -1e30f;              // running max before any key (finite: exp2(m - m_new) stays 0, not nan)

// shared memory (bytes): Q, K, Vᵀ as swizzled 16-byte chunks, then K and V scales. O's bf16 staging reuses Q, K, Vᵀ.
constexpr int kSmemK = kBlockM * kD;
constexpr int kSmemV = kSmemK + kBlockN * kD;
constexpr int kSmemKs = kSmemV + kD * kBlockN;
constexpr int kSmemVs = kSmemKs + kBlockN * 4;
constexpr int kSmemBytes = kSmemVs + kD * 2;
static_assert(kBlockM * kD * 2 <= kSmemKs, "O staging overlaps the scales");

// chunk c of row r, for 128-byte rows (Q, K), 64-byte rows (Vᵀ), 256-byte rows (O): conflict-free ldmatrix/stores
__device__ __forceinline__ uint32_t swz128(int r, int c) {
  return r * 128 + ((c ^ (r & 7)) << 4);
}

__device__ __forceinline__ uint32_t swz64(int r, int c) {
  return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
}

__device__ __forceinline__ uint32_t swz256(int r, int c) {
  return r * 256 + ((c ^ (r & 7)) << 4);
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

// d += (A 2^sfa) (B 2^sfb), e4m3 m16n8k32. sfa: row lane/4 + 8 (lane % 2)'s exponents, sfb: column lane/4's; the
// k-step's exponent is byte byte_a / byte_b.
__device__ __forceinline__ void mma_mx(
      float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb, uint16_t byte_a,
      uint16_t byte_b) {
  const uint16_t tid = 0;
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(byte_a), "h"(tid),
                 "r"(sfb), "h"(byte_b), "h"(tid));
}

using mx_flash::pack_e4m3;

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void mx_flash_fwd_body(
      const uint8_t* __restrict__ q, const uint32_t* __restrict__ q_scale, const uint8_t* __restrict__ k,
      const uint32_t* __restrict__ k_scale, const uint8_t* __restrict__ vt, const uint8_t* __restrict__ v_scale,
      __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int T, int H, int Hkv, int window, float scale_log2) {
  __shared__ __align__(128) uint8_t smem[kSmemBytes];
  const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
  const int m0 = (static_cast<int>(gridDim.x - 1 - blockIdx.x)) * kBlockM; // most keys first
  const int h = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z), hk = h / (H / Hkv);
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

  const int64_t q_ld = static_cast<int64_t>(H) * kD, k_ld = static_cast<int64_t>(Hkv) * kD;
  const uint8_t* q_blk = q + ((static_cast<int64_t>(b) * T + m0) * H + h) * kD;
  const uint8_t* k_head = k + (static_cast<int64_t>(b) * T * Hkv + hk) * kD;
  const int64_t bh_kv = static_cast<int64_t>(b) * Hkv + hk;
  const uint8_t* vt_head = vt + bh_kv * kD * T;
  const uint32_t* ks_head = k_scale + bh_kv * T;
  const uint8_t* vs_head = v_scale + bh_kv * T * 4; // (T / 64) tiles of 128 x 2 bytes

#pragma unroll
  for (int i = 0; i < kBlockM * 8 / kThreads; ++i) {
    const int idx = tid + i * kThreads, r = idx / 8, c = idx % 8;
    cp_async16(sbase + swz128(r, c), q_blk + r * q_ld + c * 16);
  }
  cp_async_commit();
  const auto load_k = [&](int n) {
#pragma unroll
    for (int i = 0; i < kBlockN * 8 / kThreads; ++i) {
      const int idx = tid + i * kThreads, r = idx / 8, c = idx % 8;
      cp_async16(sbase + kSmemK + swz128(r, c), k_head + (static_cast<int64_t>(n) * kBlockN + r) * k_ld + c * 16);
    }
    if (tid < kBlockN * 4 / 16)
      cp_async16(sbase + kSmemKs + tid * 16, ks_head + static_cast<int64_t>(n) * kBlockN + tid * 4);
    cp_async_commit();
  };
  const auto load_v = [&](int n) {
#pragma unroll
    for (int i = 0; i < kD * 4 / kThreads; ++i) {
      const int idx = tid + i * kThreads, r = idx / 4, c = idx % 4;
      cp_async16(sbase + kSmemV + swz64(r, c), vt_head + static_cast<int64_t>(r) * T + n * kBlockN + c * 16);
    }
    if (tid < kD * 2 / 16)
      cp_async16(sbase + kSmemVs + tid * 16, vs_head + static_cast<int64_t>(n) * kD * 2 + tid * 16);
    cp_async_commit();
  };

  uint32_t sfa[kMTiles];
  const int row0 = m0 + warp * 32 + lane / 4; // row of this thread's accumulator slot 0 in m-tile 0
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt)
    sfa[mt] = q_scale[(static_cast<int64_t>(b) * H + h) * T + row0 + mt * 16 + 8 * (lane % 2)];

  const int n_hi = (m0 + kBlockM - 1) / kBlockN;
  const int n_lo = window >= 0 ? max(0, m0 - window) / kBlockN : 0;

  float o[kMTiles][kDTiles][4];
  float m[kMTiles][2], l[kMTiles][2];
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int dt = 0; dt < kDTiles; ++dt)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        o[mt][dt][j] = 0.0f;
#pragma unroll
    for (int r = 0; r < 2; ++r)
      m[mt][r] = kNegBig, l[mt][r] = 0.0f;
  }

  load_k(n_lo);
  load_v(n_lo);
  // one key tile; kMasked: some keys of it are outside the causal window of some rows
  const auto step = [&](int n, auto masked) {
    constexpr bool kMasked = decltype(masked)::value;
    cp_async_wait<1>(); // Q, K_n
    __syncthreads();

    // S = Q Kᵀ (dequantized by the mma)
    float s[kMTiles][kNTiles][4];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt)
#pragma unroll
      for (int nt = 0; nt < kNTiles; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          s[mt][nt][j] = 0.0f;
    uint32_t sfb[kNTiles];
    const auto* ks_s = reinterpret_cast<const uint32_t*>(smem + kSmemKs);
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt)
      sfb[nt] = ks_s[nt * 8 + lane / 4];
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
      uint32_t a[kMTiles][4];
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt)
        ldmatrix_x4(a[mt], sbase + swz128(warp * 32 + mt * 16 + lane % 16, 2 * ks + lane / 16));
#pragma unroll
      for (int np = 0; np < kNTiles / 2; ++np) {
        uint32_t bk[4];
        ldmatrix_x4(bk, sbase + kSmemK + swz128(np * 16 + lane % 8 + (lane / 16) * 8, 2 * ks + (lane / 8) % 2));
#pragma unroll
        for (int mt = 0; mt < kMTiles; ++mt) {
          mma_mx(s[mt][2 * np], a[mt], bk[0], bk[1], sfa[mt], sfb[2 * np], ks, ks);
          mma_mx(s[mt][2 * np + 1], a[mt], bk[2], bk[3], sfa[mt], sfb[2 * np + 1], ks, ks);
        }
      }
    }
    __syncthreads(); // K consumed
    if (n < n_hi)
      load_k(n + 1);
    else
      cp_async_commit(); // empty group: the wait below still means "V_n landed"

    // online softmax in base 2, P = 256 exp2(x - m) in [0, 448]
    float alpha[kMTiles][2];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int row = row0 + mt * 16 + 8 * r;
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float& x = s[mt][nt][2 * r + e];
            if constexpr (kMasked) {
              const int key = n * kBlockN + nt * 8 + 2 * (lane % 4) + e;
              if (key > row || (window >= 0 && row - key > window))
                x = -INFINITY;
            }
            mx = fmaxf(mx, x);
          }
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        // lazy rescale (LP-FA4): keep the max while P stays within e4m3 (256 * 2^0.807 = 448)
        const float m_new = mx * scale_log2 > m[mt][r] + kRescaleThreshold ? mx * scale_log2 : m[mt][r];
        alpha[mt][r] = fast_exp2(m[mt][r] - m_new);
        m[mt][r] = m_new;
        l[mt][r] *= alpha[mt][r];
        const float bias = 8.0f - m_new;
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float& x = s[mt][nt][2 * r + e];
            x = fast_exp2(fmaf(x, scale_log2, bias));
            l[mt][r] += x;
          }
        }
      }
    }

    bool rescale = false;
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt)
      rescale |= alpha[mt][0] != 1.0f || alpha[mt][1] != 1.0f;
    if (__any_sync(0xffffffffu, rescale)) {
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt)
#pragma unroll
        for (int dt = 0; dt < kDTiles; ++dt)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            o[mt][dt][j] *= alpha[mt][j / 2];
    }

    // P as the A operand: row g holds keys {2t, 2t+1, 8+2t, 9+2t} of each 16, Vᵀ's tokens are stored in that order
    uint32_t pa[kMTiles][kPSteps][4];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
      for (int kk = 0; kk < kPSteps; ++kk) {
        const float (&s0)[4] = s[mt][4 * kk], (&s1)[4] = s[mt][4 * kk + 1];
        const float (&s2)[4] = s[mt][4 * kk + 2], (&s3)[4] = s[mt][4 * kk + 3];
        pa[mt][kk][0] = pack_e4m3(s0[0], s0[1], s1[0], s1[1]);
        pa[mt][kk][1] = pack_e4m3(s0[2], s0[3], s1[2], s1[3]);
        pa[mt][kk][2] = pack_e4m3(s2[0], s2[1], s3[0], s3[1]);
        pa[mt][kk][3] = pack_e4m3(s2[2], s2[3], s3[2], s3[3]);
      }
    }

    cp_async_wait<1>(); // V_n
    __syncthreads();
    // O += P V
    const auto* vs_s = reinterpret_cast<const uint16_t*>(smem + kSmemVs);
#pragma unroll
    for (int dp = 0; dp < kDTiles / 2; ++dp) {
      const uint32_t sfv0 = vs_s[dp * 16 + lane / 4], sfv1 = vs_s[dp * 16 + 8 + lane / 4];
#pragma unroll
      for (int kk = 0; kk < kPSteps; ++kk) {
        uint32_t bv[4];
        ldmatrix_x4(bv, sbase + kSmemV + swz64(dp * 16 + lane % 8 + (lane / 16) * 8, 2 * kk + (lane / 8) % 2));
#pragma unroll
        for (int mt = 0; mt < kMTiles; ++mt) {
          mma_mx(o[mt][2 * dp], pa[mt][kk], bv[0], bv[1], kPScale, sfv0, 0, kk);
          mma_mx(o[mt][2 * dp + 1], pa[mt][kk], bv[2], bv[3], kPScale, sfv1, 0, kk);
        }
      }
    }
    __syncthreads(); // V consumed
    if (n < n_hi)
      load_v(n + 1);
  };
  for (int n = n_lo; n <= n_hi; ++n)
    if (n * kBlockN + kBlockN - 1 > m0 || (window >= 0 && n * kBlockN < m0 + kBlockM - 1 - window))
      step(n, std::true_type{});
    else
      step(n, std::false_type{});

  // O = acc 256 / l; lse = ln(sum exp(s / sqrt(D))) = ln2 (m + log2(l) - 8)
  float inv[kMTiles][2];
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float sum = l[mt][r];
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      inv[mt][r] = 256.0f / sum;
      if (lane % 4 == 0)
        lse[(static_cast<int64_t>(b) * H + h) * T + row0 + mt * 16 + 8 * r] = 0.69314718f *
                                                                              (m[mt][r] + __log2f(sum) - 8.0f);
    }
  }
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = warp * 32 + mt * 16 + lane / 4 + 8 * r;
#pragma unroll
      for (int dt = 0; dt < kDTiles; ++dt) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(
              o[mt][dt][2 * r] * inv[mt][r], o[mt][dt][2 * r + 1] * inv[mt][r]);
        *reinterpret_cast<__nv_bfloat162*>(smem + swz256(row, dt) + (lane % 4) * 4) = v;
      }
    }
  }
  __syncthreads();
  auto* out_blk = reinterpret_cast<uint8_t*>(out) + ((static_cast<int64_t>(b) * T + m0) * H + h) * kD * 2;
#pragma unroll
  for (int i = 0; i < kBlockM * 16 / kThreads; ++i) {
    const int idx = tid + i * kThreads, r = idx / 16, c = idx % 16;
    *reinterpret_cast<uint4*>(out_blk + r * q_ld * 2 + c * 16) = *reinterpret_cast<const uint4*>(smem + swz256(r, c));
  }
}

constexpr int kQuantWarps = 8;

// one warp per (token, head) row: 4 values per lane, 8 lanes per MX block
__device__ __forceinline__ void mx_flash_quantize_rows_body(
      const __nv_bfloat16* __restrict__ x, uint32_t* __restrict__ data, uint32_t* __restrict__ scale, int64_t rows,
      int64_t T, int heads, int64_t x_stride) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kQuantWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int lane = static_cast<int>(threadIdx.x % 32);
  const int64_t bt = row / heads;
  const int h = static_cast<int>(row % heads);
  const uint2 raw = *reinterpret_cast<const uint2*>(x + bt * x_stride + static_cast<int64_t>(h) * kD + lane * 4);
  const auto* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
  const float2 v01 = __bfloat1622float2(p[0]), v23 = __bfloat1622float2(p[1]);
  float amax = fmaxf(fmaxf(fabsf(v01.x), fabsf(v01.y)), fmaxf(fabsf(v23.x), fabsf(v23.y)));
#pragma unroll
  for (int off = 1; off < 8; off *= 2)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const int e = mx_exponent(amax);
  const float mul = mx_multiplier(e);
  data[row * (kD / 4) + lane] = pack_e4m3(v01.x * mul, v01.y * mul, v23.x * mul, v23.y * mul);
  uint32_t word = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    word |= static_cast<uint32_t>(__shfl_sync(0xffffffffu, e, 8 * j)) << (8 * j);
  if (lane == 0)
    scale[(bt / T * heads + h) * T + bt % T] = word;
}

// one block per (64 tokens, head, batch)
__device__ __forceinline__ void mx_flash_quantize_vt_body(
      const __nv_bfloat16* __restrict__ v, uint8_t* __restrict__ vt, uint8_t* __restrict__ scale, int64_t T, int heads,
      int64_t v_stride) {
  __shared__ __align__(16) mx_flash::VtTile tile;
  const int n = static_cast<int>(blockIdx.x), h = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int i = 0; i < kBlockN * kD / 8 / mx_flash::kVtThreads; ++i) {
    const int idx = tid + i * mx_flash::kVtThreads, t = idx / (kD / 8), c = idx % (kD / 8);
    const auto* src = v + (static_cast<int64_t>(b) * T + static_cast<int64_t>(n) * kBlockN + t) * v_stride +
                      static_cast<int64_t>(h) * kD + c * 8;
    *reinterpret_cast<uint4*>(&tile[t][c * 8]) = *reinterpret_cast<const uint4*>(src);
  }
  __syncthreads();
  mx_flash::store_vt_tile(tile, vt, scale, static_cast<int64_t>(b) * heads + h, n, T);
}

} // namespace

} // namespace nanochat

// Kernels: global, non-template nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
__global__ void __launch_bounds__(nanochat::kThreads) nanochat_mx_flash_fwd(
      const uint8_t* __restrict__ q, const uint32_t* __restrict__ q_scale, const uint8_t* __restrict__ k,
      const uint32_t* __restrict__ k_scale, const uint8_t* __restrict__ vt, const uint8_t* __restrict__ v_scale,
      __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int T, int H, int Hkv, int window, float scale_log2) {
  nanochat::mx_flash_fwd_body(q, q_scale, k, k_scale, vt, v_scale, out, lse, T, H, Hkv, window, scale_log2);
}

__global__ void __launch_bounds__(nanochat::kQuantWarps * 32) nanochat_mx_flash_quantize_rows(
      const __nv_bfloat16* __restrict__ x, uint32_t* __restrict__ data, uint32_t* __restrict__ scale, int64_t rows,
      int64_t T, int heads, int64_t x_stride) {
  nanochat::mx_flash_quantize_rows_body(x, data, scale, rows, T, heads, x_stride);
}

__global__ void __launch_bounds__(nanochat::mx_flash::kVtThreads) nanochat_mx_flash_quantize_vt(
      const __nv_bfloat16* __restrict__ v, uint8_t* __restrict__ vt, uint8_t* __restrict__ scale, int64_t T, int heads,
      int64_t v_stride) {
  nanochat::mx_flash_quantize_vt_body(v, vt, scale, T, heads, v_stride);
}

namespace nanochat::kernels {

void mx_flash_quantize_rows(
      const void* x, void* data, uint32_t* scale, int B, int64_t T, int heads, int64_t x_stride, cudaStream_t stream) {
  const int64_t rows = static_cast<int64_t>(B) * T * heads;
  nanochat_mx_flash_quantize_rows<<<(rows + kQuantWarps - 1) / kQuantWarps, kQuantWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<uint32_t*>(data), scale, rows, T, heads, x_stride);
}

void mx_flash_quantize_vt(
      const void* v, void* vt, uint8_t* scale, int B, int64_t T, int heads, int64_t v_stride, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(T / kBlockN), heads, B);
  nanochat_mx_flash_quantize_vt<<<grid, mx_flash::kVtThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(v), static_cast<uint8_t*>(vt), scale, T, heads, v_stride);
}

void mx_flash_fwd(
      const void* q, const uint32_t* q_scale, const void* k, const uint32_t* k_scale, const void* vt,
      const uint8_t* v_scale, void* out, float* lse, int B, int64_t T, int H, int Hkv, int64_t window,
      cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(T / kBlockM), H, B);
  const float scale_log2 = 1.4426950408889634f / sqrtf(static_cast<float>(kD));
  nanochat_mx_flash_fwd<<<grid, kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(q), q_scale, static_cast<const uint8_t*>(k), k_scale,
        static_cast<const uint8_t*>(vt), v_scale, static_cast<__nv_bfloat16*>(out), lse, static_cast<int>(T), H, Hkv,
        window >= 0 && window < T ? static_cast<int>(window) : -1, scale_log2);
}

} // namespace nanochat::kernels
