// MXFP8 flash attention forward on sm_120a. Layout and tricks after SageAttention2's sm89 FP8 kernel
// (external/SageAttention/csrc/qattn, Apache-2.0): 4 warps x 32 query rows, 64-key tiles, cp.async, Vᵀ tokens
// permuted so P's accumulator is the next mma's A operand without shuffles. The P scale is LP-FA4's: exp2 offset
// by +8 (P stored as 256 p in e4m3), a constant 2^-8 ue8m0 scale in the mma.
#include "nanochat/model/mx_flash_kernel.h"

#include <type_traits>

#include <cuda_bf16.h>

#include "nanochat/model/flash_mx.cuh"

namespace nanochat {

namespace {

using namespace flash;

static_assert(kD == kernels::kMxFlashHeadDim);
constexpr int kBlockM = kernels::kMxFlashBlockM;
constexpr int kBlockN = kTTokens; // a Vᵀ tile
constexpr int kWarps = 4;         // 32 query rows each
constexpr int kThreads = kWarps * 32;
constexpr int kMTiles = 2;           // m16 tiles per warp
constexpr int kNTiles = kBlockN / 8; // n8 tiles of S
constexpr int kDTiles = kD / 8;      // n8 tiles of O
constexpr int kKSteps = kD / 32;     // k32 steps of Q·Kᵀ, one MX block each
constexpr int kPSteps = kBlockN / 32;
constexpr float kRescaleThreshold = 0.807355f; // log2(448 / 256)
constexpr float kNegBig = -1e30f;              // running max before any key (finite: exp2(m - m_new) stays 0, not nan)

// shared memory (bytes): Q, K, Vᵀ as swizzled 16-byte chunks, then K and V scales. O's bf16 staging reuses Q, K, Vᵀ.
constexpr int kSmemK = kBlockM * kD;
constexpr int kSmemV = kSmemK + kBlockN * kD;
constexpr int kSmemKs = kSmemV + kD * kBlockN;
constexpr int kSmemVs = kSmemKs + kBlockN * 4;
constexpr int kSmemBytes = kSmemVs + kBlockN / 32 * kD;
static_assert(kBlockM * kRowBytes <= kSmemKs, "O staging overlaps the scales");

__device__ __forceinline__ void mx_flash_fwd_body(
      const uint8_t* __restrict__ q, const uint32_t* __restrict__ q_scale, const uint8_t* __restrict__ k,
      const uint32_t* __restrict__ k_scale, const uint8_t* __restrict__ vt, const uint8_t* __restrict__ v_scale,
      __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int T, int H, int Hkv, int window, float scale_log2) {
  __shared__ __align__(128) uint8_t smem[kSmemBytes];
  const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
  const int m0 = static_cast<int>(gridDim.z - 1 - blockIdx.z) * kBlockM; // most keys first, over all heads
  const int h = static_cast<int>(blockIdx.x), b = static_cast<int>(blockIdx.y), hk = h / (H / Hkv);
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

  const int64_t q_ld = static_cast<int64_t>(H) * kD, k_ld = static_cast<int64_t>(Hkv) * kD;
  const uint8_t* q_blk = q + ((static_cast<int64_t>(b) * T + m0) * H + h) * kD;
  const uint8_t* k_head = k + (static_cast<int64_t>(b) * T * Hkv + hk) * kD;
  const int64_t bh_kv = static_cast<int64_t>(b) * Hkv + hk;
  const uint8_t* vt_head = vt + bh_kv * kD * T;
  const uint32_t* ks_head = k_scale + bh_kv * T;
  const uint8_t* vs_head = v_scale + bh_kv * (T / 32) * kD;

  load_rows8<kThreads, kD, kBlockM>(sbase, q_blk, q_ld);
  cp_async_commit();
  const auto load_k = [&](int n) {
    load_rows8<kThreads, kD, kBlockN>(sbase + kSmemK, k_head + static_cast<int64_t>(n) * kBlockN * k_ld, k_ld);
    load_bytes<kThreads, kBlockN * 4>(sbase + kSmemKs, ks_head + static_cast<int64_t>(n) * kBlockN);
    cp_async_commit();
  };
  const auto load_v = [&](int n) {
    load_rows8<kThreads, kBlockN, kD>(sbase + kSmemV, vt_head + static_cast<int64_t>(n) * kBlockN, T);
    load_bytes<kThreads, kPSteps * kD>(sbase + kSmemVs, vs_head + static_cast<int64_t>(n) * kPSteps * kD);
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
        ldmatrix_x4(a[mt], sbase + swz8<kD>(warp * 32 + mt * 16 + lane % 16, 2 * ks + lane / 16));
#pragma unroll
      for (int np = 0; np < kNTiles / 2; ++np) {
        uint32_t bk[4];
        ldmatrix_x4(bk, sbase + kSmemK + swz8<kD>(np * 16 + lane % 8 + (lane / 16) * 8, 2 * ks + (lane / 8) % 2));
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
    for (int mt = 0; mt < kMTiles; ++mt)
#pragma unroll
      for (int kk = 0; kk < kPSteps; ++kk)
        pack_a8(pa[mt][kk], s[mt][4 * kk], s[mt][4 * kk + 1], s[mt][4 * kk + 2], s[mt][4 * kk + 3], 1.0f, 1.0f);

    cp_async_wait<1>(); // V_n
    __syncthreads();
    // O += P V
    const uint8_t* vs_s = smem + kSmemVs;
#pragma unroll
    for (int kk = 0; kk < kPSteps; ++kk) {
      uint32_t sfv[4];
      load_t_scales(sfv, vs_s + kk * kD);
#pragma unroll
      for (int dp = 0; dp < kDTiles / 2; ++dp) {
        uint32_t bv[4];
        ldmatrix_x4(bv, sbase + kSmemV + swz8<kBlockN>(dp * 16 + lane % 8 + (lane / 16) * 8, 2 * kk + (lane / 8) % 2));
        const uint16_t b0 = dp % 2 * 2, b1 = b0 + 1;
#pragma unroll
        for (int mt = 0; mt < kMTiles; ++mt) {
          mma_mx(o[mt][2 * dp], pa[mt][kk], bv[0], bv[1], kPScale, sfv[dp / 2], 0, b0);
          mma_mx(o[mt][2 * dp + 1], pa[mt][kk], bv[2], bv[3], kPScale, sfv[dp / 2], 0, b1);
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
        *reinterpret_cast<uint32_t*>(smem + swz(row, dt) + (lane % 4) * 4) = pack_bf16(
              o[mt][dt][2 * r] * inv[mt][r], o[mt][dt][2 * r + 1] * inv[mt][r]);
      }
    }
  }
  __syncthreads();
  store_rows<kThreads>(smem, out + ((static_cast<int64_t>(b) * T + m0) * H + h) * kD, q_ld, kBlockM);
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

namespace nanochat::kernels {

void mx_flash_fwd(
      const void* q, const uint32_t* q_scale, const void* k, const uint32_t* k_scale, const void* vt,
      const uint8_t* v_scale, void* out, float* lse, int B, int64_t T, int H, int Hkv, int64_t window,
      cudaStream_t stream) {
  const dim3 grid(H, B, static_cast<unsigned>(T / kBlockM)); // tiles slowest: blocks start in order of work
  const float scale_log2 = 1.4426950408889634f / sqrtf(static_cast<float>(kD));
  nanochat_mx_flash_fwd<<<grid, kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(q), q_scale, static_cast<const uint8_t*>(k), k_scale,
        static_cast<const uint8_t*>(vt), v_scale, static_cast<__nv_bfloat16*>(out), lse, static_cast<int>(T), H, Hkv,
        window >= 0 && window < T ? static_cast<int>(window) : -1, scale_log2);
}

} // namespace nanochat::kernels
