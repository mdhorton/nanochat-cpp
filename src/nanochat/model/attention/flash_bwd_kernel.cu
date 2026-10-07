// bf16 flash attention backward on sm_120, as two kernels and without atomics:
//   dq: one CTA per 64 query rows (4 warps x 16), key tiles double-buffered. Computes delta = rowsum(dout * out) for
//       its rows, recomputes P from the forward's log-sum-exp, dP = dout Vᵀ, dS = P (dP - delta), dq = scale dS K.
//   dkv: one CTA per (warps x 16) keys of a kv head, query tiles (of every query head sharing it) double-buffered.
//       Recomputes Pᵀ and dPᵀ with K and V held as A fragments; dv = Pᵀ dout, dk = scale dSᵀ Q.
// S and dP are computed twice (7 matmuls instead of FA2's 5), in exchange for no dq atomics, no fp32 dq buffer and
// no separate delta and dq-conversion passes.
#include "nanochat/model/attention/flash_kernel.h"

#include <type_traits>

#include <cuda_bf16.h>

#include "nanochat/model/attention/flash.cuh"

namespace nanochat {

namespace {

using namespace flash;

constexpr int kKSteps = kD / 16; // k16 steps over head_dim
constexpr int kDTiles = kD / 8;  // n8 tiles over head_dim

// rows x 128 bf16 from global (ld elements apart) into swizzled shared memory
template <int Threads>
__device__ __forceinline__ void load_rows(uint32_t dst, const __nv_bfloat16* src, int64_t ld, int rows) {
  for (int idx = static_cast<int>(threadIdx.x); idx < rows * 16; idx += Threads) {
    const int r = idx / 16, c = idx % 16;
    cp_async16(dst + swz(r, c), reinterpret_cast<const uint8_t*>(src + r * ld) + c * 16);
  }
}

// A fragments of an m16 x 128 tile in swizzled shared memory at rows row0..
__device__ __forceinline__ void load_a(uint32_t (&a)[kKSteps][4], uint32_t base, int row0) {
  const int lane = static_cast<int>(threadIdx.x % 32);
#pragma unroll
  for (int ks = 0; ks < kKSteps; ++ks)
    ldmatrix_x4(a[ks], base + swz(row0 + lane % 16, 2 * ks + lane / 16));
}

// accumulator tiles (m16 x 16 kk .. + 16) as a bf16 A fragment: rows g, g + 8, columns {2t, 2t+1} and + 8
__device__ __forceinline__ void pack_a(uint32_t (&a)[4], const float (&c0)[4], const float (&c1)[4]) {
  a[0] = pack_bf16(c0[0], c0[1]);
  a[1] = pack_bf16(c0[2], c0[3]);
  a[2] = pack_bf16(c1[0], c1[1]);
  a[3] = pack_bf16(c1[2], c1[3]);
}

template <int N>
struct DqConfig {
  static constexpr int kThreads = 128, kBlockM = 64; // 16 query rows per warp
  static constexpr int kBlockN = N;
  static constexpr int kStage = 2 * N * kRowBytes; // K, V
  static constexpr int kSmemBytes = 2 * kStage > 2 * kBlockM* kRowBytes ? 2 * kStage : 2 * kBlockM* kRowBytes;
};

template <class C>
__device__ __forceinline__ void flash_bwd_dq_body(
      const __nv_bfloat16* __restrict__ dout, const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
      const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ out, const float* __restrict__ lse,
      float* __restrict__ delta, __nv_bfloat16* __restrict__ dq, int T, int H, int Hkv, int64_t q_ld, int64_t k_ld,
      int64_t v_ld, int window, float scale_log2, float scale) {
  constexpr int kBlockM = C::kBlockM, kBlockN = C::kBlockN, kNT = kBlockN / 8, kPS = kBlockN / 16;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
  const int m0 = static_cast<int>(gridDim.x - 1 - blockIdx.x) * kBlockM; // most keys first
  const int h = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z), hk = h / (H / Hkv);
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int64_t o_ld = static_cast<int64_t>(H) * kD, bh = static_cast<int64_t>(b) * H + h;
  const auto* q_blk = q + (static_cast<int64_t>(b) * T + m0) * q_ld + h * kD;
  const auto* do_blk = dout + (static_cast<int64_t>(b) * T + m0) * o_ld + h * kD;
  const auto* o_blk = out + (static_cast<int64_t>(b) * T + m0) * o_ld + h * kD;
  const auto* k_head = k + static_cast<int64_t>(b) * T * k_ld + hk * kD;
  const auto* v_head = v + static_cast<int64_t>(b) * T * v_ld + hk * kD;

  // Q and dout staged through the key buffers, into A fragments
  load_rows<C::kThreads>(sbase, q_blk, q_ld, kBlockM);
  load_rows<C::kThreads>(sbase + kBlockM * kRowBytes, do_blk, o_ld, kBlockM);
  cp_async_commit();

  // delta = rowsum(dout * out) of the warp's 16 rows; this thread keeps rows g and g + 8
  float dlt[2] = {0.0f, 0.0f};
  for (int r = 0; r < 16; ++r) {
    const int row = warp * 16 + r;
    const uint2 a = *reinterpret_cast<const uint2*>(do_blk + row * o_ld + lane * 4);
    const uint2 c = *reinterpret_cast<const uint2*>(o_blk + row * o_ld + lane * 4);
    const auto *pa = reinterpret_cast<const __nv_bfloat162*>(&a), *pc = reinterpret_cast<const __nv_bfloat162*>(&c);
    const float2 a0 = __bfloat1622float2(pa[0]), a1 = __bfloat1622float2(pa[1]);
    const float2 c0 = __bfloat1622float2(pc[0]), c1 = __bfloat1622float2(pc[1]);
    float sum = a0.x * c0.x + a0.y * c0.y + a1.x * c1.x + a1.y * c1.y;
#pragma unroll
    for (int off = 16; off > 0; off /= 2)
      sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (r == lane / 4)
      dlt[0] = sum;
    if (r == lane / 4 + 8)
      dlt[1] = sum;
    if (lane == 0)
      delta[bh * T + m0 + row] = sum;
  }
  const int row0 = m0 + warp * 16 + lane / 4;
  const float lse2[2] = {lse[bh * T + row0] * 1.4426950408889634f, lse[bh * T + row0 + 8] * 1.4426950408889634f};

  cp_async_wait<0>();
  __syncthreads();
  uint32_t rq[kKSteps][4], rdo[kKSteps][4];
  load_a(rq, sbase, warp * 16);
  load_a(rdo, sbase + kBlockM * kRowBytes, warp * 16);
  __syncthreads(); // staging free for the key tiles

  const int n_hi = (m0 + kBlockM - 1) / kBlockN;
  const int n_lo = window >= 0 ? max(0, m0 - window) / kBlockN : 0;
  const auto load_kv = [&](int n) {
    const uint32_t st = sbase + ((n - n_lo) & 1) * C::kStage;
    load_rows<C::kThreads>(st, k_head + static_cast<int64_t>(n) * kBlockN * k_ld, k_ld, kBlockN);
    load_rows<C::kThreads>(st + kBlockN * kRowBytes, v_head + static_cast<int64_t>(n) * kBlockN * v_ld, v_ld, kBlockN);
    cp_async_commit();
  };
  load_kv(n_lo);

  float acc[kDTiles][4] = {};
  const auto step = [&](int n, auto masked) {
    constexpr bool kMasked = decltype(masked)::value;
    cp_async_wait<0>();
    __syncthreads(); // tile n landed; the other stage is free
    if (n < n_hi)
      load_kv(n + 1);
    const uint32_t sk = sbase + ((n - n_lo) & 1) * C::kStage, sv = sk + kBlockN * kRowBytes;

    // S = Q Kᵀ, dP = dout Vᵀ
    float s[kNT][4] = {}, dp[kNT][4] = {};
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
      for (int np = 0; np < kNT / 2; ++np) {
        const int r = np * 16 + lane % 8 + (lane / 16) * 8, c = 2 * ks + (lane / 8) % 2;
        uint32_t bk[4], bv[4];
        ldmatrix_x4(bk, sk + swz(r, c));
        ldmatrix_x4(bv, sv + swz(r, c));
        mma(s[2 * np], rq[ks], bk[0], bk[1]);
        mma(s[2 * np + 1], rq[ks], bk[2], bk[3]);
        mma(dp[2 * np], rdo[ks], bv[0], bv[1]);
        mma(dp[2 * np + 1], rdo[ks], bv[2], bv[3]);
      }
    }
    // dS = P (dP - delta), P = exp2(S log2e / sqrt(D) - lse log2e)
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float p = fast_exp2(fmaf(s[nt][j], scale_log2, -lse2[j / 2]));
        if constexpr (kMasked) {
          const int row = row0 + 8 * (j / 2), key = n * kBlockN + nt * 8 + 2 * (lane % 4) + j % 2;
          if (key > row || (window >= 0 && row - key > window))
            p = 0.0f;
        }
        s[nt][j] = p * (dp[nt][j] - dlt[j / 2]);
      }
    }
    // dq += dS K
#pragma unroll
    for (int kk = 0; kk < kPS; ++kk) {
      uint32_t a[4];
      pack_a(a, s[2 * kk], s[2 * kk + 1]);
#pragma unroll
      for (int dp2 = 0; dp2 < kDTiles / 2; ++dp2) {
        uint32_t bk[4];
        ldmatrix_x4_trans(bk, sk + swz(kk * 16 + lane % 8 + ((lane / 8) % 2) * 8, 2 * dp2 + lane / 16));
        mma(acc[2 * dp2], a, bk[0], bk[1]);
        mma(acc[2 * dp2 + 1], a, bk[2], bk[3]);
      }
    }
  };
  for (int n = n_lo; n <= n_hi; ++n)
    if (n * kBlockN + kBlockN - 1 > m0 || (window >= 0 && n * kBlockN < m0 + kBlockM - 1 - window))
      step(n, std::true_type{});
    else
      step(n, std::false_type{});

  __syncthreads();
  stage_acc(smem, warp * 16, acc, scale);
  __syncthreads();
  store_rows<C::kThreads>(smem, dq + (static_cast<int64_t>(b) * T + m0) * o_ld + h * kD, o_ld, kBlockM);
}

template <int Warps, int M>
struct DkvConfig {
  static constexpr int kThreads = Warps * 32;
  static constexpr int kBlockN = Warps * 16; // keys: 16 per warp
  static constexpr int kBlockM = M;          // query tile
  static constexpr int kStageQ = 0, kStageDo = M * kRowBytes, kStageLse = 2 * M * kRowBytes,
                       kStageDelta = kStageLse + M * 4;
  static constexpr int kStage = kStageDelta + M * 4;
  static constexpr int kStaging = 2 * kBlockN * kRowBytes; // K, V; then dk, dv
  static constexpr int kSmemBytes = 2 * kStage > kStaging ? 2 * kStage : kStaging;
};

template <class C>
__device__ __forceinline__ void flash_bwd_dkv_body(
      const __nv_bfloat16* __restrict__ dout, const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
      const __nv_bfloat16* __restrict__ v, const float* __restrict__ lse, const float* __restrict__ delta,
      __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv, int T, int H, int Hkv, int64_t q_ld, int64_t k_ld,
      int64_t v_ld, int window, float scale_log2, float scale) {
  constexpr int kBlockM = C::kBlockM, kBlockN = C::kBlockN, kMT = kBlockM / 8, kQS = kBlockM / 16;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN; // early keys (most queries) first
  const int hk = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z), group = H / Hkv;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int64_t o_ld = static_cast<int64_t>(H) * kD;
  const auto* k_blk = k + (static_cast<int64_t>(b) * T + n0) * k_ld + hk * kD;
  const auto* v_blk = v + (static_cast<int64_t>(b) * T + n0) * v_ld + hk * kD;

  // K and V staged, into A fragments (the warp's 16 keys)
  load_rows<C::kThreads>(sbase, k_blk, k_ld, kBlockN);
  load_rows<C::kThreads>(sbase + kBlockN * kRowBytes, v_blk, v_ld, kBlockN);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  uint32_t rk[kKSteps][4], rv[kKSteps][4];
  load_a(rk, sbase, warp * 16);
  load_a(rv, sbase + kBlockN * kRowBytes, warp * 16);
  __syncthreads(); // staging free for the query tiles

  // query tiles of every head in the group: m_lo..m_hi per head
  const int m_lo = n0 / kBlockM;
  const int q_last = window >= 0 ? min(T - 1, n0 + kBlockN - 1 + window) : T - 1;
  const int m_hi = q_last / kBlockM, per_head = m_hi - m_lo + 1, tiles = per_head * group;
  const auto load_tile = [&](int i) {
    const int hq = hk * group + i / per_head, m = m_lo + i % per_head;
    const uint32_t st = sbase + (i & 1) * C::kStage;
    const int64_t row = static_cast<int64_t>(b) * T + static_cast<int64_t>(m) * kBlockM;
    load_rows<C::kThreads>(st + C::kStageQ, q + row * q_ld + hq * kD, q_ld, kBlockM);
    load_rows<C::kThreads>(st + C::kStageDo, dout + row * o_ld + hq * kD, o_ld, kBlockM);
    const int64_t at = (static_cast<int64_t>(b) * H + hq) * T + static_cast<int64_t>(m) * kBlockM;
    if (tid < kBlockM / 4)
      cp_async16(st + C::kStageLse + tid * 16, lse + at + tid * 4);
    else if (tid < kBlockM / 2)
      cp_async16(st + C::kStageDelta + (tid - kBlockM / 4) * 16, delta + at + (tid - kBlockM / 4) * 4);
    cp_async_commit();
  };
  load_tile(0);

  float acc_k[kDTiles][4] = {}, acc_v[kDTiles][4] = {};
  const int key0 = n0 + warp * 16 + lane / 4; // key of this thread's accumulator rows (and + 8)
  const auto step = [&](int i, auto masked) {
    constexpr bool kMasked = decltype(masked)::value;
    cp_async_wait<0>();
    __syncthreads(); // tile i landed; the other stage is free
    if (i + 1 < tiles)
      load_tile(i + 1);
    const uint32_t st = sbase + (i & 1) * C::kStage;
    const uint32_t sq = st + C::kStageQ, sdo = st + C::kStageDo;
    const auto* s_lse = reinterpret_cast<const float*>(smem + (i & 1) * C::kStage + C::kStageLse);
    const auto* s_delta = reinterpret_cast<const float*>(smem + (i & 1) * C::kStage + C::kStageDelta);
    const int m = m_lo + i % per_head;

    // Sᵀ = K Qᵀ, dPᵀ = V doutᵀ (rows: keys, columns: queries)
    float s[kMT][4] = {}, dp[kMT][4] = {};
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
      for (int mp = 0; mp < kMT / 2; ++mp) {
        const int r = mp * 16 + lane % 8 + (lane / 16) * 8, c = 2 * ks + (lane / 8) % 2;
        uint32_t bq[4], bd[4];
        ldmatrix_x4(bq, sq + swz(r, c));
        ldmatrix_x4(bd, sdo + swz(r, c));
        mma(s[2 * mp], rk[ks], bq[0], bq[1]);
        mma(s[2 * mp + 1], rk[ks], bq[2], bq[3]);
        mma(dp[2 * mp], rv[ks], bd[0], bd[1]);
        mma(dp[2 * mp + 1], rv[ks], bd[2], bd[3]);
      }
    }
    // Pᵀ and dSᵀ = Pᵀ (dPᵀ - delta)
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
      const int col = mt * 8 + 2 * (lane % 4);
      const float2 l2 = *reinterpret_cast<const float2*>(s_lse + col);
      const float2 d2 = *reinterpret_cast<const float2*>(s_delta + col);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float p = fast_exp2(fmaf(s[mt][j], scale_log2, -(j % 2 ? l2.y : l2.x) * 1.4426950408889634f));
        if constexpr (kMasked) {
          const int key = key0 + 8 * (j / 2), query = m * kBlockM + col + j % 2;
          if (key > query || (window >= 0 && query - key > window))
            p = 0.0f;
        }
        s[mt][j] = p;
        dp[mt][j] = p * (dp[mt][j] - (j % 2 ? d2.y : d2.x));
      }
    }
    // dv += Pᵀ dout, dk += dSᵀ Q
#pragma unroll
    for (int kk = 0; kk < kQS; ++kk) {
      uint32_t ap[4], ads[4];
      pack_a(ap, s[2 * kk], s[2 * kk + 1]);
      pack_a(ads, dp[2 * kk], dp[2 * kk + 1]);
#pragma unroll
      for (int dp2 = 0; dp2 < kDTiles / 2; ++dp2) {
        const int r = kk * 16 + lane % 8 + ((lane / 8) % 2) * 8, c = 2 * dp2 + lane / 16;
        uint32_t bd[4], bq[4];
        ldmatrix_x4_trans(bd, sdo + swz(r, c));
        ldmatrix_x4_trans(bq, sq + swz(r, c));
        mma(acc_v[2 * dp2], ap, bd[0], bd[1]);
        mma(acc_v[2 * dp2 + 1], ap, bd[2], bd[3]);
        mma(acc_k[2 * dp2], ads, bq[0], bq[1]);
        mma(acc_k[2 * dp2 + 1], ads, bq[2], bq[3]);
      }
    }
  };
  for (int i = 0; i < tiles; ++i) {
    const int m = m_lo + i % per_head;
    if (m * kBlockM < n0 + kBlockN - 1 || (window >= 0 && m * kBlockM + kBlockM - 1 - n0 > window))
      step(i, std::true_type{});
    else
      step(i, std::false_type{});
  }

  __syncthreads();
  stage_acc(smem, warp * 16, acc_k, scale);
  stage_acc(smem + kBlockN * kRowBytes, warp * 16, acc_v, 1.0f);
  __syncthreads();
  const int64_t kv_ld = static_cast<int64_t>(Hkv) * kD, row = static_cast<int64_t>(b) * T + n0;
  store_rows<C::kThreads>(smem, dk + row * kv_ld + hk * kD, kv_ld, kBlockN);
  store_rows<C::kThreads>(smem + kBlockN * kRowBytes, dv + row * kv_ld + hk * kD, kv_ld, kBlockN);
}

} // namespace

} // namespace nanochat

// Kernels: global nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
#define NANOCHAT_FLASH_BWD_DQ(N)                                                                                       \
  __global__ void __launch_bounds__(128) nanochat_flash_bwd_dq_n##N(                                                   \
        const __nv_bfloat16* __restrict__ dout, const __nv_bfloat16* __restrict__ q,                                   \
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,                                      \
        const __nv_bfloat16* __restrict__ out, const float* __restrict__ lse, float* __restrict__ delta,               \
        __nv_bfloat16* __restrict__ dq, int T, int H, int Hkv, int64_t q_ld, int64_t k_ld, int64_t v_ld, int window,   \
        float scale_log2, float scale) {                                                                               \
    nanochat::flash_bwd_dq_body<nanochat::DqConfig<N>>(                                                                \
          dout, q, k, v, out, lse, delta, dq, T, H, Hkv, q_ld, k_ld, v_ld, window, scale_log2, scale);                 \
  }

#define NANOCHAT_FLASH_BWD_DKV(W, M)                                                                                   \
  __global__ void __launch_bounds__(W * 32) nanochat_flash_bwd_dkv_w##W##_m##M(                                        \
        const __nv_bfloat16* __restrict__ dout, const __nv_bfloat16* __restrict__ q,                                   \
        const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, const float* __restrict__ lse,       \
        const float* __restrict__ delta, __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv, int T, int H, \
        int Hkv, int64_t q_ld, int64_t k_ld, int64_t v_ld, int window, float scale_log2, float scale) {                \
    nanochat::flash_bwd_dkv_body<nanochat::DkvConfig<W, M>>(                                                           \
          dout, q, k, v, lse, delta, dk, dv, T, H, Hkv, q_ld, k_ld, v_ld, window, scale_log2, scale);                  \
  }

NANOCHAT_FLASH_BWD_DQ(32)
NANOCHAT_FLASH_BWD_DQ(64)
NANOCHAT_FLASH_BWD_DKV(4, 32)
NANOCHAT_FLASH_BWD_DKV(4, 64)
NANOCHAT_FLASH_BWD_DKV(8, 32)
NANOCHAT_FLASH_BWD_DKV(8, 64)

namespace nanochat::kernels {

namespace {

using bf16 = __nv_bfloat16;
using DqKernel = void (*)(
      const bf16*, const bf16*, const bf16*, const bf16*, const bf16*, const float*, float*, bf16*, int, int, int,
      int64_t, int64_t, int64_t, int, float, float);
using DkvKernel = void (*)(
      const bf16*, const bf16*, const bf16*, const bf16*, const float*, const float*, bf16*, bf16*, int, int, int,
      int64_t, int64_t, int64_t, int, float, float);

struct DqVariant {
  const char* name;
  DqKernel kernel;
  int block_m, threads, smem;
};

struct DkvVariant {
  const char* name;
  DkvKernel kernel;
  int block_n, threads, smem;
};

template <int N>
constexpr DqVariant dq_variant(const char* name, DqKernel kernel) {
  using C = DqConfig<N>;
  return {name, kernel, C::kBlockM, C::kThreads, C::kSmemBytes};
}

template <int W, int M>
constexpr DkvVariant dkv_variant(const char* name, DkvKernel kernel) {
  using C = DkvConfig<W, M>;
  return {name, kernel, C::kBlockN, C::kThreads, C::kSmemBytes};
}

const DqVariant kDq[kFlashBwdDqVariants] = {
      dq_variant<32>("dq 64q, n32", nanochat_flash_bwd_dq_n32),
      dq_variant<64>("dq 64q, n64", nanochat_flash_bwd_dq_n64),
};

const DkvVariant kDkv[kFlashBwdDkvVariants] = {
      dkv_variant<4, 32>("dkv 64k, m32", nanochat_flash_bwd_dkv_w4_m32),
      dkv_variant<4, 64>("dkv 64k, m64", nanochat_flash_bwd_dkv_w4_m64),
      dkv_variant<8, 32>("dkv 128k, m32", nanochat_flash_bwd_dkv_w8_m32),
      dkv_variant<8, 64>("dkv 128k, m64", nanochat_flash_bwd_dkv_w8_m64),
};

constexpr int kDefaultDq = 0, kDefaultDkv = 0;

template <class Kernel>
void set_smem(Kernel kernel, int smem, bool& done) {
  if (!done)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  done = true;
}

} // namespace

const char* flash_bwd_dq_variant_name(int variant) {
  return kDq[variant < 0 ? kDefaultDq : variant].name;
}

const char* flash_bwd_dkv_variant_name(int variant) {
  return kDkv[variant < 0 ? kDefaultDkv : variant].name;
}

void flash_bwd(
      const void* dout, const void* q, const void* k, const void* v, const void* out, const float* lse, float* delta,
      void* dq, void* dk, void* dv, int B, int64_t T, int H, int Hkv, int64_t q_ld, int64_t k_ld, int64_t v_ld,
      int64_t window, cudaStream_t stream, int dq_variant, int dkv_variant) {
  const int di = dq_variant < 0 ? kDefaultDq : dq_variant, ki = dkv_variant < 0 ? kDefaultDkv : dkv_variant;
  const DqVariant& a = kDq[di];
  const DkvVariant& c = kDkv[ki];
  static bool dq_set[kFlashBwdDqVariants] = {}, dkv_set[kFlashBwdDkvVariants] = {};
  set_smem(a.kernel, a.smem, dq_set[di]);
  set_smem(c.kernel, c.smem, dkv_set[ki]);
  const float scale = 1.0f / sqrtf(static_cast<float>(kD)), scale_log2 = scale * 1.4426950408889634f;
  const int w = window >= 0 && window < T ? static_cast<int>(window) : -1;
  const auto* pdo = static_cast<const bf16*>(dout);
  const auto *pq = static_cast<const bf16*>(q), *pk = static_cast<const bf16*>(k), *pv = static_cast<const bf16*>(v);
  a.kernel<<<dim3(static_cast<unsigned>(T / a.block_m), H, B), a.threads, a.smem, stream>>>(
        pdo, pq, pk, pv, static_cast<const bf16*>(out), lse, delta, static_cast<bf16*>(dq), static_cast<int>(T), H, Hkv,
        q_ld, k_ld, v_ld, w, scale_log2, scale);
  c.kernel<<<dim3(static_cast<unsigned>(T / c.block_n), Hkv, B), c.threads, c.smem, stream>>>(
        pdo, pq, pk, pv, lse, delta, static_cast<bf16*>(dk), static_cast<bf16*>(dv), static_cast<int>(T), H, Hkv, q_ld,
        k_ld, v_ld, w, scale_log2, scale);
}

} // namespace nanochat::kernels
