// MXFP8 flash attention backward on sm_120a: flash_bwd_kernel.cu's two kernels (dq; dk, dv) with every matmul a
// block-scaled e4m3 mma.sync m16n8k32 (ue8m0 per 32 along k). Numerics after LP-FA4 (external/ads_model_kernel_library,
// lp_fa4 flash_bwd_sm100.py): P stored as 256 p with a constant 2^-8 scale, dS with a dynamic scale per row per 32,
// Q, K, dout quantized along head_dim for S and dP and along tokens for dv, dk, dq. The token-quantized copies are
// transposed and permuted (flash_mx_quantize_t) so P and dS go from the accumulator to the A fragment unshuffled.
#include "nanochat/model/flash_kernel.h"

#include <type_traits>

#include <cuda_bf16.h>

#include "nanochat/model/flash_mx.cuh"

namespace nanochat {

namespace {

using namespace flash;

constexpr int kKSteps8 = kD / 32; // k32 steps over head_dim, one MX block each
constexpr int kDTiles = kD / 8;   // n8 tiles over head_dim
constexpr int kTTokens = 64;      // tokens per flash_mx_quantize_t block

template <int N>
struct DqMxConfig {
  static constexpr int kThreads = 128, kBlockM = 64; // 16 query rows per warp
  static constexpr int kBlockN = N;
  // stage: K, V rows; Kᵀ; K, V row scales; Kᵀ scales
  static constexpr int kK = 0, kV = N * kD, kKt = 2 * N * kD, kKs = 3 * N * kD, kVs = kKs + N * 4, kKts = kVs + N * 4;
  static constexpr int kStage = kKts + N / 32 * kD;
  static constexpr int kSmemBytes = 2 * kStage > kBlockM* kRowBytes ? 2 * kStage : kBlockM* kRowBytes; // dq staging
};

template <class C>
__device__ __forceinline__ void flash_bwd_mx_dq_body(
      const kernels::FlashBwdMxInputs& in, __nv_bfloat16* __restrict__ dq, int T, int H, int Hkv, int window,
      float scale_log2, float scale) {
  constexpr int kBlockM = C::kBlockM, kBlockN = C::kBlockN, kNT = kBlockN / 8;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
  const int m0 = static_cast<int>(gridDim.x - 1 - blockIdx.x) * kBlockM; // most keys first
  const int h = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z), hk = h / (H / Hkv);
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int64_t bh = static_cast<int64_t>(b) * H + h, bhk = static_cast<int64_t>(b) * Hkv + hk;
  const int64_t q_ld = static_cast<int64_t>(H) * kD, kv_ld = static_cast<int64_t>(Hkv) * kD; // bytes (e4m3)
  const auto* q = static_cast<const uint8_t*>(in.q);
  const auto* dout = static_cast<const uint8_t*>(in.dout);
  const auto* k_head = static_cast<const uint8_t*>(in.k) + static_cast<int64_t>(b) * T * kv_ld + hk * kD;
  const auto* v_head = static_cast<const uint8_t*>(in.v) + static_cast<int64_t>(b) * T * kv_ld + hk * kD;
  const auto* kt_head = static_cast<const uint8_t*>(in.kt) + bhk * kD * T;
  const uint32_t *ks_head = in.k_scale + bhk * T, *vs_head = in.v_scale + bhk * T;
  const uint8_t* kts_head = in.kt_scale + bhk * (T / 32) * kD;

  // Q and dout rows staged through the key buffers, into A fragments
  const int64_t row_off = (static_cast<int64_t>(b) * T + m0) * q_ld + h * kD;
  load_rows8<C::kThreads, kD>(sbase, q + row_off, q_ld, kBlockM);
  load_rows8<C::kThreads, kD>(sbase + kBlockM * kD, dout + row_off, q_ld, kBlockM);
  cp_async_commit();
  const int row0 = m0 + warp * 16 + lane / 4; // this thread's accumulator rows: row0, row0 + 8
  const uint32_t sfa_q = in.q_scale[bh * T + row0 + 8 * (lane % 2)];
  const uint32_t sfa_do = in.dout_scale[bh * T + row0 + 8 * (lane % 2)];
  const float dlt[2] = {in.delta[bh * T + row0], in.delta[bh * T + row0 + 8]};
  const float lse2[2] = {in.lse[bh * T + row0] * 1.4426950408889634f, in.lse[bh * T + row0 + 8] * 1.4426950408889634f};
  cp_async_wait<0>();
  __syncthreads();
  uint32_t rq[kKSteps8][4], rdo[kKSteps8][4];
#pragma unroll
  for (int ks = 0; ks < kKSteps8; ++ks) {
    ldmatrix_x4(rq[ks], sbase + swz8<kD>(warp * 16 + lane % 16, 2 * ks + lane / 16));
    ldmatrix_x4(rdo[ks], sbase + kBlockM * kD + swz8<kD>(warp * 16 + lane % 16, 2 * ks + lane / 16));
  }
  __syncthreads(); // staging free for the key tiles

  const int n_hi = (m0 + kBlockM - 1) / kBlockN;
  const int n_lo = window >= 0 ? max(0, m0 - window) / kBlockN : 0;
  const auto load_kv = [&](int n) {
    const uint32_t st = sbase + ((n - n_lo) & 1) * C::kStage;
    const int64_t t0 = static_cast<int64_t>(n) * kBlockN;
    load_rows8<C::kThreads, kD>(st + C::kK, k_head + t0 * kv_ld, kv_ld, kBlockN);
    load_rows8<C::kThreads, kD>(st + C::kV, v_head + t0 * kv_ld, kv_ld, kBlockN);
    load_rows8<C::kThreads, kBlockN>(st + C::kKt, kt_head + t0, T, kD);
    load_bytes<C::kThreads>(st + C::kKs, ks_head + t0, kBlockN * 4);
    load_bytes<C::kThreads>(st + C::kVs, vs_head + t0, kBlockN * 4);
    load_bytes<C::kThreads>(st + C::kKts, kts_head + t0 / 32 * kD, kBlockN / 32 * kD);
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
    const int so = ((n - n_lo) & 1) * C::kStage;
    const uint32_t st = sbase + so;
    const auto* ks_s = reinterpret_cast<const uint32_t*>(smem + so + C::kKs);
    const auto* vs_s = reinterpret_cast<const uint32_t*>(smem + so + C::kVs);
    const uint8_t* kts_s = smem + so + C::kKts;

    // S = Q Kᵀ, dP = dout Vᵀ
    float s[kNT][4] = {}, dp[kNT][4] = {};
    uint32_t sfk[kNT], sfv[kNT];
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt)
      sfk[nt] = ks_s[nt * 8 + lane / 4], sfv[nt] = vs_s[nt * 8 + lane / 4];
#pragma unroll
    for (int ks = 0; ks < kKSteps8; ++ks) {
#pragma unroll
      for (int np = 0; np < kNT / 2; ++np) {
        const int r = np * 16 + lane % 8 + (lane / 16) * 8, c = 2 * ks + (lane / 8) % 2;
        uint32_t bk[4], bv[4];
        ldmatrix_x4(bk, st + C::kK + swz8<kD>(r, c));
        ldmatrix_x4(bv, st + C::kV + swz8<kD>(r, c));
        mma_mx(s[2 * np], rq[ks], bk[0], bk[1], sfa_q, sfk[2 * np], ks, ks);
        mma_mx(s[2 * np + 1], rq[ks], bk[2], bk[3], sfa_q, sfk[2 * np + 1], ks, ks);
        mma_mx(dp[2 * np], rdo[ks], bv[0], bv[1], sfa_do, sfv[2 * np], ks, ks);
        mma_mx(dp[2 * np + 1], rdo[ks], bv[2], bv[3], sfa_do, sfv[2 * np + 1], ks, ks);
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
    // dq += dS K (Kᵀ's tokens permuted as dS's A fragment)
#pragma unroll
    for (int kk = 0; kk < kBlockN / 32; ++kk) {
      uint32_t a[4];
      const uint32_t sfa = pack_a8_scaled(a, s[4 * kk], s[4 * kk + 1], s[4 * kk + 2], s[4 * kk + 3]);
#pragma unroll
      for (int dp2 = 0; dp2 < kDTiles / 2; ++dp2) {
        uint32_t bk[4];
        ldmatrix_x4(bk, st + C::kKt + swz8<kBlockN>(dp2 * 16 + lane % 8 + (lane / 16) * 8, 2 * kk + (lane / 8) % 2));
        const uint32_t sfb0 = kts_s[kk * kD + dp2 * 16 + lane / 4], sfb1 = kts_s[kk * kD + dp2 * 16 + 8 + lane / 4];
        mma_mx(acc[2 * dp2], a, bk[0], bk[1], sfa, sfb0, 0, 0);
        mma_mx(acc[2 * dp2 + 1], a, bk[2], bk[3], sfa, sfb1, 0, 0);
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
  store_rows<C::kThreads>(smem, dq + (static_cast<int64_t>(b) * T + m0) * q_ld + h * kD, q_ld, kBlockM);
}

template <int Warps, int M>
struct DkvMxConfig {
  static constexpr int kThreads = Warps * 32;
  static constexpr int kBlockN = Warps * 16; // keys: 16 per warp
  static constexpr int kBlockM = M;          // query tile
  // stage: Q, dout rows; Qᵀ, doutᵀ; their scales; lse, delta
  static constexpr int kQ = 0, kDo = M * kD, kQt = 2 * M * kD, kDot = 3 * M * kD;
  static constexpr int kQs = 4 * M * kD, kDos = kQs + M * 4, kQts = kDos + M * 4, kDots = kQts + M / 32 * kD;
  static constexpr int kLse = kDots + M / 32 * kD, kDelta = kLse + M * 4;
  static constexpr int kStage = kDelta + M * 4;
  static constexpr int kStaging = 2 * kBlockN * kRowBytes; // dk, dv (bf16)
  static constexpr int kSmemBytes = 2 * kStage > kStaging ? 2 * kStage : kStaging;
};

template <class C>
__device__ __forceinline__ void flash_bwd_mx_dkv_body(
      const kernels::FlashBwdMxInputs& in, __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv, int T, int H,
      int Hkv, int window, float scale_log2, float scale) {
  constexpr int kBlockM = C::kBlockM, kBlockN = C::kBlockN, kMT = kBlockM / 8;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
  const int n0 = static_cast<int>(blockIdx.x) * kBlockN; // early keys (most queries) first
  const int hk = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z), group = H / Hkv;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int64_t q_ld = static_cast<int64_t>(H) * kD, kv_ld = static_cast<int64_t>(Hkv) * kD; // bytes (e4m3)
  const int64_t bhk = static_cast<int64_t>(b) * Hkv + hk;

  // K and V rows staged, into A fragments (the warp's 16 keys)
  const int64_t kv_off = (static_cast<int64_t>(b) * T + n0) * kv_ld + hk * kD;
  load_rows8<C::kThreads, kD>(sbase, static_cast<const uint8_t*>(in.k) + kv_off, kv_ld, kBlockN);
  load_rows8<C::kThreads, kD>(sbase + kBlockN * kD, static_cast<const uint8_t*>(in.v) + kv_off, kv_ld, kBlockN);
  cp_async_commit();
  const int key0 = n0 + warp * 16 + lane / 4; // key of this thread's accumulator rows (and + 8)
  const uint32_t sfa_k = in.k_scale[bhk * T + key0 + 8 * (lane % 2)];
  const uint32_t sfa_v = in.v_scale[bhk * T + key0 + 8 * (lane % 2)];
  cp_async_wait<0>();
  __syncthreads();
  uint32_t rk[kKSteps8][4], rv[kKSteps8][4];
#pragma unroll
  for (int ks = 0; ks < kKSteps8; ++ks) {
    ldmatrix_x4(rk[ks], sbase + swz8<kD>(warp * 16 + lane % 16, 2 * ks + lane / 16));
    ldmatrix_x4(rv[ks], sbase + kBlockN * kD + swz8<kD>(warp * 16 + lane % 16, 2 * ks + lane / 16));
  }
  __syncthreads(); // staging free for the query tiles

  // query tiles of every head in the group: m_lo..m_hi per head
  const int m_lo = n0 / kBlockM;
  const int q_last = window >= 0 ? min(T - 1, n0 + kBlockN - 1 + window) : T - 1;
  const int m_hi = q_last / kBlockM, per_head = m_hi - m_lo + 1, tiles = per_head * group;
  const auto load_tile = [&](int i) {
    const int hq = hk * group + i / per_head;
    const int64_t t0 = static_cast<int64_t>(m_lo + i % per_head) * kBlockM, bh = static_cast<int64_t>(b) * H + hq;
    const uint32_t st = sbase + (i & 1) * C::kStage;
    const int64_t row_off = (static_cast<int64_t>(b) * T + t0) * q_ld + hq * kD;
    load_rows8<C::kThreads, kD>(st + C::kQ, static_cast<const uint8_t*>(in.q) + row_off, q_ld, kBlockM);
    load_rows8<C::kThreads, kD>(st + C::kDo, static_cast<const uint8_t*>(in.dout) + row_off, q_ld, kBlockM);
    load_rows8<C::kThreads, kBlockM>(st + C::kQt, static_cast<const uint8_t*>(in.qt) + bh * kD * T + t0, T, kD);
    load_rows8<C::kThreads, kBlockM>(st + C::kDot, static_cast<const uint8_t*>(in.doutt) + bh * kD * T + t0, T, kD);
    load_bytes<C::kThreads>(st + C::kQs, in.q_scale + bh * T + t0, kBlockM * 4);
    load_bytes<C::kThreads>(st + C::kDos, in.dout_scale + bh * T + t0, kBlockM * 4);
    const int64_t ts = (bh * (T / 32) + t0 / 32) * kD;
    load_bytes<C::kThreads>(st + C::kQts, in.qt_scale + ts, kBlockM / 32 * kD);
    load_bytes<C::kThreads>(st + C::kDots, in.doutt_scale + ts, kBlockM / 32 * kD);
    load_bytes<C::kThreads>(st + C::kLse, in.lse + bh * T + t0, kBlockM * 4);
    load_bytes<C::kThreads>(st + C::kDelta, in.delta + bh * T + t0, kBlockM * 4);
    cp_async_commit();
  };
  load_tile(0);

  float acc_k[kDTiles][4] = {}, acc_v[kDTiles][4] = {};
  const auto step = [&](int i, auto masked) {
    constexpr bool kMasked = decltype(masked)::value;
    cp_async_wait<0>();
    __syncthreads(); // tile i landed; the other stage is free
    if (i + 1 < tiles)
      load_tile(i + 1);
    const int so = (i & 1) * C::kStage;
    const uint32_t st = sbase + so;
    const auto* qs_s = reinterpret_cast<const uint32_t*>(smem + so + C::kQs);
    const auto* dos_s = reinterpret_cast<const uint32_t*>(smem + so + C::kDos);
    const uint8_t *qts_s = smem + so + C::kQts, *dots_s = smem + so + C::kDots;
    const auto* s_lse = reinterpret_cast<const float*>(smem + so + C::kLse);
    const auto* s_delta = reinterpret_cast<const float*>(smem + so + C::kDelta);
    const int m = m_lo + i % per_head;

    // Sᵀ = K Qᵀ, dPᵀ = V doutᵀ (rows: keys, columns: queries)
    float s[kMT][4] = {}, dp[kMT][4] = {};
    uint32_t sfq[kMT], sfd[kMT];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
      sfq[mt] = qs_s[mt * 8 + lane / 4], sfd[mt] = dos_s[mt * 8 + lane / 4];
#pragma unroll
    for (int ks = 0; ks < kKSteps8; ++ks) {
#pragma unroll
      for (int mp = 0; mp < kMT / 2; ++mp) {
        const int r = mp * 16 + lane % 8 + (lane / 16) * 8, c = 2 * ks + (lane / 8) % 2;
        uint32_t bq[4], bd[4];
        ldmatrix_x4(bq, st + C::kQ + swz8<kD>(r, c));
        ldmatrix_x4(bd, st + C::kDo + swz8<kD>(r, c));
        mma_mx(s[2 * mp], rk[ks], bq[0], bq[1], sfa_k, sfq[2 * mp], ks, ks);
        mma_mx(s[2 * mp + 1], rk[ks], bq[2], bq[3], sfa_k, sfq[2 * mp + 1], ks, ks);
        mma_mx(dp[2 * mp], rv[ks], bd[0], bd[1], sfa_v, sfd[2 * mp], ks, ks);
        mma_mx(dp[2 * mp + 1], rv[ks], bd[2], bd[3], sfa_v, sfd[2 * mp + 1], ks, ks);
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
    // dv += Pᵀ dout, dk += dSᵀ Q (doutᵀ's and Qᵀ's tokens permuted as the A fragments)
#pragma unroll
    for (int kk = 0; kk < kBlockM / 32; ++kk) {
      uint32_t ap[4], ads[4];
      pack_a8(ap, s[4 * kk], s[4 * kk + 1], s[4 * kk + 2], s[4 * kk + 3], 256.0f, 256.0f);
      const uint32_t sfa_ds = pack_a8_scaled(ads, dp[4 * kk], dp[4 * kk + 1], dp[4 * kk + 2], dp[4 * kk + 3]);
#pragma unroll
      for (int dp2 = 0; dp2 < kDTiles / 2; ++dp2) {
        const int r = dp2 * 16 + lane % 8 + (lane / 16) * 8, c = 2 * kk + (lane / 8) % 2;
        uint32_t bd[4], bq[4];
        ldmatrix_x4(bd, st + C::kDot + swz8<kBlockM>(r, c));
        ldmatrix_x4(bq, st + C::kQt + swz8<kBlockM>(r, c));
        const int si = kk * kD + dp2 * 16 + lane / 4;
        mma_mx(acc_v[2 * dp2], ap, bd[0], bd[1], kPScale, dots_s[si], 0, 0);
        mma_mx(acc_v[2 * dp2 + 1], ap, bd[2], bd[3], kPScale, dots_s[si + 8], 0, 0);
        mma_mx(acc_k[2 * dp2], ads, bq[0], bq[1], sfa_ds, qts_s[si], 0, 0);
        mma_mx(acc_k[2 * dp2 + 1], ads, bq[2], bq[3], sfa_ds, qts_s[si + 8], 0, 0);
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
  const int64_t out_ld = static_cast<int64_t>(Hkv) * kD, row = static_cast<int64_t>(b) * T + n0;
  store_rows<C::kThreads>(smem, dk + row * out_ld + hk * kD, out_ld, kBlockN);
  store_rows<C::kThreads>(smem + kBlockN * kRowBytes, dv + row * out_ld + hk * kD, out_ld, kBlockN);
}

constexpr int kQuantWarps = 8;

// one warp per (token, head) row: 4 values per lane, 8 lanes per MX block; with out, delta = rowsum(x * out)
__device__ __forceinline__ void flash_mx_quantize_rows_body(
      const __nv_bfloat16* __restrict__ x, int64_t x_ld, uint32_t* __restrict__ data, uint32_t* __restrict__ scale,
      const __nv_bfloat16* __restrict__ out, float* __restrict__ delta, int64_t rows, int64_t T, int heads) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kQuantWarps + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int lane = static_cast<int>(threadIdx.x % 32);
  const int64_t bt = row / heads;
  const int h = static_cast<int>(row % heads);
  const int64_t at = (bt / T * heads + h) * T + bt % T;
  const uint2 raw = *reinterpret_cast<const uint2*>(x + bt * x_ld + static_cast<int64_t>(h) * kD + lane * 4);
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
    scale[at] = word;
  if (out != nullptr) {
    const uint2 raw_o = *reinterpret_cast<const uint2*>(out + row * kD + lane * 4);
    const auto* po = reinterpret_cast<const __nv_bfloat162*>(&raw_o);
    const float2 o01 = __bfloat1622float2(po[0]), o23 = __bfloat1622float2(po[1]);
    float sum = v01.x * o01.x + v01.y * o01.y + v23.x * o23.x + v23.y * o23.y;
#pragma unroll
    for (int off = 16; off > 0; off /= 2)
      sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (lane == 0)
      delta[at] = sum;
  }
}

constexpr int kTThreads = 256;  // (dim, 32-token half) per thread
constexpr int kTPitch = kD + 8; // bf16 per tile row: 16-byte rows, conflict-free column reads

// one block per (64 tokens, head, batch); layouts as kernels::flash_mx_quantize_t
__device__ __forceinline__ void flash_mx_quantize_t_body(
      const __nv_bfloat16* __restrict__ x, int64_t x_ld, uint8_t* __restrict__ data, uint8_t* __restrict__ scale,
      int64_t T, int heads) {
  __shared__ __align__(16) __nv_bfloat16 tile[kTTokens][kTPitch];
  const int n = static_cast<int>(blockIdx.x), h = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int i = 0; i < kTTokens * kD / 8 / kTThreads; ++i) {
    const int idx = tid + i * kTThreads, t = idx / (kD / 8), c = idx % (kD / 8);
    const auto* src = x + (static_cast<int64_t>(b) * T + static_cast<int64_t>(n) * kTTokens + t) * x_ld +
                      static_cast<int64_t>(h) * kD + c * 8;
    *reinterpret_cast<uint4*>(&tile[t][c * 8]) = *reinterpret_cast<const uint4*>(src);
  }
  __syncthreads();
  const int d = tid % kD, half = tid / kD;
  float v[32];
  float amax = 0.0f;
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    v[t] = __bfloat162float(tile[half * 32 + t][d]);
    amax = fmaxf(amax, fabsf(v[t]));
  }
  const int e = mx_exponent(amax);
  const float mul = mx_multiplier(e);
  // position p of each 16 holds token [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15][p]
  const auto token = [](int p) {
    const int i = p % 16;
    return p / 16 * 16 + 2 * (i / 4) + i % 2 + 8 * (i % 4 / 2);
  };
  uint32_t w[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    w[j] = pack_e4m3(
          v[token(4 * j)] * mul, v[token(4 * j + 1)] * mul, v[token(4 * j + 2)] * mul, v[token(4 * j + 3)] * mul);
  const int64_t bh = static_cast<int64_t>(b) * heads + h;
  auto* dst = reinterpret_cast<uint4*>(data + (bh * kD + d) * T + static_cast<int64_t>(n) * kTTokens + half * 32);
  dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
  dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
  scale[(bh * (T / 32) + 2 * n + half) * kD + d] = static_cast<uint8_t>(e);
}

} // namespace

} // namespace nanochat

// Kernels: global nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
#define NANOCHAT_FLASH_BWD_MX_DQ(N)                                                                                    \
  __global__ void __launch_bounds__(128) nanochat_flash_bwd_mx_dq_n##N(                                                \
        const nanochat::kernels::FlashBwdMxInputs in, __nv_bfloat16* __restrict__ dq, int T, int H, int Hkv,           \
        int window, float scale_log2, float scale) {                                                                   \
    nanochat::flash_bwd_mx_dq_body<nanochat::DqMxConfig<N>>(in, dq, T, H, Hkv, window, scale_log2, scale);             \
  }

#define NANOCHAT_FLASH_BWD_MX_DKV(W, M)                                                                                \
  __global__ void __launch_bounds__(W * 32) nanochat_flash_bwd_mx_dkv_w##W##_m##M(                                     \
        const nanochat::kernels::FlashBwdMxInputs in, __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dv,  \
        int T, int H, int Hkv, int window, float scale_log2, float scale) {                                            \
    nanochat::flash_bwd_mx_dkv_body<nanochat::DkvMxConfig<W, M>>(in, dk, dv, T, H, Hkv, window, scale_log2, scale);    \
  }

NANOCHAT_FLASH_BWD_MX_DQ(32)
NANOCHAT_FLASH_BWD_MX_DQ(64)
NANOCHAT_FLASH_BWD_MX_DKV(4, 32)
NANOCHAT_FLASH_BWD_MX_DKV(4, 64)
NANOCHAT_FLASH_BWD_MX_DKV(8, 32)
NANOCHAT_FLASH_BWD_MX_DKV(8, 64)

__global__ void __launch_bounds__(nanochat::kQuantWarps * 32) nanochat_flash_mx_quantize_rows(
      const __nv_bfloat16* __restrict__ x, int64_t x_ld, uint32_t* __restrict__ data, uint32_t* __restrict__ scale,
      const __nv_bfloat16* __restrict__ out, float* __restrict__ delta, int64_t rows, int64_t T, int heads) {
  nanochat::flash_mx_quantize_rows_body(x, x_ld, data, scale, out, delta, rows, T, heads);
}

__global__ void __launch_bounds__(nanochat::kTThreads) nanochat_flash_mx_quantize_t(
      const __nv_bfloat16* __restrict__ x, int64_t x_ld, uint8_t* __restrict__ data, uint8_t* __restrict__ scale,
      int64_t T, int heads) {
  nanochat::flash_mx_quantize_t_body(x, x_ld, data, scale, T, heads);
}

namespace nanochat::kernels {

namespace {

using bf16 = __nv_bfloat16;
using DqKernel = void (*)(const FlashBwdMxInputs, bf16*, int, int, int, int, float, float);
using DkvKernel = void (*)(const FlashBwdMxInputs, bf16*, bf16*, int, int, int, int, float, float);

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
  using C = DqMxConfig<N>;
  return {name, kernel, C::kBlockM, C::kThreads, C::kSmemBytes};
}

template <int W, int M>
constexpr DkvVariant dkv_variant(const char* name, DkvKernel kernel) {
  using C = DkvMxConfig<W, M>;
  return {name, kernel, C::kBlockN, C::kThreads, C::kSmemBytes};
}

const DqVariant kDq[kFlashBwdMxDqVariants] = {
      dq_variant<32>("mx dq 64q, n32", nanochat_flash_bwd_mx_dq_n32),
      dq_variant<64>("mx dq 64q, n64", nanochat_flash_bwd_mx_dq_n64),
};

const DkvVariant kDkv[kFlashBwdMxDkvVariants] = {
      dkv_variant<4, 32>("mx dkv 64k, m32", nanochat_flash_bwd_mx_dkv_w4_m32),
      dkv_variant<4, 64>("mx dkv 64k, m64", nanochat_flash_bwd_mx_dkv_w4_m64),
      dkv_variant<8, 32>("mx dkv 128k, m32", nanochat_flash_bwd_mx_dkv_w8_m32),
      dkv_variant<8, 64>("mx dkv 128k, m64", nanochat_flash_bwd_mx_dkv_w8_m64),
};

constexpr int kDefaultDq = 0, kDefaultDkv = 0;

template <class Kernel>
void set_smem(Kernel kernel, int smem, bool& done) {
  if (!done)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  done = true;
}

} // namespace

const char* flash_bwd_mx_dq_variant_name(int variant) {
  return kDq[variant < 0 ? kDefaultDq : variant].name;
}

const char* flash_bwd_mx_dkv_variant_name(int variant) {
  return kDkv[variant < 0 ? kDefaultDkv : variant].name;
}

void flash_mx_quantize_rows(
      const void* x, int64_t x_ld, void* data, uint32_t* scale, const void* out, float* delta, int B, int64_t T,
      int heads, cudaStream_t stream) {
  const int64_t rows = static_cast<int64_t>(B) * T * heads;
  nanochat_flash_mx_quantize_rows<<<(rows + kQuantWarps - 1) / kQuantWarps, kQuantWarps * 32, 0, stream>>>(
        static_cast<const bf16*>(x), x_ld, static_cast<uint32_t*>(data), scale, static_cast<const bf16*>(out), delta,
        rows, T, heads);
}

void flash_mx_quantize_t(
      const void* x, int64_t x_ld, void* data, uint8_t* scale, int B, int64_t T, int heads, cudaStream_t stream) {
  nanochat_flash_mx_quantize_t<<<dim3(static_cast<unsigned>(T / kTTokens), heads, B), kTThreads, 0, stream>>>(
        static_cast<const bf16*>(x), x_ld, static_cast<uint8_t*>(data), scale, T, heads);
}

void flash_bwd_mx(
      const FlashBwdMxInputs& in, void* dq, void* dk, void* dv, int B, int64_t T, int H, int Hkv, int64_t window,
      cudaStream_t stream, int dq_variant, int dkv_variant) {
  const int di = dq_variant < 0 ? kDefaultDq : dq_variant, ki = dkv_variant < 0 ? kDefaultDkv : dkv_variant;
  const DqVariant& a = kDq[di];
  const DkvVariant& c = kDkv[ki];
  static bool dq_set[kFlashBwdMxDqVariants] = {}, dkv_set[kFlashBwdMxDkvVariants] = {};
  set_smem(a.kernel, a.smem, dq_set[di]);
  set_smem(c.kernel, c.smem, dkv_set[ki]);
  const float scale = 1.0f / sqrtf(static_cast<float>(kD)), scale_log2 = scale * 1.4426950408889634f;
  const int w = window >= 0 && window < T ? static_cast<int>(window) : -1;
  a.kernel<<<dim3(static_cast<unsigned>(T / a.block_m), H, B), a.threads, a.smem, stream>>>(
        in, static_cast<bf16*>(dq), static_cast<int>(T), H, Hkv, w, scale_log2, scale);
  c.kernel<<<dim3(static_cast<unsigned>(T / c.block_n), Hkv, B), c.threads, c.smem, stream>>>(
        in, static_cast<bf16*>(dk), static_cast<bf16*>(dv), static_cast<int>(T), H, Hkv, w, scale_log2, scale);
}

} // namespace nanochat::kernels
