// bf16 flash attention forward on sm_120. Each warp owns 16 or 32 query rows; key tiles stream through shared memory
// with cp.async (K of the next tile loads during softmax and P·V, V during the next Q·Kᵀ). P's accumulator is the
// next mma's A operand as is; V is read with ldmatrix.trans. Tiles wholly inside the causal window skip the mask, and
// O is rescaled only when a row max grows by more than 2^8 (FA4's lazy rescale).
#include "nanochat/model/flash_kernel.h"

#include <type_traits>

#include <cuda_bf16.h>

namespace nanochat {

namespace {

constexpr int kD = kernels::kFlashHeadDim;
constexpr int kRowBytes = kD * 2;      // a bf16 row of Q, K, V or O: 16 chunks of 16 bytes
constexpr float kRescaleThreshold = 8; // log2: P = exp2(x - m) up to 256
constexpr float kNegBig = -1e30f;      // running max before any key (finite: exp2(m - m_new) stays 0, not nan)

template <int Warps, int MTiles, int BlockN>
struct Config {
  static constexpr int kThreads = Warps * 32;
  static constexpr int kMTiles = MTiles; // m16 tiles per warp
  static constexpr int kBlockM = Warps * 16 * MTiles;
  static constexpr int kBlockN = BlockN;
  static constexpr int kNTiles = BlockN / 8;         // n8 tiles of S
  static constexpr int kDTiles = kD / 8;             // n8 tiles of O
  static constexpr int kKSteps = kD / 16;            // k16 steps of Q·Kᵀ
  static constexpr int kPSteps = BlockN / 16;        // k16 steps of P·V
  static constexpr bool kQInRegs = MTiles == 1;      // Q's fragments loaded once (32 registers)
  static constexpr int kSmemK = kBlockM * kRowBytes; // Q, then K, V (one tile each); O's staging reuses Q
  static constexpr int kSmemV = kSmemK + BlockN * kRowBytes;
  static constexpr int kSmemBytes = kSmemV + BlockN * kRowBytes;
};

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

template <class C>
__device__ __forceinline__ void flash_fwd_body(
      const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
      __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int T, int H, int Hkv, int64_t q_ld, int64_t k_ld,
      int64_t v_ld, int window, float scale_log2) {
  constexpr int kMT = C::kMTiles, kNT = C::kNTiles, kBlockM = C::kBlockM, kBlockN = C::kBlockN;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
  const int m0 = (static_cast<int>(gridDim.x - 1 - blockIdx.x)) * kBlockM; // most keys first
  const int h = static_cast<int>(blockIdx.y), b = static_cast<int>(blockIdx.z), hk = h / (H / Hkv);
  const uint32_t sq = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t sk = sq + C::kSmemK, sv = sq + C::kSmemV;

  const auto* q_blk = reinterpret_cast<const uint8_t*>(q + (static_cast<int64_t>(b) * T + m0) * q_ld + h * kD);
  const auto* k_head = reinterpret_cast<const uint8_t*>(k + static_cast<int64_t>(b) * T * k_ld + hk * kD);
  const auto* v_head = reinterpret_cast<const uint8_t*>(v + static_cast<int64_t>(b) * T * v_ld + hk * kD);
  const auto load_rows = [&](uint32_t dst, const uint8_t* src, int64_t ld, int rows) {
    for (int idx = tid; idx < rows * 16; idx += C::kThreads) {
      const int r = idx / 16, c = idx % 16;
      cp_async16(dst + swz(r, c), src + r * ld * 2 + c * 16);
    }
    cp_async_commit();
  };
  const auto load_k = [&](int n) {
    load_rows(sk, k_head + static_cast<int64_t>(n) * kBlockN * k_ld * 2, k_ld, kBlockN);
  };
  const auto load_v = [&](int n) {
    load_rows(sv, v_head + static_cast<int64_t>(n) * kBlockN * v_ld * 2, v_ld, kBlockN);
  };

  const int n_hi = (m0 + kBlockM - 1) / kBlockN;
  const int n_lo = window >= 0 ? max(0, m0 - window) / kBlockN : 0;
  const int row0 = m0 + warp * 16 * kMT + lane / 4; // row of this thread's accumulator slot 0 in m-tile 0

  load_rows(sq, q_blk, q_ld, kBlockM);
  load_k(n_lo);
  load_v(n_lo);

  // Q's A fragments for m-tile mt, k-step ks
  const auto q_frag = [&](uint32_t (&a)[4], int mt, int ks) {
    ldmatrix_x4(a, sq + swz(warp * 16 * kMT + mt * 16 + lane % 16, 2 * ks + lane / 16));
  };
  uint32_t rq[C::kQInRegs ? C::kKSteps : 1][4];
  if constexpr (C::kQInRegs) {
    cp_async_wait<2>(); // Q
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < C::kKSteps; ++ks)
      q_frag(rq[ks], 0, ks);
  }

  float o[kMT][C::kDTiles][4];
  float m[kMT][2], l[kMT][2];
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int dt = 0; dt < C::kDTiles; ++dt)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        o[mt][dt][j] = 0.0f;
#pragma unroll
    for (int r = 0; r < 2; ++r)
      m[mt][r] = kNegBig, l[mt][r] = 0.0f;
  }

  // one key tile; kMasked: some of its keys are outside the causal window of some rows
  const auto step = [&](int n, auto masked) {
    constexpr bool kMasked = decltype(masked)::value;
    cp_async_wait<1>(); // Q, K_n
    __syncthreads();

    // S = Q Kᵀ
    float s[kMT][kNT][4];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          s[mt][nt][j] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < C::kKSteps; ++ks) {
      uint32_t a[kMT][4];
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt) {
        if constexpr (C::kQInRegs)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            a[mt][j] = rq[ks][j];
        else
          q_frag(a[mt], mt, ks);
      }
#pragma unroll
      for (int np = 0; np < kNT / 2; ++np) {
        uint32_t bk[4];
        ldmatrix_x4(bk, sk + swz(np * 16 + lane % 8 + (lane / 16) * 8, 2 * ks + (lane / 8) % 2));
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
          mma(s[mt][2 * np], a[mt], bk[0], bk[1]);
          mma(s[mt][2 * np + 1], a[mt], bk[2], bk[3]);
        }
      }
    }
    __syncthreads(); // K consumed
    if (n < n_hi)
      load_k(n + 1);
    else
      cp_async_commit(); // empty group: the wait below still means "V_n landed"

    // online softmax in base 2
    float alpha[kMT][2];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int row = row0 + mt * 16 + 8 * r;
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < kNT; ++nt) {
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
        const float m_new = mx * scale_log2 > m[mt][r] + kRescaleThreshold ? mx * scale_log2 : m[mt][r];
        alpha[mt][r] = fast_exp2(m[mt][r] - m_new);
        m[mt][r] = m_new;
        l[mt][r] *= alpha[mt][r];
#pragma unroll
        for (int nt = 0; nt < kNT; ++nt) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float& x = s[mt][nt][2 * r + e];
            x = fast_exp2(fmaf(x, scale_log2, -m_new));
            l[mt][r] += x;
          }
        }
      }
    }
    bool rescale = false;
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
      rescale |= alpha[mt][0] != 1.0f || alpha[mt][1] != 1.0f;
    if (__any_sync(0xffffffffu, rescale)) {
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int dt = 0; dt < C::kDTiles; ++dt)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            o[mt][dt][j] *= alpha[mt][j / 2];
    }

    // P as the A operand, k-step kk: keys 16 kk + {2t, 2t+1} and + 8
    uint32_t pa[kMT][C::kPSteps][4];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
      for (int kk = 0; kk < C::kPSteps; ++kk) {
        const float (&s0)[4] = s[mt][2 * kk], (&s1)[4] = s[mt][2 * kk + 1];
        pa[mt][kk][0] = pack_bf16(s0[0], s0[1]);
        pa[mt][kk][1] = pack_bf16(s0[2], s0[3]);
        pa[mt][kk][2] = pack_bf16(s1[0], s1[1]);
        pa[mt][kk][3] = pack_bf16(s1[2], s1[3]);
      }
    }

    cp_async_wait<1>(); // V_n
    __syncthreads();
    // O += P V
#pragma unroll
    for (int kk = 0; kk < C::kPSteps; ++kk) {
#pragma unroll
      for (int dp = 0; dp < C::kDTiles / 2; ++dp) {
        uint32_t bv[4];
        ldmatrix_x4_trans(bv, sv + swz(kk * 16 + lane % 8 + ((lane / 8) % 2) * 8, 2 * dp + lane / 16));
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
          mma(o[mt][2 * dp], pa[mt][kk], bv[0], bv[1]);
          mma(o[mt][2 * dp + 1], pa[mt][kk], bv[2], bv[3]);
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

  // O = acc / l; lse = ln(sum exp(s / sqrt(D))) = ln2 (m + log2(l))
  float inv[kMT][2];
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float sum = l[mt][r];
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      inv[mt][r] = 1.0f / sum;
      if (lane % 4 == 0)
        lse[(static_cast<int64_t>(b) * H + h) * T + row0 + mt * 16 + 8 * r] = 0.69314718f * (m[mt][r] + __log2f(sum));
    }
  }
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = warp * 16 * kMT + mt * 16 + lane / 4 + 8 * r;
#pragma unroll
      for (int dt = 0; dt < C::kDTiles; ++dt)
        *reinterpret_cast<uint32_t*>(smem + swz(row, dt) + (lane % 4) * 4) = pack_bf16(
              o[mt][dt][2 * r] * inv[mt][r], o[mt][dt][2 * r + 1] * inv[mt][r]);
    }
  }
  __syncthreads();
  auto* out_blk = reinterpret_cast<uint8_t*>(out + ((static_cast<int64_t>(b) * T + m0) * H + h) * kD);
  const int64_t out_ld = static_cast<int64_t>(H) * kRowBytes;
  for (int idx = tid; idx < kBlockM * 16; idx += C::kThreads) {
    const int r = idx / 16, c = idx % 16;
    *reinterpret_cast<uint4*>(out_blk + r * out_ld + c * 16) = *reinterpret_cast<const uint4*>(smem + swz(r, c));
  }
}

} // namespace

} // namespace nanochat

// Kernels: global nanochat_* names read the same in nsys and ncu (see softcap_ce_kernel.cu).
#define NANOCHAT_FLASH_FWD(W, MT, N)                                                                                   \
  __global__ void __launch_bounds__(W * 32) nanochat_flash_fwd_w##W##_m##MT##_n##N(                                    \
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, \
        __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int T, int H, int Hkv, int64_t q_ld, int64_t k_ld,   \
        int64_t v_ld, int window, float scale_log2) {                                                                  \
    nanochat::flash_fwd_body<nanochat::Config<W, MT, N>>(                                                              \
          q, k, v, out, lse, T, H, Hkv, q_ld, k_ld, v_ld, window, scale_log2);                                         \
  }

NANOCHAT_FLASH_FWD(4, 1, 64)
NANOCHAT_FLASH_FWD(4, 2, 32)
NANOCHAT_FLASH_FWD(8, 1, 64)
NANOCHAT_FLASH_FWD(4, 2, 64)
NANOCHAT_FLASH_FWD(8, 1, 32)

namespace nanochat::kernels {

namespace {

using Kernel = void (*)(
      const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int, int, int, int64_t,
      int64_t, int64_t, int, float);

struct Variant {
  const char* name;
  Kernel kernel;
  int block_m, threads, smem;
};

template <int W, int MT, int N>
constexpr Variant variant(const char* name, Kernel kernel) {
  using C = Config<W, MT, N>;
  return {name, kernel, C::kBlockM, C::kThreads, C::kSmemBytes};
}

const Variant kVariants[kFlashVariants] = {
      variant<4, 1, 64>("4x16, n64", nanochat_flash_fwd_w4_m1_n64),
      variant<4, 2, 32>("4x32, n32", nanochat_flash_fwd_w4_m2_n32),
      variant<8, 1, 64>("8x16, n64", nanochat_flash_fwd_w8_m1_n64),
      variant<4, 2, 64>("4x32, n64", nanochat_flash_fwd_w4_m2_n64),
      variant<8, 1, 32>("8x16, n32", nanochat_flash_fwd_w8_m1_n32),
};
constexpr int kDefaultVariant = 0;

} // namespace

const char* flash_variant_name(int variant) {
  return kVariants[variant < 0 ? kDefaultVariant : variant].name;
}

void flash_fwd(
      const void* q, const void* k, const void* v, void* out, float* lse, int B, int64_t T, int H, int Hkv,
      int64_t q_ld, int64_t k_ld, int64_t v_ld, int64_t window, cudaStream_t stream, int variant) {
  const Variant& var = kVariants[variant < 0 ? kDefaultVariant : variant];
  static bool smem_set[kFlashVariants] = {};
  const int vi = variant < 0 ? kDefaultVariant : variant;
  if (!smem_set[vi]) {
    cudaFuncSetAttribute(var.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, var.smem);
    smem_set[vi] = true;
  }
  const dim3 grid(static_cast<unsigned>(T / var.block_m), H, B);
  const float scale_log2 = 1.4426950408889634f / sqrtf(static_cast<float>(kD));
  var.kernel<<<grid, var.threads, var.smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v), static_cast<__nv_bfloat16*>(out), lse, static_cast<int>(T), H, Hkv, q_ld,
        k_ld, v_ld, window >= 0 && window < T ? static_cast<int>(window) : -1, scale_log2);
}

} // namespace nanochat::kernels
