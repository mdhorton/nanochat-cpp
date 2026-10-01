#include "nanochat/model/mx_gemm_kernel.h"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "nanochat/model/mx_kernel.cuh"
#include "nanochat/model/nvfp4_alpha.cuh"

namespace {

using namespace cute;

using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Mx = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Nv = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// MXFP8 operands, or NVFP4 (Element Nv, alignment 32, its pingpong schedule)
template <
      class Epilogue, class Element = Mx, int kAlign = 16,
      class Schedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong>
struct Kernel {
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Element, cutlass::layout::RowMajor, kAlign,
        Element, cutlass::layout::ColumnMajor, kAlign, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        Schedule>::CollectiveOp;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>>;
};

// ElementC void: no source (beta 0)
template <class ElementC, class ElementD>
struct MxGemm : Kernel<typename cutlass::epilogue::collective::CollectiveBuilder<
                      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
                      cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC,
                      cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<ElementD>::value, ElementD,
                      cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<ElementD>::value,
                      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp> {};

namespace fusion = cutlass::epilogue::fusion;
using EpiTile = Shape<_64, _32>;
using StrideH = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;

// The transpose (N, M) row-major: e4m3, MX along M (scales in the swizzled layout with M / 128 tiles), or NVFP4 when
// fp4.data is set. rows.data set: the values (M, N) as NVFP4 along N too. amax: NVFP4 operands' (device scalars),
// the accumulator's scale their tensor scales' product (null: 1).
struct MxT {
  uint8_t* data;
  uint8_t* scale;
  nanochat::kernels::Nvfp4Out fp4, rows;
  const float* amax[2];
};

constexpr int kEpiM = 64, kPitch = kEpiM + 8; // MxStoreT's subtile rows and its stage's row pitch

// MxStoreT's MS-EDEN row store of one staged 64x64 subtile (origin (m, n)): a lane pair per row, each 32 values
// (columns 16 half + 32 i + j), with nvfp4_eden_store8's rotation (H64's stages over the index bits in the same
// order), rounding and group sums (8-value runs, then pairs, then halves: the same bits as quantize_mx's). Out of
// line as mx_store_subtile. Returns smax updated.
__device__ __noinline__ float eden_rows_subtile(
      const MxT* __restrict__ params, const cutlass::bfloat16_t* stage, int t, int64_t m, int64_t n, float smax) {
  const nanochat::kernels::Nvfp4Out rows = params->rows;
  const int r = t / 2, half = t % 2;
  float f[2][16];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i) {
    const cutlass::bfloat16_t* s = stage + (half + 2 * i) * 16 * kPitch + r;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 16; ++j)
      f[i][j] = static_cast<float>(s[j * kPitch]);
  }
  // x . diag(signs) / 8, then H64: index bits 0-3 within a thread's block, bit 4 across the lane pair, bit 5 across
  // its two blocks
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i) {
    const uint32_t bits = static_cast<uint32_t>(rows.eden_signs >> ((half + 2 * i) * 16)) & 0xffffu;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 16; ++j)
      f[i][j] *= (bits >> j) & 1u ? -nanochat::kEdenMul : nanochat::kEdenMul;
    CUTLASS_PRAGMA_UNROLL
    for (int h = 1; h < 16; h *= 2)
      CUTLASS_PRAGMA_UNROLL
    for (int a = 0; a < 16; a += 2 * h)
      CUTLASS_PRAGMA_UNROLL
    for (int j = a; j < a + h; ++j) {
      const float x = f[i][j], y = f[i][j + h];
      f[i][j] = x + y, f[i][j + h] = x - y;
    }
  }
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i)
    CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 16; ++j) {
    const float p = __shfl_xor_sync(0xffffffff, f[i][j], 1);
    f[i][j] = half ? p - f[i][j] : f[i][j] + p;
  }
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 16; ++j) {
    const float x = f[0][j], y = f[1][j];
    f[0][j] = x + y, f[1][j] = x - y;
  }
  // each block's scale and codes, and the group's ||a||², <a, q> in block units
  float s[2], num[2], den[2];
  uint2 codes[2];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i) {
    float mx = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 16; ++j)
      mx = fmaxf(mx, fabsf(f[i][j]));
    s[i] = nanochat::nvfp4_round_scale(mx * (1.f / nanochat::kEdenValMax));
    const float to_q = s[i] > 0.f ? nanochat::nvfp4_rcp(s[i]) : 0.f;
    float run_num[2], run_den[2];
    uint32_t c[2];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 2; ++k) {
      float a[8], q[8];
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j)
        a[j] = f[i][8 * k + j] * to_q;
      c[k] = nanochat::e2m1x8_rn(a);
      nanochat::e2m1x8_values(c[k], q);
      run_num[k] = 0.f, run_den[k] = 0.f;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j)
        run_num[k] = fmaf(a[j], a[j], run_num[k]), run_den[k] = fmaf(a[j], q[j], run_den[k]);
    }
    codes[i] = make_uint2(c[0], c[1]);
    num[i] = run_num[0] + run_num[1], den[i] = run_den[0] + run_den[1];
    num[i] += __shfl_xor_sync(0xffffffff, num[i], 1);
    den[i] += __shfl_xor_sync(0xffffffff, den[i], 1);
  }
  const float gnum = num[0] + num[1], gden = den[0] + den[1];
  const float corr = gden > 0.f ? gnum / gden : 1.f;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i) {
    const int64_t row = m + r, col = n + (half + 2 * i) * 16, idx = row * rows.ld + col;
    const uint32_t rnd = nanochat::nvfp4_random(rows.seed, (rows.index0 + idx) / 16);
    const float sr = __uint_as_float((__float_as_uint(s[i] * corr) + (rnd & 0xfffffu)) & 0xfff00000u);
    *reinterpret_cast<uint2*>(static_cast<uint8_t*>(rows.data) + idx / 2) = codes[i];
    nanochat::nvfp4_store_scale(rows, row, col, sr);
    smax = fmaxf(smax, sr);
  }
  return smax;
}

// mx_store_subtile's body: p must not alias the stores. store_rows false: rows done already (eden_rows_subtile).
template <int kEpiN>
CUTLASS_DEVICE float2 mx_store_subtile_body(
      const MxT& p, const cutlass::bfloat16_t* stage, int t, int64_t M, int64_t m, int64_t n, float2 smax,
      bool store_rows = true) {
  constexpr int kP = kEpiN / 32;
  const nanochat::kernels::Nvfp4Out rows = p.rows, fp4 = p.fp4; // loaded up front: one latency
  // lanes 4c .. 4c + 3: column c's 16-row quarters; lanes 4c, 4c + 1 hold block 0, 4c + 2, 4c + 3 block 1
  const int col = t / 4, quarter = t % 4;
  if (store_rows && rows.data != nullptr) {
    // row r's columns 16 half .. 16 half + 15: a warp reads 32 consecutive rows at a time, conflict-free
    const int r = t % kEpiM, half = t / kEpiM;
    float f[kP][16];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kP; ++i) {
      const cutlass::bfloat16_t* s = stage + (half + 2 * i) * 16 * kPitch + r;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 16; ++j)
        f[i][j] = static_cast<float>(s[j * kPitch]);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kP; ++i)
      smax.y = fmaxf(smax.y, nanochat::nvfp4_store16(f[i], rows, m + r, n + (half + 2 * i) * 16));
  }
  uint4 raw[kP][2];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kP; ++i) {
    const auto* src = reinterpret_cast<const uint4*>(stage + (col + 32 * i) * kPitch + quarter * 16);
    raw[i][0] = src[0], raw[i][1] = src[1];
  }
  const int64_t mq = m + quarter * 16;
  if (fp4.data != nullptr) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kP; ++i) {
      const auto* v = reinterpret_cast<const __nv_bfloat162*>(raw[i]);
      float f[16];
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        const float2 q = __bfloat1622float2(v[j]);
        f[2 * j] = q.x, f[2 * j + 1] = q.y;
      }
      smax.x = fmaxf(smax.x, nanochat::nvfp4_store16(f, fp4, n + col + 32 * i, mq));
    }
    return smax;
  }
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kP; ++i) {
    const auto* v = reinterpret_cast<const __nv_bfloat162*>(raw[i]);
    const int64_t nc = n + col + 32 * i;
    __nv_bfloat162 amax2 = __habs2(v[0]);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 1; j < 8; ++j)
      amax2 = __hmax2(amax2, __habs2(v[j]));
    float amax = fmaxf(__low2float(amax2), __high2float(amax2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    const int e = nanochat::mx_exponent(amax);
    const float mul = nanochat::mx_multiplier(e);
    uint4 q4;
    auto* q = reinterpret_cast<__nv_fp8x2_storage_t*>(&q4);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 8; ++j) {
      const float2 f = __bfloat1622float2(v[j]);
      q[j] = __nv_cvt_float2_to_fp8x2(make_float2(f.x * mul, f.y * mul), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint4*>(p.data + nc * M + mq) = q4;
    if (quarter % 2 == 0)
      p.scale[nanochat::mx_scale_index(nc, mq / 32, M / 128)] = static_cast<uint8_t>(e);
  }
  return smax;
}

// MxStoreT's quantization of one staged 64 x kEpiN subtile (origin (m, n)) by thread t: out of line, so the
// epilogue's unrolled subtiles share one copy (inlined, they overflow the instruction cache). Each thread takes kEpiN /
// 32 columns' and rows' blocks, loading them all first. smax: (fp4's, rows') so far, returned updated. params: 64x64
// copies them all, which the compiler then passes in uniform registers; that spills in the 64x32 kernels, so restrict
// instead (the stores can't change params, so they aren't reloaded after each).
// kEden: the rows are MS-EDEN (eden_rows_subtile; a separate instantiation, so the other keeps its registers)
template <int kEpiN, bool kEden = false>
__device__ __noinline__ float2 mx_store_subtile(
      const MxT* __restrict__ params, const cutlass::bfloat16_t* stage, int t, int64_t M, int64_t m, int64_t n,
      float2 smax) {
  if constexpr (kEden) {
    static_assert(kEpiN == 64, "MS-EDEN rows: 64-groups");
    smax.y = eden_rows_subtile(params, stage, t, m, n, smax.y);
    const MxT p = *params;
    return mx_store_subtile_body<kEpiN>(p, stage, t, M, m, n, smax, false);
  }
  else if constexpr (kEpiN == 64) {
    const MxT p = *params;
    return mx_store_subtile_body<kEpiN>(p, stage, t, M, m, n, smax);
  }
  else {
    return mx_store_subtile_body<kEpiN>(*params, stage, t, M, m, n, smax);
  }
}

// Epilogue node: returns Fn(alpha, its children's values) per element (bf16) and writes their transpose quantized
// along M (as quantize_mx's out_t), and with rows the values as NVFP4 along N (kEdenRows: MS-EDEN when rows.eden). Each
// 64 x kEpiN subtile (the collective's EpiTile) is staged in smem transposed; each of the 128 threads then quantizes 16
// rows of kEpiN / 32 columns, lane pairs sharing a 32-row block's amax (NVFP4: one 16-value block each), and for rows
// 16 columns of kEpiN / 32 rows. Pingpong: one warpgroup per tile, the two warpgroups' epilogues serialized.
template <int FragmentSize, class Fn, int kEpiN = 32, bool kEdenRows = false>
struct MxStoreT {
  static constexpr int kThreads = 128;
  static_assert(kEpiN % 32 == 0, "32 columns x 4 quarters = 128 threads");
  static_assert(!kEdenRows || kEpiN == 64, "MS-EDEN rows: 64-groups");

  // [subtile parity][col][row], rows padded to 72: conflict-free fragment writes and 16 B quarter reads. The collective
  // syncs the warpgroup before reduce, so a subtile's stage is complete there, and its readers are done before the
  // subtile after next writes the buffer again.
  static constexpr int kStage = kEpiN * kPitch;

  struct SharedStorage {
    cutlass::array_aligned<cutlass::bfloat16_t, 2 * kStage, 16> stage;
  };

  using Arguments = MxT;
  using Params = Arguments;
  using ElementAux = cutlass::bfloat16_t; // the element type when this is the root and D is void

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(const ProblemShape&, const Arguments& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(const ProblemShape& problem_shape, const Arguments&) {
    auto [M, N, K, L] = append<4>(problem_shape, 1);
    return M % 128 == 0 && N % 128 == 0;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(const ProblemShape&, const Arguments&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
        const ProblemShape&, const Arguments&, void*, cudaStream_t, cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE MxStoreT() {}

  CUTLASS_HOST_DEVICE MxStoreT(const Params& params, const SharedStorage& shared)
      : params(&params)
      , stage(const_cast<cutlass::bfloat16_t*>(shared.stage.data())) {}

  const Params* params = nullptr;
  cutlass::bfloat16_t* stage = nullptr;

  CUTLASS_DEVICE bool is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(const fusion::detail::ProducerLoadArgs<Args...>&) {
    return fusion::EmptyProducerLoadCallbacks{};
  }

  template <class CoordTensor>
  struct ConsumerStoreCallbacks : fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE ConsumerStoreCallbacks(
          CoordTensor tCcD, int thr_m, int thr_n, const Params* params, cutlass::bfloat16_t* stage, int64_t M, int m0,
          int n0, int thread_idx)
        : tCcD(tCcD)
        , thr_m(thr_m)
        , thr_n(thr_n)
        , params(params)
        , stage(stage)
        , M(M)
        , m0(m0)
        , n0(n0)
        , thread_idx(thread_idx)
        , alpha(params->amax[0] != nullptr ? *params->amax[0] / (nanochat::kE2m1Max * nanochat::kE4m3Max) *
                                                   (*params->amax[1] / (nanochat::kE2m1Max * nanochat::kE4m3Max))
                                           : 1.f) {}

    CoordTensor tCcD; // (CPY, CPY_M, CPY_N, EPI_M, EPI_N): (m, n) relative to the thread's first element
    int thr_m, thr_n; // that element's (m, n)
    const Params* params;
    cutlass::bfloat16_t* stage;
    int64_t M;
    int m0, n0; // the CTA tile's origin
    int thread_idx;
    float alpha;
    float smax = 0.f, smax_rows = 0.f; // NVFP4: the tile's max block scale so far

    template <class ElementAccumulator, class... ElementIn>
    CUTLASS_DEVICE cutlass::Array<float, FragmentSize> visit(
          const cutlass::Array<ElementAccumulator, FragmentSize>&, int epi_v, int epi_m, int epi_n,
          const cutlass::Array<ElementIn, FragmentSize>&... frg) {
      cutlass::bfloat16_t* buf = stage + (epi_m & 1) * kStage;
      auto crd = tCcD(_, _, _, epi_m, epi_n);
      cutlass::Array<float, FragmentSize> out;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        const cutlass::bfloat16_t v = Fn{}(alpha, static_cast<float>(frg[i])...);
        const auto c = crd(epi_v * FragmentSize + i);
        const int row = (thr_m + get<0>(c)) % kEpiM, col = (thr_n + get<1>(c)) % kEpiN;
        buf[col * kPitch + row] = v;
        out[i] = static_cast<float>(v);
      }
      return out;
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(STensor&&, const SyncFn&, int epi_m, int epi_n, bool, VTensor) {
      const cutlass::bfloat16_t* buf = stage + (epi_m & 1) * kStage;
      const int t = thread_idx % kThreads;
      const int64_t m = m0 + epi_m * kEpiM, n = n0 + epi_n * kEpiN;
      float2 r;
      if constexpr (kEdenRows)
        if (params->rows.eden)
          r = mx_store_subtile<kEpiN, true>(params, buf, t, M, m, n, make_float2(smax, smax_rows));
        else
          r = mx_store_subtile<kEpiN>(params, buf, t, M, m, n, make_float2(smax, smax_rows));
      else
        r = mx_store_subtile<kEpiN>(params, buf, t, M, m, n, make_float2(smax, smax_rows));
      smax = r.x, smax_rows = r.y;
    }

    CUTLASS_DEVICE void end() {
      if (params->fp4.data != nullptr)
        nanochat::nvfp4_smax(params->fp4, smax);
      if (params->rows.data != nullptr)
        nanochat::nvfp4_smax(params->rows, smax_rows);
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(const fusion::detail::ConsumerStoreArgs<Args...>& args) {
    using E = decltype(args.epi_tile);
    static_assert(size<0>(E{}) == kEpiM && size<1>(E{}) == kEpiN, "MxStoreT's subtile is the EpiTile");
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    // residue: (M, N) - the thread's first (m, n)
    const int thr_m = static_cast<int>(M - get<0>(args.residue_tCcD)),
              thr_n = static_cast<int>(N - get<1>(args.residue_tCcD));
    return ConsumerStoreCallbacks<decltype(args.tCcD)>(
          args.tCcD, thr_m, thr_n, params, stage, M, m * size<0>(Tile{}), n * size<1>(Tile{}), args.thread_idx);
  }
};

using RowScales =
      fusion::Sm120BlockScaleFactorRowStore<32, EpiTile, Tile, 4, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t>;

template <class Tree>
using MxGemmTree = Kernel<typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster, EpiTile, float, float, void,
      cutlass::layout::RowMajor, 16, cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
      cutlass::epilogue::collective::EpilogueScheduleAuto, Tree>::CollectiveOp>;

// bf16(relu(bf16(h))^2): fp8_kernel.cu's ReluSquare on the bf16 h (h = alpha * acc)
struct ReluSquare {
  CUTLASS_DEVICE cutlass::bfloat16_t operator()(float alpha, float v) const {
    const float h = static_cast<float>(cutlass::bfloat16_t(alpha * v));
    const float r = h <= 0.f ? 0.f : h;
    return cutlass::bfloat16_t(r * r);
  }
};

// Epilogue: h (bf16, through smem + TMA) -> relu^2, its transpose MX-quantized along M -> MX scales per 32 along rows
// -> D e4m3. The epilogue tile is the 64x32 EpilogueTileAuto picks.
template <class E>
using HStoreT = fusion::Sm90AuxStore<
      2, E, cutlass::bfloat16_t, cutlass::FloatRoundStyle::round_to_nearest, StrideH,
      decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
               StrideH, cutlass::bfloat16_t, E>()),
      decltype(cutlass::epilogue::collective::detail::sm120_get_smem_store_op_for_accumulator<
               StrideH, cutlass::bfloat16_t>())>;
using MxGemmReluSquare = MxGemmTree<fusion::Sm90EVT<
      RowScales, fusion::Sm90EVT<MxStoreT<4, ReluSquare>, fusion::Sm90EVT<HStoreT<EpiTile>, fusion::Sm90AccFetch>>>>;

// dh = bf16(h > 0 ? bf16(ga) * 2h : 0), as quantize_mx_relu_square_bwd (ga = alpha * acc)
struct ReluSquareGrad {
  CUTLASS_DEVICE cutlass::bfloat16_t operator()(float alpha, float ga, float h) const {
    const float g = static_cast<float>(cutlass::bfloat16_t(alpha * ga));
    return cutlass::bfloat16_t(h <= 0.f ? 0.f : g * (2.f * h));
  }
};

// Epilogue: h (bf16, TMA load) and the accumulator (ga) -> dh, its transpose MX-quantized along M -> MX scales per 32
// along rows -> D e4m3
template <class E>
using HLoadT = fusion::Sm90AuxLoad<
      2, E, cutlass::bfloat16_t, StrideH,
      decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
               StrideH, cutlass::bfloat16_t, E>()),
      decltype(cutlass::epilogue::collective::detail::sm120_get_smem_load_op_for_source<
               StrideH, cutlass::bfloat16_t>())>;
using HLoad = HLoadT<EpiTile>;
using MxGemmReluSquareGrad =
      MxGemmTree<fusion::Sm90EVT<RowScales, fusion::Sm90EVT<MxStoreT<4, ReluSquareGrad>, fusion::Sm90AccFetch, HLoad>>>;

// The same with NVFP4 operands: no D, MxStoreT writes dh as NVFP4 (rows) and its transpose. 64x64 subtiles: half
// the per-subtile syncs, two independent quantize chains per thread.
using EpiTile64 = Shape<_64, _64>;
using Nvfp4GemmReluSquareGrad = Kernel<
      typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster, EpiTile64, float, float,
            void, cutlass::layout::RowMajor, 8, void, cutlass::layout::RowMajor, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto,
            fusion::Sm90EVT<MxStoreT<4, ReluSquareGrad, 64, true>, fusion::Sm90AccFetch, HLoadT<EpiTile64>>>::
            CollectiveOp,
      Nv, 32, cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>;

// The forward's with NVFP4 operands: h = the accumulator times the tensor scales (Nvfp4Alpha), stored by TMA; no D,
// MxStoreT writes relu^2 as NVFP4 (rows) and its transpose. 64x64 subtiles as Nvfp4GemmReluSquareGrad.
using Nvfp4GemmReluSquare = Kernel<
      typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster, EpiTile64, float, float,
            void, cutlass::layout::RowMajor, 8, void, cutlass::layout::RowMajor, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto,
            fusion::Sm90EVT<
                  MxStoreT<4, ReluSquare, 64>,
                  fusion::Sm90EVT<
                        HStoreT<EpiTile64>,
                        fusion::Sm90EVT<
                              fusion::Sm90Compute<
                                    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
                              nanochat::Nvfp4Alpha, fusion::Sm90AccFetch>>>>::CollectiveOp,
      Nv, 32, cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>;

// the block scale store's norm constant (scales amax / 448)
__device__ const float kOne = 1.f;

// Also makes the device's primary context current: the TMA descriptors come from the driver API, which needs one, and
// torch's autograd threads may have none (cudaSetDevice skipped as the device already matches).
int sm_count() {
  static int counts[64] = {};
  thread_local int bound = -1;
  int device = 0;
  cudaGetDevice(&device);
  if (bound != device) {
    cudaSetDevice(device);
    bound = device;
  }
  if (counts[device] == 0)
    counts[device] = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  return counts[device];
}

const float* one() {
  static const float* ptrs[64] = {};
  int device = 0;
  cudaGetDevice(&device);
  if (ptrs[device] == nullptr) {
    void* p = nullptr;
    cudaGetSymbolAddress(&p, kOne);
    ptrs[device] = static_cast<const float*>(p);
  }
  return ptrs[device];
}

// everything but the epilogue
template <class G>
typename G::Gemm::Arguments make_args(
      const void* a, const void* a_scale, const void* b, const void* b_scale, int m, int n, int k) {
  using Gemm = typename G::Gemm;
  using Mainloop = typename G::Mainloop;
  using Sf = typename Mainloop::Sm1xxBlkScaledConfig;
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {m, n, k, 1};
  args.mainloop = {
        static_cast<const typename Mainloop::ElementA*>(a),
        cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {m, k, 1}),
        static_cast<const typename Mainloop::ElementB*>(b),
        cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {n, k, 1}),
        static_cast<const typename Mainloop::ElementSF*>(a_scale),
        Sf::tile_atom_to_shape_SFA(make_shape(m, n, k, 1)),
        static_cast<const typename Mainloop::ElementSF*>(b_scale),
        Sf::tile_atom_to_shape_SFB(make_shape(m, n, k, 1))};
  args.hw_info.sm_count = sm_count();
  return args;
}

template <class G>
const char* launch(const typename G::Gemm::Arguments& args, cudaStream_t stream) {
  using Gemm = typename G::Gemm;
  Gemm gemm;
  if (Gemm::get_workspace_size(args) != 0)
    return "cutlass_mx_gemm: unexpected workspace";
  if (gemm.can_implement(args) != cutlass::Status::kSuccess)
    return "cutlass_mx_gemm: cannot implement (alignment?)";
  const auto status = gemm.run(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? nullptr : cutlassGetStatusString(status);
}

bool fits(int64_t M, int64_t N, int64_t K) {
  return M % 128 == 0 && N % 128 == 0 && K % 128 == 0;
}

template <class G>
const char* run(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      const float* alpha, bool accumulate, cudaStream_t stream) {
  using ElementD = typename G::Gemm::ElementD;
  if (!fits(M, N, K))
    return "cutlass_mx_gemm: M, N, K must be % 128";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  auto args = make_args<G>(a, a_scale, b, b_scale, m, n, k);
  const auto stride_d = cutlass::make_cute_packed_stride(typename G::Gemm::GemmKernel::StrideD{}, {m, n, 1});
  args.epilogue = {{}, static_cast<const ElementD*>(d), stride_d, static_cast<ElementD*>(d), stride_d};
  auto& scalars = args.epilogue.thread;
  scalars.alpha = 1.f;
  scalars.alpha_ptr = alpha;
  scalars.beta = accumulate ? 1.f : 0.f;
  return launch<G>(args, stream);
}

} // namespace

namespace nanochat::kernels {

const char* cutlass_mx_gemm_bf16(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      cudaStream_t stream) {
  return run<MxGemm<void, cutlass::bfloat16_t>>(a, a_scale, b, b_scale, d, M, N, K, nullptr, false, stream);
}

const char* cutlass_mx_gemm_f32(
      const void* a, const void* a_scale, const void* b, const void* b_scale, float* d, int64_t M, int64_t N, int64_t K,
      const float* alpha, bool accumulate, cudaStream_t stream) {
  return run<MxGemm<float, float>>(a, a_scale, b, b_scale, d, M, N, K, alpha, accumulate, stream);
}

const char* cutlass_mx_gemm_relu_square(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* h, void* q, void* q_scale,
      void* q_t, void* q_t_scale, const Nvfp4Out& q_t_fp4, int64_t M, int64_t N, int64_t K, cudaStream_t stream) {
  using G = MxGemmReluSquare;
  if (!fits(M, N, K))
    return "cutlass_mx_gemm_relu_square: M, N, K must be % 128";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  auto args = make_args<G>(a, a_scale, b, b_scale, m, n, k);
  args.epilogue.ptr_D = static_cast<cutlass::float_e4m3_t*>(q);
  args.epilogue.dD = cutlass::make_cute_packed_stride(typename G::Gemm::GemmKernel::StrideD{}, {m, n, 1});
  // {{{{acc}, {h}}, {q^T}}, {scales}}
  args.epilogue.thread = {
        {{{}, {static_cast<cutlass::bfloat16_t*>(h), cutlass::make_cute_packed_stride(StrideH{}, {m, n, 1})}},
         {static_cast<uint8_t*>(q_t), static_cast<uint8_t*>(q_t_scale), q_t_fp4, {}, {}}},
        {static_cast<cutlass::float_ue8m0_t*>(q_scale), one(), {}}};
  return launch<G>(args, stream);
}

const char* cutlass_nvfp4_gemm_relu_square(
      const void* a, const void* a_scale, const void* b, const void* b_scale, const float* a_amax, const float* b_amax,
      void* h, const Nvfp4Out& q, void* q_t, void* q_t_scale, const Nvfp4Out& q_t_fp4, int64_t M, int64_t N, int64_t K,
      cudaStream_t stream) {
  using G = Nvfp4GemmReluSquare;
  if (!fits(M, N, K) || K % 256 != 0 || q.data == nullptr)
    return "cutlass_nvfp4_gemm_relu_square: M, N must be % 128, K % 256, q NVFP4";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  auto args = make_args<G>(a, a_scale, b, b_scale, m, n, k);
  // {{{{amaxes}, {acc}, {*}}, {h}}, {q^T, q}}
  args.epilogue.thread = {
        {{{a_amax, b_amax, nullptr}, {}, {}},
         {static_cast<cutlass::bfloat16_t*>(h), cutlass::make_cute_packed_stride(StrideH{}, {m, n, 1})}},
        {static_cast<uint8_t*>(q_t), static_cast<uint8_t*>(q_t_scale), q_t_fp4, q, {}}};
  return launch<G>(args, stream);
}

const char* cutlass_mx_gemm_relu_square_bwd(
      const void* a, const void* a_scale, const void* b, const void* b_scale, const void* h, void* dh, void* dh_scale,
      void* dh_t, void* dh_t_scale, const Nvfp4Out& dh_t_fp4, int64_t M, int64_t N, int64_t K, cudaStream_t stream) {
  using G = MxGemmReluSquareGrad;
  if (!fits(M, N, K))
    return "cutlass_mx_gemm_relu_square_bwd: M, N, K must be % 128";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  auto args = make_args<G>(a, a_scale, b, b_scale, m, n, k);
  args.epilogue.ptr_D = static_cast<cutlass::float_e4m3_t*>(dh);
  args.epilogue.dD = cutlass::make_cute_packed_stride(typename G::Gemm::GemmKernel::StrideD{}, {m, n, 1});
  // {{{acc}, {h}, {dh^T}}, {scales}}
  args.epilogue.thread = {
        {{},
         {static_cast<const cutlass::bfloat16_t*>(h), cutlass::bfloat16_t(0),
          cutlass::make_cute_packed_stride(StrideH{}, {m, n, 1})},
         {static_cast<uint8_t*>(dh_t), static_cast<uint8_t*>(dh_t_scale), dh_t_fp4, {}, {}}},
        {static_cast<cutlass::float_ue8m0_t*>(dh_scale), one(), {}}};
  return launch<G>(args, stream);
}

const char* cutlass_nvfp4_gemm_relu_square_bwd(
      const void* a, const void* a_scale, const void* b, const void* b_scale, const float* a_amax, const float* b_amax,
      const void* h, const Nvfp4Out& dh, void* dh_t, void* dh_t_scale, const Nvfp4Out& dh_t_fp4, int64_t M, int64_t N,
      int64_t K, cudaStream_t stream) {
  using G = Nvfp4GemmReluSquareGrad;
  if (!fits(M, N, K) || K % 256 != 0 || dh.data == nullptr)
    return "cutlass_nvfp4_gemm_relu_square_bwd: M, N must be % 128, K % 256, dh NVFP4";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  auto args = make_args<G>(a, a_scale, b, b_scale, m, n, k);
  // {{acc}, {h}, {dh^T, dh}}
  args.epilogue.thread = {
        {},
        {static_cast<const cutlass::bfloat16_t*>(h), cutlass::bfloat16_t(0),
         cutlass::make_cute_packed_stride(StrideH{}, {m, n, 1})},
        {static_cast<uint8_t*>(dh_t), static_cast<uint8_t*>(dh_t_scale), dh_t_fp4, dh, {a_amax, b_amax}}};
  return launch<G>(args, stream);
}

} // namespace nanochat::kernels
