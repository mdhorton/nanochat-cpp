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
// fp4.data is set. rows.data set: the values (M, N) as NVFP4 along N too. alpha: the accumulator's scale (device
// scalar, null: 1).
struct MxT {
  uint8_t* data;
  uint8_t* scale;
  nanochat::kernels::Nvfp4Out fp4, rows;
  const float* alpha;
};

// Epilogue node: returns Fn(alpha, its children's values) per element (bf16) and writes their transpose quantized
// along M (as quantize_mx's out_t), and with rows the values as NVFP4 along N. Each 64x32 subtile is staged in smem
// transposed; each of the 128 threads then quantizes 16 rows of one column, lane pairs sharing a 32-row block's amax
// (NVFP4: one 16-value block each), and for rows 16 columns of one row. Pingpong: one warpgroup per tile, the two
// warpgroups' epilogues serialized.
template <int FragmentSize, class Fn>
struct MxStoreT {
  static constexpr int kEpiM = size<0>(EpiTile{}), kEpiN = size<1>(EpiTile{}), kThreads = 128;
  static_assert(kEpiM == 64 && kEpiN == 32, "32 columns x 4 quarters = 128 threads");

  // [subtile parity][col][row], rows padded to 72: conflict-free fragment writes and 16 B quarter reads. The collective
  // syncs the warpgroup before reduce, so a subtile's stage is complete there, and its readers are done before the
  // subtile after next writes the buffer again.
  static constexpr int kPitch = kEpiM + 8, kStage = kEpiN * kPitch;

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
        , alpha(params->alpha != nullptr ? *params->alpha : 1.f) {}

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
      // lanes 4c .. 4c + 3: column c's 16-row quarters; lanes 4c, 4c + 1 hold block 0, 4c + 2, 4c + 3 block 1
      const int t = thread_idx % kThreads, col = t / 4, quarter = t % 4;
      if (params->rows.data != nullptr) {
        // row r's columns 16 half .. 16 half + 15: a warp reads 32 consecutive rows at a time, conflict-free
        const int r = t % kEpiM, half = t / kEpiM;
        const cutlass::bfloat16_t* p = stage + (epi_m & 1) * kStage + half * 16 * kPitch + r;
        float f[16];
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 16; ++j)
          f[j] = static_cast<float>(p[j * kPitch]);
        smax_rows = fmaxf(
              smax_rows,
              nanochat::nvfp4_store16(f, params->rows, m0 + epi_m * kEpiM + r, n0 + epi_n * kEpiN + half * 16));
      }
      const auto* src = reinterpret_cast<const uint4*>(stage + (epi_m & 1) * kStage + col * kPitch + quarter * 16);
      uint4 raw[2] = {src[0], src[1]};
      const auto* v = reinterpret_cast<const __nv_bfloat162*>(raw);
      const int64_t n = n0 + epi_n * kEpiN + col, m = m0 + epi_m * kEpiM + quarter * 16;
      if (params->fp4.data != nullptr) {
        float f[16];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
          const float2 p = __bfloat1622float2(v[i]);
          f[2 * i] = p.x, f[2 * i + 1] = p.y;
        }
        smax = fmaxf(smax, nanochat::nvfp4_store16(f, params->fp4, n, m));
        return;
      }
      __nv_bfloat162 amax2 = __habs2(v[0]);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < 8; ++i)
        amax2 = __hmax2(amax2, __habs2(v[i]));
      float amax = fmaxf(__low2float(amax2), __high2float(amax2));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
      const int e = nanochat::mx_exponent(amax);
      const float mul = nanochat::mx_multiplier(e);
      uint4 q4;
      auto* q = reinterpret_cast<__nv_fp8x2_storage_t*>(&q4);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) {
        const float2 f = __bfloat1622float2(v[i]);
        q[i] = __nv_cvt_float2_to_fp8x2(make_float2(f.x * mul, f.y * mul), __NV_SATFINITE, __NV_E4M3);
      }
      *reinterpret_cast<uint4*>(params->data + n * M + m) = q4;
      if (quarter % 2 == 0)
        params->scale[nanochat::mx_scale_index(n, m / 32, M / 128)] = static_cast<uint8_t>(e);
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
using HStore = fusion::Sm90AuxStore<
      2, EpiTile, cutlass::bfloat16_t, cutlass::FloatRoundStyle::round_to_nearest, StrideH,
      decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
               StrideH, cutlass::bfloat16_t, EpiTile>()),
      decltype(cutlass::epilogue::collective::detail::sm120_get_smem_store_op_for_accumulator<
               StrideH, cutlass::bfloat16_t>())>;
using MxGemmReluSquare = MxGemmTree<fusion::Sm90EVT<
      RowScales, fusion::Sm90EVT<MxStoreT<4, ReluSquare>, fusion::Sm90EVT<HStore, fusion::Sm90AccFetch>>>>;

// dh = bf16(h > 0 ? bf16(ga) * 2h : 0), as quantize_mx_relu_square_bwd (ga = alpha * acc)
struct ReluSquareGrad {
  CUTLASS_DEVICE cutlass::bfloat16_t operator()(float alpha, float ga, float h) const {
    const float g = static_cast<float>(cutlass::bfloat16_t(alpha * ga));
    return cutlass::bfloat16_t(h <= 0.f ? 0.f : g * (2.f * h));
  }
};

// Epilogue: h (bf16, TMA load) and the accumulator (ga) -> dh, its transpose MX-quantized along M -> MX scales per 32
// along rows -> D e4m3
using HLoad = fusion::Sm90AuxLoad<
      2, EpiTile, cutlass::bfloat16_t, StrideH,
      decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
               StrideH, cutlass::bfloat16_t, EpiTile>()),
      decltype(cutlass::epilogue::collective::detail::sm120_get_smem_load_op_for_source<
               StrideH, cutlass::bfloat16_t>())>;
using MxGemmReluSquareGrad =
      MxGemmTree<fusion::Sm90EVT<RowScales, fusion::Sm90EVT<MxStoreT<4, ReluSquareGrad>, fusion::Sm90AccFetch, HLoad>>>;

// The same with NVFP4 operands: no D, MxStoreT writes dh as NVFP4 (rows) and its transpose
using Nvfp4GemmReluSquareGrad = Kernel<
      typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster, EpiTile, float, float, void,
            cutlass::layout::RowMajor, 8, void, cutlass::layout::RowMajor, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto,
            fusion::Sm90EVT<MxStoreT<4, ReluSquareGrad>, fusion::Sm90AccFetch, HLoad>>::CollectiveOp,
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
         {static_cast<uint8_t*>(q_t), static_cast<uint8_t*>(q_t_scale), q_t_fp4, {}, nullptr}},
        {static_cast<cutlass::float_ue8m0_t*>(q_scale), one(), {}}};
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
         {static_cast<uint8_t*>(dh_t), static_cast<uint8_t*>(dh_t_scale), dh_t_fp4, {}, nullptr}},
        {static_cast<cutlass::float_ue8m0_t*>(dh_scale), one(), {}}};
  return launch<G>(args, stream);
}

const char* cutlass_nvfp4_gemm_relu_square_bwd(
      const void* a, const void* a_scale, const void* b, const void* b_scale, const float* alpha, const void* h,
      const Nvfp4Out& dh, void* dh_t, void* dh_t_scale, const Nvfp4Out& dh_t_fp4, int64_t M, int64_t N, int64_t K,
      cudaStream_t stream) {
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
        {static_cast<uint8_t*>(dh_t), static_cast<uint8_t*>(dh_t_scale), dh_t_fp4, dh, alpha}};
  return launch<G>(args, stream);
}

} // namespace nanochat::kernels
