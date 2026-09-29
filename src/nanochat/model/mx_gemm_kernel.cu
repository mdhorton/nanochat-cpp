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

template <class Epilogue>
struct Kernel {
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Mx, cutlass::layout::RowMajor, 16, Mx,
        cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
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

// bf16(relu(bf16(v))^2): fp8_kernel.cu's ReluSquare on the bf16 h
template <class T>
struct ReluSquare {
  CUTLASS_HOST_DEVICE T operator()(const T& v) const {
    const float h = static_cast<float>(cutlass::bfloat16_t(v));
    const float r = h <= 0.f ? 0.f : h;
    return T(static_cast<float>(cutlass::bfloat16_t(r * r)));
  }
};

template <class T, int N>
struct ReluSquare<cutlass::Array<T, N>> {
  CUTLASS_HOST_DEVICE cutlass::Array<T, N> operator()(const cutlass::Array<T, N>& v) const {
    cutlass::Array<T, N> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i)
      out[i] = ReluSquare<T>{}(v[i]);
    return out;
  }
};

// Epilogue: h (bf16, through smem + TMA) -> relu^2 -> MX scales per 32 along rows -> D e4m3. The epilogue tile is
// the 64x32 EpilogueTileAuto picks.
namespace fusion = cutlass::epilogue::fusion;
using EpiTile = Shape<_64, _32>;
using StrideH = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
using HStore = fusion::Sm90AuxStore<
      2, EpiTile, cutlass::bfloat16_t, cutlass::FloatRoundStyle::round_to_nearest, StrideH,
      decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
               StrideH, cutlass::bfloat16_t, EpiTile>()),
      decltype(cutlass::epilogue::collective::detail::sm120_get_smem_store_op_for_accumulator<
               StrideH, cutlass::bfloat16_t>())>;
using ReluSquareTree = fusion::Sm90EVT<
      fusion::Sm120BlockScaleFactorRowStore<32, EpiTile, Tile, 4, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t>,
      fusion::Sm90EVT<
            fusion::Sm90Compute<ReluSquare, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
            fusion::Sm90EVT<HStore, fusion::Sm90AccFetch>>>;
using MxGemmReluSquare = Kernel<typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster, EpiTile, float, float, void,
      cutlass::layout::RowMajor, 16, cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
      cutlass::epilogue::collective::EpilogueScheduleAuto, ReluSquareTree>::CollectiveOp>;

// e4m3 (N, M) row-major, MX along M: scales in the swizzled layout with M / 128 tiles
struct MxT {
  uint8_t* data;
  uint8_t* scale;
};

// Epilogue node for relu^2's backward: from the accumulator (ga) and h, returns dh = bf16(h > 0 ? bf16(ga) * 2h : 0)
// and writes dh^T and relu(h)^2^T quantized along M, as quantize_mx_relu_square_bwd. Each 64x32 subtile is staged in
// smem transposed; each of the 128 threads then quantizes one 32-row block of one column of one tensor. Pingpong: one
// warpgroup per tile, the two warpgroups' epilogues serialized.
template <int FragmentSize>
struct ReluSquareGradT {
  static constexpr int kEpiM = size<0>(EpiTile{}), kEpiN = size<1>(EpiTile{}), kThreads = 128;
  static_assert(kEpiM == 64 && kEpiN == 32, "2 tensors x 2 row blocks x 32 columns = 128 threads");

  // [subtile parity][tensor][col][row], rows padded to 72: conflict-free fragment writes and 16 B block reads. The
  // collective syncs the warpgroup before reduce, so a subtile's stage is complete there, and its readers are done
  // before the subtile after next writes the buffer again.
  static constexpr int kPitch = kEpiM + 8, kStage = kEpiN * kPitch;

  struct SharedStorage {
    cutlass::array_aligned<cutlass::bfloat16_t, 2 * 2 * kStage, 16> stage;
  };

  struct Arguments {
    MxT dh_t, a_t;
  };

  using Params = Arguments;

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

  CUTLASS_HOST_DEVICE ReluSquareGradT() {}

  CUTLASS_HOST_DEVICE ReluSquareGradT(const Params& params, const SharedStorage& shared)
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
        , thread_idx(thread_idx) {}

    CoordTensor tCcD; // (CPY, CPY_M, CPY_N, EPI_M, EPI_N): (m, n) relative to the thread's first element
    int thr_m, thr_n; // that element's (m, n)
    const Params* params;
    cutlass::bfloat16_t* stage;
    int64_t M;
    int m0, n0; // the CTA tile's origin
    int thread_idx;

    template <class ElementAccumulator, class ElementG, class ElementH>
    CUTLASS_DEVICE cutlass::Array<float, FragmentSize> visit(
          const cutlass::Array<ElementAccumulator, FragmentSize>&, int epi_v, int epi_m, int epi_n,
          const cutlass::Array<ElementG, FragmentSize>& frg_g, const cutlass::Array<ElementH, FragmentSize>& frg_h) {
      using Bf16 = cutlass::bfloat16_t;
      Bf16* buf = stage + (epi_m & 1) * 2 * kStage;
      auto crd = tCcD(_, _, _, epi_m, epi_n);
      cutlass::Array<float, FragmentSize> dh;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        const float g = static_cast<float>(Bf16(static_cast<float>(frg_g[i])));
        const float h = static_cast<float>(frg_h[i]);
        const Bf16 d(h <= 0.f ? 0.f : g * (2.f * h));
        const float r = h <= 0.f ? 0.f : h;
        const auto c = crd(epi_v * FragmentSize + i);
        const int row = (thr_m + get<0>(c)) % kEpiM, col = (thr_n + get<1>(c)) % kEpiN;
        buf[col * kPitch + row] = d;
        buf[kStage + col * kPitch + row] = Bf16(r * r);
        dh[i] = static_cast<float>(d);
      }
      return dh;
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(STensor&&, const SyncFn&, int epi_m, int epi_n, bool, VTensor) {
      // warp: one tensor, 16 columns x 2 row blocks; lanes 2c, 2c + 1 hold blocks 0, 1 of column c
      const int t = thread_idx % kThreads, lane = t % 32, tensor = t / 64, col = t / 32 % 2 * 16 + lane / 2,
                blk = lane % 2;
      const auto* src = reinterpret_cast<const uint4*>(
            stage + (epi_m & 1) * 2 * kStage + tensor * kStage + col * kPitch + blk * 32);
      uint4 raw[4];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i)
        raw[i] = src[i];
      const auto* v = reinterpret_cast<const __nv_bfloat162*>(raw);
      __nv_bfloat162 amax2 = __habs2(v[0]);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < 16; ++i)
        amax2 = __hmax2(amax2, __habs2(v[i]));
      const int e = nanochat::mx_exponent(fmaxf(__low2float(amax2), __high2float(amax2)));
      const float mul = nanochat::mx_multiplier(e);
      uint4 q4[2];
      auto* q = reinterpret_cast<__nv_fp8x2_storage_t*>(q4);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 16; ++i) {
        const float2 f = __bfloat1622float2(v[i]);
        q[i] = __nv_cvt_float2_to_fp8x2(make_float2(f.x * mul, f.y * mul), __NV_SATFINITE, __NV_E4M3);
      }
      const MxT& out = tensor == 0 ? params->dh_t : params->a_t;
      const int64_t n = n0 + epi_n * kEpiN + col, m = m0 + epi_m * kEpiM + blk * 32;
      // swap halves across the lane pair so each store writes whole 32 B sectors: of the 64 B row segment, even lanes
      // store [0, 16) and [32, 48), odd lanes [16, 32) and [48, 64)
      const uint4 keep = blk == 0 ? q4[0] : q4[1], send = blk == 0 ? q4[1] : q4[0];
      uint4 recv;
      recv.x = __shfl_xor_sync(0xffffffff, send.x, 1);
      recv.y = __shfl_xor_sync(0xffffffff, send.y, 1);
      recv.z = __shfl_xor_sync(0xffffffff, send.z, 1);
      recv.w = __shfl_xor_sync(0xffffffff, send.w, 1);
      auto* seg = reinterpret_cast<uint4*>(out.data + n * M + m0 + epi_m * kEpiM);
      seg[blk] = blk == 0 ? keep : recv;
      seg[2 + blk] = blk == 0 ? recv : keep;
      out.scale[nanochat::mx_scale_index(n, m / 32, M / 128)] = static_cast<uint8_t>(e);
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

// Epilogue: h (bf16, TMA load) and the accumulator -> ReluSquareGradT (dh^T, relu(h)^2^T) -> MX scales per 32 along
// rows -> D e4m3 (dh)
using HLoad = fusion::Sm90AuxLoad<
      2, EpiTile, cutlass::bfloat16_t, StrideH,
      decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
               StrideH, cutlass::bfloat16_t, EpiTile>()),
      decltype(cutlass::epilogue::collective::detail::sm120_get_smem_load_op_for_source<
               StrideH, cutlass::bfloat16_t>())>;
using ReluSquareGradTree = fusion::Sm90EVT<
      fusion::Sm120BlockScaleFactorRowStore<32, EpiTile, Tile, 4, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t>,
      fusion::Sm90EVT<ReluSquareGradT<4>, fusion::Sm90AccFetch, HLoad>>;
using MxGemmReluSquareGrad = Kernel<typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster, EpiTile, float, float, void,
      cutlass::layout::RowMajor, 16, cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
      cutlass::epilogue::collective::EpilogueScheduleAuto, ReluSquareGradTree>::CollectiveOp>;

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
      int64_t M, int64_t N, int64_t K, cudaStream_t stream) {
  using G = MxGemmReluSquare;
  if (!fits(M, N, K))
    return "cutlass_mx_gemm_relu_square: M, N, K must be % 128";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  auto args = make_args<G>(a, a_scale, b, b_scale, m, n, k);
  args.epilogue.ptr_D = static_cast<cutlass::float_e4m3_t*>(q);
  args.epilogue.dD = cutlass::make_cute_packed_stride(typename G::Gemm::GemmKernel::StrideD{}, {m, n, 1});
  // {{{{acc}, {h}}, {relu^2}}, {scales}}
  args.epilogue.thread = {
        {{{}, {static_cast<cutlass::bfloat16_t*>(h), cutlass::make_cute_packed_stride(StrideH{}, {m, n, 1})}}, {}},
        {static_cast<cutlass::float_ue8m0_t*>(q_scale), one(), {}}};
  return launch<G>(args, stream);
}

const char* cutlass_mx_gemm_relu_square_bwd(
      const void* a, const void* a_scale, const void* b, const void* b_scale, const void* h, void* dh, void* dh_scale,
      void* dh_t, void* dh_t_scale, void* a_t, void* a_t_scale, int64_t M, int64_t N, int64_t K, cudaStream_t stream) {
  using G = MxGemmReluSquareGrad;
  if (!fits(M, N, K))
    return "cutlass_mx_gemm_relu_square_bwd: M, N, K must be % 128";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  auto args = make_args<G>(a, a_scale, b, b_scale, m, n, k);
  args.epilogue.ptr_D = static_cast<cutlass::float_e4m3_t*>(dh);
  args.epilogue.dD = cutlass::make_cute_packed_stride(typename G::Gemm::GemmKernel::StrideD{}, {m, n, 1});
  const auto u8 = [](void* p) {
    return static_cast<uint8_t*>(p);
  };
  // {{{acc}, {h}, {dh^T, a^T}}, {scales}}
  args.epilogue.thread = {
        {{},
         {static_cast<const cutlass::bfloat16_t*>(h), cutlass::bfloat16_t(0),
          cutlass::make_cute_packed_stride(StrideH{}, {m, n, 1})},
         {{u8(dh_t), u8(dh_t_scale)}, {u8(a_t), u8(a_t_scale)}}},
        {static_cast<cutlass::float_ue8m0_t*>(dh_scale), one(), {}}};
  return launch<G>(args, stream);
}

} // namespace nanochat::kernels
