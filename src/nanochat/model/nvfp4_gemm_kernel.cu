#include "nanochat/model/nvfp4_gemm_kernel.h"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "nanochat/model/nvfp4.cuh"

namespace {

using namespace cute;
namespace fusion = cutlass::epilogue::fusion;

using Cluster = Shape<_1, _1, _1>;
using Nv = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// Epilogue leaf: alpha = a_amax / (6 * 448) * (b_amax / (6 * 448)) (* *alpha), the operands' tensor scales; null
// amaxes: 1. In the GEMM rather than a kernel of its own.
struct Nvfp4Alpha {
  struct SharedStorage {};

  struct Arguments {
    const float *a_amax = nullptr, *b_amax = nullptr, *alpha = nullptr;
  };

  using Params = Arguments;

  template <class P>
  static constexpr Params to_underlying_arguments(const P&, const Arguments& args, void*) {
    return args;
  }

  template <class P>
  static bool can_implement(const P&, const Arguments&) {
    return true;
  }

  template <class P>
  static size_t get_workspace_size(const P&, const Arguments&) {
    return 0;
  }

  template <class P>
  static cutlass::Status initialize_workspace(
        const P&, const Arguments&, void*, cudaStream_t, cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE Nvfp4Alpha() {}

  CUTLASS_HOST_DEVICE Nvfp4Alpha(const Params& params, const SharedStorage&)
      : params(params) {}

  Params params;

  CUTLASS_DEVICE bool is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(const fusion::ProducerLoadArgs<Args...>&) {
    return fusion::EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks : fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE explicit ConsumerStoreCallbacks(float scalar)
        : scalar(scalar) {}

    float scalar;

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<float, FragmentSize> visit(
          const cutlass::Array<ElementAccumulator, FragmentSize>&, int, int, int) {
      cutlass::Array<float, FragmentSize> f;
      f.fill(scalar);
      return f;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(const fusion::ConsumerStoreArgs<Args...>&) {
    constexpr float kNorm = nanochat::kE2m1Max * nanochat::kE4m3Max;
    float s = params.a_amax != nullptr ? *params.a_amax / kNorm * (*params.b_amax / kNorm) : 1.f;
    if (params.alpha != nullptr)
      s = s * *params.alpha;
    return ConsumerStoreCallbacks(s);
  }
};

// D = beta * C + alpha * acc: CUTLASS's LinearCombination with Nvfp4Alpha
template <class ElementD, class ElementC>
using ScaledLinComb = fusion::Sm90EVT<
      fusion::Sm90Compute<
            cutlass::homogeneous_multiply_add, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>,
      fusion::Sm90ScalarBroadcast<float, Stride<_0, _0, int64_t>>, fusion::Sm90SrcFetch<ElementC>,
      fusion::Sm90EVT<
            fusion::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
            Nvfp4Alpha, fusion::Sm90AccFetch>>;

// ElementC void: no source (beta 0)
template <class Tile, class Schedule, class ElementD, class ElementC = void>
struct Nvfp4Gemm {
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC, cutlass::layout::RowMajor, kAlignD,
        ElementD, cutlass::layout::RowMajor, kAlignD, cutlass::epilogue::collective::EpilogueScheduleAuto,
        ScaledLinComb<ElementD, ElementC>>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Nv, cutlass::layout::RowMajor, 32, Nv,
        cutlass::layout::ColumnMajor, 32, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        Schedule>::CollectiveOp;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>>;
};

using Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
using Cooperative = cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120;

// d = Nvfp4Alpha * a . b^T (+ d with accumulate)
template <class G>
const char* run(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      cudaStream_t stream, Nvfp4Alpha::Arguments alpha = {}, bool accumulate = false) {
  using Gemm = typename G::Gemm;
  using Mainloop = typename G::Mainloop;
  using Sf = typename Mainloop::Sm1xxBlkScaledConfig;
  using ElementD = typename Gemm::ElementD;
  if (M % 128 != 0 || N % 128 != 0 || K % 256 != 0)
    return "cutlass_nvfp4_gemm: M, N must be % 128, K % 256";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
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
  const auto stride_d = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {m, n, 1});
  args.epilogue = {{}, static_cast<const ElementD*>(d), stride_d, static_cast<ElementD*>(d), stride_d};
  // {beta, C, {alpha, acc, *}, +}
  args.epilogue.thread = {{{accumulate ? 1.f : 0.f}}, {}, {alpha, {}, {}}, {}};
  int device = 0;
  cudaGetDevice(&device);
  args.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  Gemm gemm;
  if (Gemm::get_workspace_size(args) != 0)
    return "cutlass_nvfp4_gemm: unexpected workspace";
  if (gemm.can_implement(args) != cutlass::Status::kSuccess)
    return "cutlass_nvfp4_gemm: cannot implement (alignment?)";
  const auto status = gemm.run(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? nullptr : cutlassGetStatusString(status);
}

struct Config {
  const char* name;
  const char* (*bf16)(
        const void*, const void*, const void*, const void*, void*, int64_t, int64_t, int64_t, cudaStream_t);
  const char* (*f32)(
        const void*, const void*, const void*, const void*, void*, int64_t, int64_t, int64_t, cudaStream_t);
};

// run with alpha 1, no accumulation
template <class G>
const char* run_plain(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      cudaStream_t stream) {
  return run<G>(a, a_scale, b, b_scale, d, M, N, K, stream);
}

template <class Tile, class Schedule>
constexpr Config config(const char* name) {
  return {name, run_plain<Nvfp4Gemm<Tile, Schedule, cutlass::bfloat16_t>>, run_plain<Nvfp4Gemm<Tile, Schedule, float>>};
}

const Config configs[] = {
      config<Shape<_128, _128, _128>, Pingpong>("128x128x128 pingpong"),
      config<Shape<_128, _128, _256>, Pingpong>("128x128x256 pingpong"),
      config<Shape<_128, _128, _256>, Cooperative>("128x128x256 cooperative"),
};

} // namespace

namespace nanochat::kernels {

int nvfp4_gemm_configs() {
  return static_cast<int>(std::size(configs));
}

const char* nvfp4_gemm_config_name(int config) {
  return configs[config].name;
}

int64_t nvfp4_scale_bytes(int64_t rows, int64_t K) {
  using Sf = Nvfp4Gemm<Shape<_128, _128, _128>, Pingpong, float>::Mainloop::Sm1xxBlkScaledConfig;
  return size(
        filter_zeros(Sf::tile_atom_to_shape_SFA(make_shape(static_cast<int>(rows), 128, static_cast<int>(K), 1))));
}

const char* cutlass_nvfp4_gemm(
      int config, const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, bool f32, int64_t M,
      int64_t N, int64_t K, cudaStream_t stream) {
  if (config < 0 || config >= nvfp4_gemm_configs())
    return "cutlass_nvfp4_gemm: bad config";
  const auto& c = configs[config];
  return (f32 ? c.f32 : c.bf16)(a, a_scale, b, b_scale, d, M, N, K, stream);
}

const char* cutlass_nvfp4_gemm_f32(
      const void* a, const void* a_scale, const void* b, const void* b_scale, float* d, int64_t M, int64_t N, int64_t K,
      const float* a_amax, const float* b_amax, const float* alpha, bool accumulate, cudaStream_t stream) {
  return run<Nvfp4Gemm<Shape<_128, _128, _128>, Pingpong, float, float>>(
        a, a_scale, b, b_scale, d, M, N, K, stream, {a_amax, b_amax, alpha}, accumulate);
}

const char* cutlass_nvfp4_gemm_bf16(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      const float* a_amax, const float* b_amax, cudaStream_t stream) {
  return run<Nvfp4Gemm<Shape<_128, _128, _128>, Pingpong, cutlass::bfloat16_t>>(
        a, a_scale, b, b_scale, d, M, N, K, stream, {a_amax, b_amax, nullptr});
}

} // namespace nanochat::kernels
