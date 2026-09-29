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

namespace {

using namespace cute;

using Cluster = Shape<_1, _1, _1>;
using Nv = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// ElementC void: no source (beta 0)
template <class Tile, class Schedule, class ElementD, class ElementC = void>
struct Nvfp4Gemm {
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC, cutlass::layout::RowMajor, kAlignD,
        ElementD, cutlass::layout::RowMajor, kAlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
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

// d = alpha * a . b^T (+ d with accumulate); alpha: a device scalar, or null for 1
template <class G>
const char* run(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      cudaStream_t stream, const float* alpha = nullptr, bool accumulate = false) {
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
  auto& scalars = args.epilogue.thread;
  scalars.alpha = 1.f;
  scalars.alpha_ptr = alpha;
  scalars.beta = accumulate ? 1.f : 0.f;
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
      const float* alpha, bool accumulate, cudaStream_t stream) {
  return run<Nvfp4Gemm<Shape<_128, _128, _128>, Pingpong, float, float>>(
        a, a_scale, b, b_scale, d, M, N, K, stream, alpha, accumulate);
}

const char* cutlass_nvfp4_gemm_bf16(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      const float* alpha, cudaStream_t stream) {
  return run<Nvfp4Gemm<Shape<_128, _128, _128>, Pingpong, cutlass::bfloat16_t>>(
        a, a_scale, b, b_scale, d, M, N, K, stream, alpha);
}

} // namespace nanochat::kernels
