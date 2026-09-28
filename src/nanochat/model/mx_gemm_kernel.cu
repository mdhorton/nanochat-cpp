#include "nanochat/model/mx_gemm_kernel.h"

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

using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Mx = cutlass::mx_float8_t<cutlass::float_e4m3_t>;

// ElementC void: no source (beta 0)
template <class ElementC, class ElementD>
struct MxGemm {
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC, cutlass::layout::RowMajor, kAlignD,
        ElementD, cutlass::layout::RowMajor, kAlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Mx, cutlass::layout::RowMajor, 16, Mx,
        cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

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

template <class G>
const char* run(
      const void* a, const void* a_scale, const void* b, const void* b_scale, void* d, int64_t M, int64_t N, int64_t K,
      const float* alpha, bool accumulate, cudaStream_t stream) {
  using Gemm = typename G::Gemm;
  using Mainloop = typename G::Mainloop;
  using Sf = typename Mainloop::Sm1xxBlkScaledConfig;
  using ElementD = typename Gemm::ElementD;
  if (M % 128 != 0 || N % 128 != 0 || K % 128 != 0)
    return "cutlass_mx_gemm: M, N, K must be % 128";
  const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
  const auto stride_a = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {n, k, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {m, n, 1});
  typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {static_cast<const typename Mainloop::ElementA*>(a), stride_a,
         static_cast<const typename Mainloop::ElementB*>(b), stride_b,
         static_cast<const typename Mainloop::ElementSF*>(a_scale), Sf::tile_atom_to_shape_SFA(make_shape(m, n, k, 1)),
         static_cast<const typename Mainloop::ElementSF*>(b_scale), Sf::tile_atom_to_shape_SFB(make_shape(m, n, k, 1))},
        {{}, static_cast<const ElementD*>(d), stride_d, static_cast<ElementD*>(d), stride_d}};
  auto& scalars = args.epilogue.thread;
  scalars.alpha = 1.f;
  scalars.alpha_ptr = alpha;
  scalars.beta = accumulate ? 1.f : 0.f;
  args.hw_info.sm_count = sm_count();
  Gemm gemm;
  if (Gemm::get_workspace_size(args) != 0)
    return "cutlass_mx_gemm: unexpected workspace";
  if (gemm.can_implement(args) != cutlass::Status::kSuccess)
    return "cutlass_mx_gemm: cannot implement (alignment?)";
  const auto status = gemm.run(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? nullptr : cutlassGetStatusString(status);
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

} // namespace nanochat::kernels
