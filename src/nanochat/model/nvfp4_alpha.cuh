// The NVFP4 GEMMs' epilogue leaf (CUTLASS EVT) for the operands' tensor scales.
#pragma once

#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>

#include "nanochat/model/nvfp4.cuh"

namespace nanochat {

// alpha = a_amax / (6 * 448) * (b_amax / (6 * 448)) (* *alpha), the operands' tensor scales; null amaxes: 1. In the
// GEMM rather than a kernel of its own.
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
  CUTLASS_DEVICE auto get_producer_load_callbacks(const cutlass::epilogue::fusion::ProducerLoadArgs<Args...>&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
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
  CUTLASS_DEVICE auto get_consumer_store_callbacks(const cutlass::epilogue::fusion::ConsumerStoreArgs<Args...>&) {
    constexpr float kNorm = nanochat::kE2m1Max * nanochat::kE4m3Max;
    float s = params.a_amax != nullptr ? *params.a_amax / kNorm * (*params.b_amax / kNorm) : 1.f;
    if (params.alpha != nullptr)
      s = s * *params.alpha;
    return ConsumerStoreCallbacks(s);
  }
};

} // namespace nanochat
