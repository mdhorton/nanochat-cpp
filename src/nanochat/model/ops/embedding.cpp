#include "nanochat/model/ops/embedding.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/ops/embedding_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using Grads = kernels::EmbeddingGrads;

class Embeddings : public torch::autograd::Function<Embeddings> {
public:
  static variable_list forward(AutogradContext* ctx, const torch::Tensor& idx, at::TensorList weights) {
    TORCH_CHECK(static_cast<int>(weights.size()) <= Grads::kMaxTables, "too many embedding tables");
    variable_list out;
    for (const auto& w : weights) {
      TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.is_contiguous(), "expected bf16 tables");
      TORCH_CHECK(w.size(1) % 8 == 0 && w.size(1) <= Grads::kMaxCols, "embedding dim must be a multiple of 8, <= 2048");
      out.push_back(at::embedding(w, idx));
    }
    ctx->set_materialize_grads(false);
    ctx->save_for_backward({idx});
    ctx->saved_data["weights"] = weights.vec(); // the parameters themselves: backward writes their .grad
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto idx = ctx->get_saved_variables()[0].flatten();
    auto weights = ctx->saved_data["weights"].toTensorVector();
    const auto [ids, order] = idx.sort(/*stable=*/std::optional<bool>(true), 0);
    Grads a{};
    std::vector<torch::Tensor> keep; // contiguous upstream gradients, alive until the kernel is queued
    for (size_t i = 0; i < weights.size(); ++i) {
      if (!grads[i].defined())
        continue;
      auto& w = weights[i];
      if (!w.grad().defined())
        w.mutable_grad() = torch::zeros_like(w);
      const auto& grad = w.grad();
      TORCH_CHECK(grad.scalar_type() == torch::kBFloat16 && grad.is_contiguous(), "expected a contiguous bf16 .grad");
      keep.push_back(grads[i].contiguous());
      TORCH_CHECK(keep.back().scalar_type() == torch::kBFloat16 && keep.back().numel() == idx.numel() * w.size(1));
      a.g[a.tables] = keep.back().data_ptr();
      a.grad[a.tables] = grad.data_ptr();
      a.cols[a.tables] = static_cast<int>(w.size(1));
      ++a.tables;
    }
    kernels::embedding_grad_accumulate(
          ids.data_ptr<int64_t>(), order.data_ptr<int64_t>(), idx.numel(), a,
          at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return variable_list(1 + weights.size()); // accumulated in place: no gradients to return
  }
};

} // namespace

std::vector<torch::Tensor> embeddings(const torch::Tensor& idx, const std::vector<torch::Tensor>& weights) {
  return Embeddings::apply(idx, at::TensorList(weights));
}

} // namespace nanochat
