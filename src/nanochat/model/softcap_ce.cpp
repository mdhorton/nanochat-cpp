#include "nanochat/model/softcap_ce.h"

#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/softcap_ce_kernel.h"

namespace nanochat {

    namespace {

        using torch::autograd::AutogradContext;
        using torch::autograd::variable_list;

        // Per chunk of rows: bf16 logits via cuBLAS, then the fused kernel (loss, and the logits' gradient in place),
        // then grad_x = dlogits @ w and grad_w += dlogits^T @ x (fp32 accumulate). Undefined outputs are skipped.
        void run_chunks(const torch::Tensor &x, const torch::Tensor &w, const torch::Tensor &targets, int64_t vocab,
                        double softcap, int64_t chunk_rows, const torch::Tensor &losses, const torch::Tensor &grad_x,
                        const torch::Tensor &grad_w, const torch::Tensor &grad_scale, const torch::Tensor &num_valid) {
            const int64_t N = x.size(0), padded = w.size(0);
            const bool grad = grad_x.defined();
            auto logits = torch::empty({std::min(N, chunk_rows), padded}, x.options());
            const auto stream = at::cuda::getCurrentCUDAStream().stream();
            for (int64_t begin = 0; begin < N; begin += chunk_rows) {
                const int64_t end = std::min(N, begin + chunk_rows);
                const auto xc = x.slice(0, begin, end);
                auto buf = logits.slice(0, 0, end - begin);
                at::mm_out(buf, xc, w.t());
                const int64_t scale_stride = grad_scale.defined() && grad_scale.dim() == 1 ? 1 : 0;
                kernels::softcap_ce(buf.data_ptr(), end - begin, padded, targets.data_ptr<int64_t>() + begin,
                                    static_cast<int>(vocab), static_cast<int>(padded), static_cast<float>(softcap),
                                    losses.defined() ? losses.data_ptr<float>() + begin : nullptr,
                                    grad_scale.defined() ? grad_scale.data_ptr<float>() + begin * scale_stride
                                                         : nullptr,
                                    scale_stride, num_valid.defined() ? num_valid.data_ptr<int64_t>() : nullptr, grad,
                                    stream);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                if (grad) {
                    auto gx = grad_x.slice(0, begin, end);
                    at::mm_out(gx, buf, w);
                    at::addmm_out(const_cast<torch::Tensor &>(grad_w), grad_w, buf.t(), xc, torch::kFloat32);
                }
            }
        }

        // For mean/sum the incoming gradient is a scalar, so forward computes the gradients at once (no recompute)
        // and backward only scales them. For none, backward recomputes each chunk with the per-row gradient.
        class SoftcapCrossEntropy : public torch::autograd::Function<SoftcapCrossEntropy> {
        public:
            static torch::Tensor forward(AutogradContext *ctx, const torch::Tensor &x, const torch::Tensor &weight,
                                         const torch::Tensor &targets, int64_t vocab_size, double softcap,
                                         int64_t chunk_rows, int64_t reduction_int, bool needs_grad) {
                const auto reduction = static_cast<LossReduction>(reduction_int);
                const auto w = weight.to(x.scalar_type());
                const auto num_valid = (targets >= 0).sum();
                auto losses = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
                const bool precompute = needs_grad && reduction != LossReduction::None;
                torch::Tensor grad_x, grad_w;
                if (precompute) {
                    grad_x = torch::empty_like(x);
                    grad_w = torch::zeros(weight.sizes(), weight.options().dtype(torch::kFloat32));
                }
                run_chunks(x, w, targets, vocab_size, softcap, chunk_rows, losses, grad_x, grad_w, {},
                           reduction == LossReduction::Mean ? num_valid : torch::Tensor());
                if (precompute)
                    ctx->save_for_backward({grad_x, grad_w});
                else if (needs_grad)
                    ctx->save_for_backward({x, weight, targets});
                ctx->saved_data["precomputed"] = precompute;
                ctx->saved_data["vocab_size"] = vocab_size;
                ctx->saved_data["softcap"] = softcap;
                ctx->saved_data["chunk_rows"] = chunk_rows;
                switch (reduction) {
                    case LossReduction::Mean:
                        return losses.sum() / num_valid; // ignored targets have loss 0
                    case LossReduction::Sum:
                        return losses.sum();
                    default:
                        return losses;
                }
            }

            static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
                const auto saved = ctx->get_saved_variables();
                if (saved.empty())
                    throw std::logic_error("softcap_cross_entropy: backward without grad mode in forward");
                const auto &g = grad_outputs[0];
                if (ctx->saved_data["precomputed"].toBool())
                    return {saved[0] * g, saved[1] * g.to(torch::kFloat32), {}, {}, {}, {}, {}, {}};

                const auto &x = saved[0], &weight = saved[1], &targets = saved[2];
                auto grad_x = torch::empty_like(x);
                auto grad_w = torch::zeros(weight.sizes(), weight.options().dtype(torch::kFloat32));
                run_chunks(x, weight.to(x.scalar_type()), targets, ctx->saved_data["vocab_size"].toInt(),
                           ctx->saved_data["softcap"].toDouble(), ctx->saved_data["chunk_rows"].toInt(), {}, grad_x,
                           grad_w, g.to(torch::kFloat32).contiguous(), {});
                return {grad_x, grad_w.to(weight.scalar_type()), {}, {}, {}, {}, {}, {}};
            }
        };

    } // namespace

    torch::Tensor softcap_cross_entropy(const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &targets,
                                        int64_t vocab_size, double softcap, int64_t chunk_rows, LossReduction reduction) {
        if (x.dim() != 2 || targets.dim() != 1 || targets.size(0) != x.size(0))
            throw std::invalid_argument("softcap_cross_entropy expects x (N, C) and targets (N)");
        if (chunk_rows < 1)
            throw std::invalid_argument("chunk_rows must be positive");
        if (!x.is_cuda() || x.scalar_type() != torch::kBFloat16)
            throw std::invalid_argument("softcap_cross_entropy expects bf16 CUDA activations");
        if (weight.size(0) % 8 != 0 || weight.size(0) < vocab_size)
            throw std::invalid_argument("softcap_cross_entropy expects the padded vocab to be a multiple of 8");
        const bool needs_grad = torch::GradMode::is_enabled() && (x.requires_grad() || weight.requires_grad());
        return SoftcapCrossEntropy::apply(x.contiguous(), weight, targets.to(torch::kInt64).contiguous(), vocab_size,
                                          softcap, chunk_rows, static_cast<int64_t>(reduction), needs_grad);
    }

} // namespace nanochat
