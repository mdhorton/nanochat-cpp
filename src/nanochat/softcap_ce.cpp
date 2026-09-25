#include "nanochat/softcap_ce.h"

#include <stdexcept>

namespace nanochat {

    namespace {

        using torch::autograd::AutogradContext;
        using torch::autograd::variable_list;
        using torch::indexing::Slice;

        namespace F = torch::nn::functional;

        // Per-row losses of rows [begin, end), with the same ops as GPT::forward unchunked.
        torch::Tensor chunk_losses(const torch::Tensor &x, const torch::Tensor &w, const torch::Tensor &targets,
                                   int64_t vocab_size, double softcap) {
            auto logits = torch::matmul(x, w.t()).slice(1, 0, vocab_size).to(torch::kFloat32);
            logits = softcap * torch::tanh(logits / softcap);
            return F::cross_entropy(logits, targets,
                                    F::CrossEntropyFuncOptions().ignore_index(-1).reduction(torch::kNone));
        }

        // Backward recomputes each chunk with autograd on, so PyTorch's fused backward kernels do the work.
        class SoftcapCrossEntropy : public torch::autograd::Function<SoftcapCrossEntropy> {
        public:
            static torch::Tensor forward(AutogradContext *ctx, const torch::Tensor &x, const torch::Tensor &weight,
                                         const torch::Tensor &targets, int64_t vocab_size, double softcap,
                                         int64_t chunk_rows, int64_t reduction) {
                const int64_t N = x.size(0);
                const auto w = weight.to(x.scalar_type());
                auto losses = torch::empty({N}, x.options().dtype(torch::kFloat32));
                for (int64_t begin = 0; begin < N; begin += chunk_rows) {
                    const int64_t end = std::min(N, begin + chunk_rows);
                    losses.slice(0, begin, end)
                            .copy_(chunk_losses(x.slice(0, begin, end), w, targets.slice(0, begin, end), vocab_size,
                                                softcap));
                }
                const auto num_valid = (targets >= 0).sum();
                ctx->save_for_backward({x, weight, targets, num_valid});
                ctx->saved_data["vocab_size"] = vocab_size;
                ctx->saved_data["softcap"] = softcap;
                ctx->saved_data["chunk_rows"] = chunk_rows;
                ctx->saved_data["reduction"] = reduction;
                switch (static_cast<LossReduction>(reduction)) {
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
                const auto &x = saved[0], &weight = saved[1], &targets = saved[2], &num_valid = saved[3];
                const int64_t vocab_size = ctx->saved_data["vocab_size"].toInt();
                const double softcap = ctx->saved_data["softcap"].toDouble();
                const int64_t chunk_rows = ctx->saved_data["chunk_rows"].toInt();
                const auto reduction = static_cast<LossReduction>(ctx->saved_data["reduction"].toInt());

                const int64_t N = x.size(0);
                auto row_grad = grad_outputs[0]; // d loss / d loss_i
                if (reduction == LossReduction::Mean)
                    row_grad = (row_grad / num_valid).expand({N});
                else if (reduction == LossReduction::Sum)
                    row_grad = row_grad.expand({N});

                auto grad_x = torch::empty_like(x);
                auto grad_w = torch::zeros_like(weight);
                const auto w_leaf = weight.detach().requires_grad_();
                for (int64_t begin = 0; begin < N; begin += chunk_rows) {
                    const int64_t end = std::min(N, begin + chunk_rows);
                    const auto x_leaf = x.slice(0, begin, end).detach().requires_grad_();
                    torch::AutoGradMode enable_grad(true);
                    auto losses = chunk_losses(x_leaf, w_leaf.to(x.scalar_type()), targets.slice(0, begin, end),
                                               vocab_size, softcap);
                    auto grads = torch::autograd::grad({losses}, {x_leaf, w_leaf}, {row_grad.slice(0, begin, end)});
                    grad_x.slice(0, begin, end).copy_(grads[0]);
                    grad_w.add_(grads[1]);
                }
                return {grad_x, grad_w, {}, {}, {}, {}, {}};
            }
        };

    } // namespace

    torch::Tensor softcap_cross_entropy(const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &targets,
                                        int64_t vocab_size, double softcap, int64_t chunk_rows, LossReduction reduction) {
        if (x.dim() != 2 || targets.dim() != 1 || targets.size(0) != x.size(0))
            throw std::invalid_argument("softcap_cross_entropy expects x (N, C) and targets (N)");
        if (chunk_rows < 1)
            throw std::invalid_argument("chunk_rows must be positive");
        return SoftcapCrossEntropy::apply(x, weight, targets, vocab_size, softcap, chunk_rows,
                                          static_cast<int64_t>(reduction));
    }

} // namespace nanochat
