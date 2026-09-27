#include "nanochat/model/softcap_ce.h"

#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/fp8.h"
#include "nanochat/model/softcap_ce_kernel.h"

namespace nanochat {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

// Per chunk of rows: logits via cuBLAS, then the fused kernel (loss, and the logits' gradient in place), then
// grad_x = dlogits @ w and grad_w += dlogits^T @ x (fp32 accumulate). Undefined outputs are skipped.
// bf16: w cast to x's dtype. fp8 (as fp8.py's Float8Matmul): x and weight quantized once (e4m3); each chunk's
// gradient is quantized (e5m2) with its own scale, from the amax the kernel reports (python: one scale for all rows).
// mx: MXFP8 instead (x's transpose quantized per chunk, its scales run along the rows). cache (optional): the
// weight's FP8 copy.
void run_chunks(
      const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& targets, int64_t vocab, double softcap,
      int64_t chunk_rows, bool fp8, bool mx, const torch::Tensor& losses, const torch::Tensor& grad_x,
      const torch::Tensor& grad_w, const torch::Tensor& grad_scale, const torch::Tensor& num_valid,
      Fp8WeightCache* cache) {
  const int64_t N = x.size(0), padded = weight.size(0);
  const bool grad = grad_x.defined();
  Fp8Tensor xq, wq;
  torch::Tensor w;
  if (fp8) {
    xq = mx ? quantize_mx(x, true, false) : quantize_fp8(x, torch::kFloat8_e4m3fn);
    wq = quantize_fp8_weight(weight, cache, mx ? Fp8Recipe::Mx : Fp8Recipe::Tensorwise);
  }
  else
    w = weight.to(x.scalar_type());
  auto logits = torch::empty({std::min(N, chunk_rows), padded}, x.options());
  const auto scalars = torch::empty({2}, x.options().dtype(torch::kFloat32)); // gradient amax, inverse scale
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  for (int64_t begin = 0; begin < N; begin += chunk_rows) {
    const int64_t end = std::min(N, begin + chunk_rows);
    const auto xc = x.slice(0, begin, end);
    auto buf = logits.slice(0, 0, end - begin);
    if (fp8) {                               // tensorwise: fast accumulation in forward only
      const int64_t blocks = x.size(1) / 32; // MX scales of a row, rows in whole tiles (begin % 128 == 0)
      at::_scaled_mm_out(
            buf, xq.data.slice(0, begin, end), wq.data.t(),
            mx ? xq.inv_scale.slice(0, begin * blocks, end * blocks) : xq.inv_scale, wq.inv_scale, {}, {},
            x.scalar_type(), !mx);
    }
    else
      at::mm_out(buf, xc, w.t());
    const int64_t scale_stride = grad_scale.defined() && grad_scale.dim() == 1 ? 1 : 0;
    kernels::softcap_ce(
          buf.data_ptr(), end - begin, padded, targets.data_ptr<int64_t>() + begin, static_cast<int>(vocab),
          static_cast<int>(padded), static_cast<float>(softcap),
          losses.defined() ? losses.data_ptr<float>() + begin : nullptr,
          grad_scale.defined() ? grad_scale.data_ptr<float>() + begin * scale_stride : nullptr, scale_stride,
          num_valid.defined() ? num_valid.data_ptr<int64_t>() : nullptr, grad,
          fp8 && !mx && grad ? scalars.data_ptr<float>() : nullptr, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!grad)
      continue;
    auto gx = grad_x.slice(0, begin, end);
    if (!fp8) {
      at::mm_out(gx, buf, w);
      at::addmm_out(const_cast<torch::Tensor&>(grad_w), grad_w, buf.t(), xc, torch::kFloat32);
      continue;
    }
    const auto gq = mx ? quantize_mx(buf) : quantize_fp8_amax_ready(buf, torch::kFloat8_e5m2, scalars);
    at::_scaled_mm_out(gx, gq.data, wq.data_t.t(), gq.inv_scale, wq.inv_t(), {}, {}, x.scalar_type(), false);
    // x_t's columns of this chunk, column-major
    const auto xt = mx ? quantize_mx(xc, false, true) : Fp8Tensor{};
    const auto xc_t = mx ? xt.data_t.t() : xq.data_t.slice(1, begin, end).t();
    const auto& xc_inv = mx ? xt.inv_scale_t : xq.inv_scale;
    if (begin == 0)
      at::_scaled_mm_out(
            const_cast<torch::Tensor&>(grad_w), gq.data_t, xc_t, gq.inv_t(), xc_inv, {}, {}, torch::kFloat32, false);
    else
      const_cast<torch::Tensor&>(grad_w).add_(
            at::_scaled_mm(gq.data_t, xc_t, gq.inv_t(), xc_inv, {}, {}, torch::kFloat32, false));
  }
}

// For mean/sum the incoming gradient is a scalar, so forward computes the gradients at once (no recompute)
// and backward only scales them. For none, backward recomputes each chunk with the per-row gradient.
class SoftcapCrossEntropy : public torch::autograd::Function<SoftcapCrossEntropy> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& targets,
        int64_t vocab_size, double softcap, int64_t chunk_rows, int64_t reduction_int, bool needs_grad, bool fp8,
        bool mx, Fp8WeightCache* cache) {
    const auto reduction = static_cast<LossReduction>(reduction_int);
    const auto num_valid = (targets >= 0).sum();
    auto losses = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
    const bool precompute = needs_grad && reduction != LossReduction::None;
    torch::Tensor grad_x, grad_w;
    if (precompute) {
      grad_x = torch::empty_like(x);
      grad_w = torch::zeros(weight.sizes(), weight.options().dtype(torch::kFloat32));
    }
    run_chunks(
          x, weight, targets, vocab_size, softcap, chunk_rows, fp8, mx, losses, grad_x, grad_w, {},
          reduction == LossReduction::Mean ? num_valid : torch::Tensor(), cache);
    if (precompute)
      ctx->save_for_backward({grad_x, grad_w});
    else if (needs_grad)
      ctx->save_for_backward({x, weight, targets});
    ctx->saved_data["precomputed"] = precompute;
    ctx->saved_data["vocab_size"] = vocab_size;
    ctx->saved_data["softcap"] = softcap;
    ctx->saved_data["chunk_rows"] = chunk_rows;
    ctx->saved_data["fp8"] = fp8;
    ctx->saved_data["mx"] = mx;
    switch (reduction) {
      case LossReduction::Mean:
        return losses.sum() / num_valid; // ignored targets have loss 0
      case LossReduction::Sum:
        return losses.sum();
      default:
        return losses;
    }
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    if (saved.empty())
      throw std::logic_error("softcap_cross_entropy: backward without grad mode in forward");
    const auto& g = grad_outputs[0];
    if (ctx->saved_data["precomputed"].toBool())
      return {saved[0] * g, saved[1] * g.to(torch::kFloat32), {}, {}, {}, {}, {}, {}, {}, {}, {}};

    const auto &x = saved[0], &weight = saved[1], &targets = saved[2];
    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::zeros(weight.sizes(), weight.options().dtype(torch::kFloat32));
    run_chunks(
          x, weight, targets, ctx->saved_data["vocab_size"].toInt(), ctx->saved_data["softcap"].toDouble(),
          ctx->saved_data["chunk_rows"].toInt(), ctx->saved_data["fp8"].toBool(), ctx->saved_data["mx"].toBool(), {},
          grad_x, grad_w, g.to(torch::kFloat32).contiguous(), {}, nullptr);
    return {grad_x, grad_w.to(weight.scalar_type()), {}, {}, {}, {}, {}, {}, {}, {}, {}};
  }
};

} // namespace

torch::Tensor softcap_cross_entropy(
      const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& targets, int64_t vocab_size,
      double softcap, int64_t chunk_rows, LossReduction reduction, bool fp8, Fp8WeightCache* cache, Fp8Recipe recipe) {
  if (x.dim() != 2 || targets.dim() != 1 || targets.size(0) != x.size(0))
    throw std::invalid_argument("softcap_cross_entropy expects x (N, C) and targets (N)");
  if (chunk_rows < 1)
    throw std::invalid_argument("chunk_rows must be positive");
  if (!x.is_cuda() || x.scalar_type() != torch::kBFloat16)
    throw std::invalid_argument("softcap_cross_entropy expects bf16 CUDA activations");
  if (weight.size(0) % 8 != 0 || weight.size(0) < vocab_size)
    throw std::invalid_argument("softcap_cross_entropy expects the padded vocab to be a multiple of 8");
  const bool needs_grad = torch::GradMode::is_enabled() && (x.requires_grad() || weight.requires_grad());
  // _scaled_mm: every chunk's rows and the dims % 16
  fp8 = fp8 && x.size(0) % 16 == 0 && chunk_rows % 16 == 0 && x.size(1) % 16 == 0 && weight.size(0) % 16 == 0;
  // MX: whole scale tiles (128 rows)
  const bool mx = fp8 && recipe == Fp8Recipe::Mx && mx_fits(x.size(0), x.size(1)) && chunk_rows % 128 == 0 &&
                  weight.size(0) % 128 == 0;
  return SoftcapCrossEntropy::apply(
        x.contiguous(), weight, targets.to(torch::kInt64).contiguous(), vocab_size, softcap, chunk_rows,
        static_cast<int64_t>(reduction), needs_grad, fp8, mx, cache);
}

} // namespace nanochat
