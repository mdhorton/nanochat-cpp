#include "nanochat/model/flash_attention.h"

#include <cmath>

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/model/flash_attention_kernel.h"

namespace nanochat {

namespace {

namespace fl = FLASH_NAMESPACE;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

constexpr int kHeadDim = 128;

int64_t round_up(int64_t x, int64_t m) {
  return (x + m - 1) / m * m;
}

// zeros: the backward kernel reads them even without dropout
uint64_t* rng_state() {
  static const auto* t = new torch::Tensor(
        torch::zeros({2}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64)));
  return reinterpret_cast<uint64_t*>(t->data_ptr());
}

// As flash_api.cpp's set_params_fprop (no dropout, alibi, softcap or varlen); window < 0: causal, else local with
// `window` keys to the left.
void set_fwd(
      fl::Flash_fwd_params& p, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
      const torch::Tensor& out, const torch::Tensor& lse, int64_t window) {
  p.is_bf16 = true;
  p.q_ptr = q.data_ptr();
  p.k_ptr = k.data_ptr();
  p.v_ptr = v.data_ptr();
  p.o_ptr = out.data_ptr();
  p.q_batch_stride = q.stride(0), p.q_row_stride = q.stride(1), p.q_head_stride = q.stride(2);
  p.k_batch_stride = k.stride(0), p.k_row_stride = k.stride(1), p.k_head_stride = k.stride(2);
  p.v_batch_stride = v.stride(0), p.v_row_stride = v.stride(1), p.v_head_stride = v.stride(2);
  p.o_batch_stride = out.stride(0), p.o_row_stride = out.stride(1), p.o_head_stride = out.stride(2);
  p.softmax_lse_ptr = lse.data_ptr();
  p.b = static_cast<int>(q.size(0));
  p.seqlen_q = static_cast<int>(q.size(1));
  p.seqlen_k = static_cast<int>(k.size(1));
  p.seqlen_q_rounded = static_cast<int>(round_up(p.seqlen_q, 128));
  p.seqlen_k_rounded = static_cast<int>(round_up(p.seqlen_k, 128));
  p.h = p.h_k = static_cast<int>(q.size(2));
  p.h_h_k_ratio = 1;
  p.d = p.d_rounded = kHeadDim;
  const auto scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim))); // as PyTorch's default
  p.scale_softmax = scale;
  p.scale_softmax_log2 = static_cast<float>(scale * M_LOG2E);
  p.p_dropout = 1.f; // keep probability
  p.p_dropout_in_uint8_t = 255;
  p.rp_dropout = 1.f;
  p.scale_softmax_rp_dropout = scale;
  p.is_causal = window < 0 || window >= p.seqlen_k;
  p.window_size_left = p.is_causal ? p.seqlen_k : static_cast<int>(window);
  p.window_size_right = 0;
  p.is_seqlens_k_cumulative = true;
  p.rng_state = rng_state();
}

class FlashAttention : public torch::autograd::Function<FlashAttention> {
public:
  static torch::Tensor forward(
        AutogradContext* ctx, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window) {
    TORCH_CHECK(
          flash_attention_fits(q, k, v),
          "flash_attention: (B, T, H, 128) bf16 q, k, v with contiguous last dims, the same B and H");
    auto out = torch::empty(q.sizes(), q.options());
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(torch::kFloat32));
    fl::Flash_fwd_params p{};
    set_fwd(p, q, k, v, out, lse, window);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernels::flash_fwd(p, stream);
    ctx->save_for_backward({q, k, v, out, lse});
    ctx->saved_data["window"] = window;
    return out;
  }

  // As flash_api.cpp's mha_bwd (nondeterministic: dq accumulated with atomics)
  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    const auto s = ctx->get_saved_variables();
    const auto &q = s[0], &k = s[1], &v = s[2], &out = s[3], &lse = s[4];
    const auto dout = grads[0].stride(-1) == 1 ? grads[0] : grads[0].contiguous();
    auto dq = torch::empty(q.sizes(), q.options()), dk = torch::empty(k.sizes(), k.options());
    auto dv = torch::empty(v.sizes(), v.options());
    const int64_t B = q.size(0), H = q.size(2), Tq_rounded = round_up(q.size(1), 128);
    const auto f32 = q.options().dtype(torch::kFloat32);
    auto softmax_d = torch::empty({B, H, Tq_rounded}, f32), dq_accum = torch::empty({B, Tq_rounded, H, kHeadDim}, f32);
    fl::Flash_bwd_params p{};
    set_fwd(p, q, k, v, out, lse, ctx->saved_data["window"].toInt());
    p.do_ptr = dout.data_ptr();
    p.dq_ptr = dq.data_ptr();
    p.dk_ptr = dk.data_ptr();
    p.dv_ptr = dv.data_ptr();
    p.do_batch_stride = dout.stride(0), p.do_row_stride = dout.stride(1), p.do_head_stride = dout.stride(2);
    p.dq_batch_stride = dq.stride(0), p.dq_row_stride = dq.stride(1), p.dq_head_stride = dq.stride(2);
    p.dk_batch_stride = dk.stride(0), p.dk_row_stride = dk.stride(1), p.dk_head_stride = dk.stride(2);
    p.dv_batch_stride = dv.stride(0), p.dv_row_stride = dv.stride(1), p.dv_head_stride = dv.stride(2);
    p.dq_accum_ptr = dq_accum.data_ptr();
    p.dsoftmax_sum = softmax_d.data_ptr();
    p.deterministic = false;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernels::flash_bwd(p, stream);
    return {dq, dk, dv, torch::Tensor()};
  }
};

} // namespace

bool flash_attention_fits(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  const auto bf16 = [](const torch::Tensor& t) {
    return t.is_cuda() && t.dim() == 4 && t.scalar_type() == torch::kBFloat16 && t.stride(-1) == 1;
  };
  return bf16(q) && bf16(k) && bf16(v) && q.size(3) == kHeadDim && k.sizes() == v.sizes() && q.size(0) == k.size(0) &&
         q.size(2) == k.size(2) && q.size(3) == k.size(3);
}

torch::Tensor flash_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window) {
  return FlashAttention::apply(q, k, v, window);
}

} // namespace nanochat
