// GPT model. Port of nanochat/gpt.py (training forward; KV-cache inference comes in phase 4).
// Parameter names match the Python state_dict, so safetensors exports of Python weights load directly.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "nanochat/model/fp8.h"
#include "nanochat/train/safetensors.h"

namespace nanochat {

struct MxFlashInputs;

// Activations and matmuls run in bf16; master weights stay fp32 except the embeddings.
inline constexpr auto kComputeDtype = torch::kBFloat16;

// FA2: PyTorch's built-in FlashAttention-2, with native sliding windows (fast).
// SDPA: nanochat's fallback (explicit mask for sliding windows); bit-identical to Python nanochat on sm_120.
// MX: MXFP8 flash-attention forward (mx_flash.h), FA2's bf16 backward.
enum class Attention { FA2, SDPA, MX };
Attention attention_from_string(const std::string& name); // "fa2", "sdpa" or "mx"

struct GPTConfig {
  int64_t sequence_len = 2048;
  int64_t vocab_size = 32768;
  int64_t n_layer = 12;
  int64_t n_head = 6;    // query heads
  int64_t n_kv_head = 6; // key/value heads (GQA)
  int64_t n_embd = 768;
  // Tiled across layers, last layer always L. L = full context, S = quarter context.
  std::string window_pattern = "SSSL";

  int64_t head_dim() const {
    return n_embd / n_head;
  }
};

// base_train.py build_model_meta: width = depth * aspect_ratio rounded up to a multiple of head_dim.
GPTConfig config_from_depth(
      int64_t depth, int64_t vocab_size, int64_t aspect_ratio = 64, int64_t head_dim = 128, int64_t max_seq_len = 2048,
      const std::string& window_pattern = "SSSL");

// Left window per layer: sequence_len (full) or the short window, rounded up to 128.
std::vector<int64_t> window_sizes(const GPTConfig& config);

// Value embeddings on alternating layers, last layer always included.
bool has_ve(int64_t layer_idx, int64_t n_layer);

// Parameter-free RMSNorm over the last dim, default eps.
torch::Tensor rms_norm(const torch::Tensor& x);

// Linear without bias whose weight is cast to the input dtype in forward (replaces autocast). With fp8 set, the
// matmuls run in FP8 (fp8.h), as Python's Float8Linear.
class LinearImpl : public torch::nn::Module {
public:
  LinearImpl(int64_t in_features, int64_t out_features, const torch::TensorOptions& options);
  torch::Tensor forward(const torch::Tensor& x);
  torch::Tensor weight;
  bool fp8 = false;
  Fp8Recipe fp8_recipe = Fp8Recipe::Tensorwise;
  Fp8WeightCache fp8_cache; // enabled by GPT's set_fused
};

TORCH_MODULE(Linear);

class EmbeddingImpl : public torch::nn::Module {
public:
  EmbeddingImpl(int64_t num_embeddings, int64_t dim, const torch::TensorOptions& options);
  torch::Tensor forward(const torch::Tensor& idx);
  torch::Tensor weight;
};

TORCH_MODULE(Embedding);

class CausalSelfAttentionImpl : public torch::nn::Module {
public:
  CausalSelfAttentionImpl(const GPTConfig& config, int64_t layer_idx, const torch::TensorOptions& options);
  // ve: value embedding (B, T, n_kv_head * head_dim) or undefined; cos/sin: (1, T, 1, head_dim / 2). x_mx (with
  // mx_inputs): x already quantized (residual_norm_mx).
  torch::Tensor forward(
        const torch::Tensor& x, const torch::Tensor& ve, const torch::Tensor& cos, const torch::Tensor& sin,
        int64_t window, const Fp8Tensor* x_mx = nullptr);
  // Whether an (N, C) input goes through mx_attention_inputs.
  bool mx_inputs(int64_t N, int64_t C) const;

  static constexpr int64_t kVeGateChannels = 12;
  int64_t n_head, n_kv_head, head_dim;
  Attention attention = Attention::FA2;
  // merged q/k/v (fp8.h's fp8_qkv), rotary + QK norm in one kernel (rotary_norm.h); MXFP8: mx_attention.h
  bool fused = false;
  Linear c_q{nullptr}, c_k{nullptr}, c_v{nullptr}, c_proj{nullptr}, ve_gate{nullptr};
  Fp8WeightCache qkv_cache; // merged q/k/v weights

private:
  // attention over (B, T, H, D) q, k, v
  // pre: q, k, v already MX-quantized for Attention::MX (mx_attention_inputs)
  torch::Tensor attend(
        const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int64_t window,
        const MxFlashInputs* pre = nullptr) const;
};

TORCH_MODULE(CausalSelfAttention);

class MLPImpl : public torch::nn::Module {
public:
  MLPImpl(const GPTConfig& config, const torch::TensorOptions& options);
  // x_mx (with mx_inputs): x already quantized (residual_norm_mx)
  torch::Tensor forward(const torch::Tensor& x, const Fp8Tensor* x_mx = nullptr);
  // Whether an (N, C) input goes through the Mx fp8_relu_square_mlp.
  bool mx_inputs(int64_t N, int64_t C) const;
  bool fused = false; // relu^2 in one kernel; with FP8, also folded into the quantization (relu_square.h, fp8.h)
  Linear c_fc{nullptr}, c_proj{nullptr};
};

TORCH_MODULE(MLP);

class BlockImpl : public torch::nn::Module {
public:
  BlockImpl(const GPTConfig& config, int64_t layer_idx, const torch::TensorOptions& options);
  torch::Tensor forward(
        const torch::Tensor& x, const torch::Tensor& ve, const torch::Tensor& cos, const torch::Tensor& sin,
        int64_t window);
  // Fused path (residual_norm.h): from x and x_norm = rms_norm(x), returns {y, m}, the block output y + m unsummed
  // so the next residual_norm adds it. x_norm_mx: x_norm's quantization when attn->mx_inputs (residual_norm_mx).
  std::pair<torch::Tensor, torch::Tensor> forward_split(
        const torch::Tensor& x, const torch::Tensor& x_norm, const torch::Tensor& ve, const torch::Tensor& cos,
        const torch::Tensor& sin, int64_t window, const Fp8Tensor* x_norm_mx = nullptr);
  CausalSelfAttention attn{nullptr};
  MLP mlp{nullptr};
};

TORCH_MODULE(Block);

// Holds wte and h, so parameter names get the "transformer." prefix.
class TransformerImpl : public torch::nn::Module {
public:
  TransformerImpl(const GPTConfig& config, int64_t padded_vocab, const torch::TensorOptions& options);
  Embedding wte{nullptr};
  torch::nn::ModuleList h{nullptr};
};

TORCH_MODULE(Transformer);

struct ScalingParams {
  int64_t wte = 0, value_embeds = 0, lm_head = 0, transformer_matrices = 0, scalars = 0, total = 0;
};

class GPTImpl : public torch::nn::Module {
public:
  // Parameters are allocated uninitialized on device; call init_weights() or load_state().
  explicit GPTImpl(GPTConfig config, torch::Device device = torch::kCUDA, int64_t pad_vocab_size_to = 64);

  // Uses the global RNG in the same order as Python, so equal seeds give identical weights.
  void init_weights();

  // idx: (B, T) int64. Returns fp32 logits (B, T, vocab_size) when targets is undefined, else the loss.
  torch::Tensor forward(
        const torch::Tensor& idx, const torch::Tensor& targets = {},
        torch::nn::functional::CrossEntropyFuncOptions::reduction_t reduction = torch::kMean);

  safetensors::TensorMap state_dict() const;
  // Copies tensors into parameters; names and shapes must match exactly, dtypes come from the state.
  void load_state(const safetensors::TensorMap& state);

  int64_t num_matmul_params() const;
  int64_t estimate_flops() const; // per token, forward + backward
  ScalingParams num_scaling_params() const;

  const GPTConfig& config() const {
    return config_;
  }

  const std::vector<int64_t>& windows() const {
    return windows_;
  }

  void set_attention(Attention attention);

  // Fused CUDA kernels for elementwise chains (rotary + QK norm, resid/x0 blend) and FP8 weights cached between
  // optimizer steps. false = Python's op-by-op path.
  void set_fused(bool fused);

  // > 0: compute the training loss a chunk of rows at a time (softcap_ce.h), never materializing all logits.
  // 0: unchunked, as Python.
  void set_loss_chunk_rows(int64_t rows) {
    loss_chunk_rows_ = rows;
  }

  // FP8 matmuls for the Linears that qualify (fp8_eligible), as convert_to_float8_training; false = bf16 (eval).
  // Returns the number of Linears switched.
  int set_fp8(bool enabled);
  void set_fp8_recipe(Fp8Recipe recipe);
  int num_linears();

  Transformer transformer{nullptr};
  Linear lm_head{nullptr};
  torch::Tensor resid_lambdas, x0_lambdas, smear_lambda, backout_lambda;
  Linear smear_gate{nullptr};
  torch::nn::ModuleDict value_embeds{nullptr};

private:
  void precompute_rotary(int64_t seq_len);

  GPTConfig config_;
  std::vector<int64_t> windows_;
  int64_t loss_chunk_rows_ = 0;
  bool fused_ = false;
  torch::Tensor cos_, sin_; // (1, 10 * sequence_len, 1, head_dim / 2), not saved
};

TORCH_MODULE(GPT);

} // namespace nanochat
