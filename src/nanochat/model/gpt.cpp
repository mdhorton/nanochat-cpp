#include "nanochat/model/gpt.h"

#include "nanochat/model/softcap_ce.h"

#include <cmath>
#include <set>
#include <stdexcept>

namespace nanochat {

    namespace F = torch::nn::functional;
    using torch::indexing::None;
    using torch::indexing::Slice;

    GPTConfig config_from_depth(int64_t depth, int64_t vocab_size, int64_t aspect_ratio, int64_t head_dim,
                                int64_t max_seq_len, const std::string &window_pattern) {
        const int64_t base_dim = depth * aspect_ratio;
        const int64_t model_dim = (base_dim + head_dim - 1) / head_dim * head_dim;
        const int64_t num_heads = model_dim / head_dim;
        return {.sequence_len = max_seq_len, .vocab_size = vocab_size, .n_layer = depth, .n_head = num_heads,
                .n_kv_head = num_heads, .n_embd = model_dim, .window_pattern = window_pattern};
    }

    std::vector<int64_t> window_sizes(const GPTConfig &config) {
        if (config.window_pattern.empty())
            throw std::invalid_argument("empty window_pattern");
        const int64_t long_window = config.sequence_len;
        const int64_t short_window = (long_window / 4 + 127) / 128 * 128; // ceil to FA3 tile size
        std::vector<int64_t> windows;
        for (int64_t i = 0; i < config.n_layer; ++i) {
            const char c = static_cast<char>(std::toupper(config.window_pattern[i % config.window_pattern.size()]));
            if (c != 'S' && c != 'L')
                throw std::invalid_argument("invalid window_pattern: " + config.window_pattern + " (use S and L)");
            windows.push_back(c == 'L' ? long_window : short_window);
        }
        windows.back() = long_window; // final layer always gets full context
        return windows;
    }

    bool has_ve(int64_t layer_idx, int64_t n_layer) { return layer_idx % 2 == (n_layer - 1) % 2; }

    torch::Tensor rms_norm(const torch::Tensor &x) { return at::rms_norm(x, {x.size(-1)}); }

    // Rotates by -theta (the transpose of the textbook convention), as in Python, for checkpoint compatibility.
    // Ops on reused tensors are created in Python's left-to-right order: autograd accumulates their gradients in
    // creation order, and bf16 sums depend on it (C++ leaves the operand order of `a + b` unspecified).
    static torch::Tensor apply_rotary_emb(const torch::Tensor &x, const torch::Tensor &cos, const torch::Tensor &sin) {
        const int64_t d = x.size(3) / 2;
        auto x1 = x.index({"...", Slice(None, d)});
        auto x2 = x.index({"...", Slice(d, None)});
        auto a = x1 * cos;
        auto y1 = a + x2 * sin;
        auto b = x1 * (-sin);
        auto y2 = b + x2 * cos;
        return torch::cat({y1, y2}, 3);
    }

    // SDPA with a causal left window of `window` tokens (window + 1 keys including the query), as flash_attention.py.
    static torch::Tensor sdpa_attention(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                                        int64_t window) {
        const int64_t Tq = q.size(2), Tk = k.size(2);
        const bool enable_gqa = q.size(1) != k.size(1);
        if ((window < 0 || window >= Tq) && Tq == Tk)
            return at::scaled_dot_product_attention(q, k, v, {}, 0.0, /*is_causal=*/true, std::nullopt, enable_gqa);
        const auto opts = torch::TensorOptions().device(q.device()).dtype(torch::kInt64);
        auto row = (Tk - Tq) + torch::arange(Tq, opts).unsqueeze(1);
        auto col = torch::arange(Tk, opts).unsqueeze(0);
        auto mask = col <= row;
        if (window >= 0 && window < Tk)
            mask = mask & ((row - col) <= window);
        return at::scaled_dot_product_attention(q, k, v, mask, 0.0, false, std::nullopt, enable_gqa);
    }

    // FlashAttention-2 on (B, T, H, D) with the same window semantics: `window` keys to the left plus the query.
    static torch::Tensor fa2_attention(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                                       int64_t window) {
        const int64_t Tq = q.size(1), Tk = k.size(1);
        std::optional<int64_t> left, right;
        if (window >= 0 && window < Tk) {
            left = window;
            right = 0;
        }
        return std::get<0>(at::_flash_attention_forward(q, k, v, std::nullopt, std::nullopt, Tq, Tk, 0.0,
                                                        /*is_causal=*/true, false, std::nullopt, left, right));
    }

    Attention attention_from_string(const std::string &name) {
        if (name == "fa2")
            return Attention::FA2;
        if (name == "sdpa")
            return Attention::SDPA;
        throw std::invalid_argument("unknown attention: " + name + " (use fa2 or sdpa)");
    }

    // ---------------------------------------------------------------------------------------------------------------

    LinearImpl::LinearImpl(int64_t in_features, int64_t out_features, const torch::TensorOptions &options) {
        weight = register_parameter("weight", torch::empty({out_features, in_features}, options));
    }

    torch::Tensor LinearImpl::forward(const torch::Tensor &x) { return F::linear(x, weight.to(x.scalar_type())); }

    EmbeddingImpl::EmbeddingImpl(int64_t num_embeddings, int64_t dim, const torch::TensorOptions &options) {
        weight = register_parameter("weight", torch::empty({num_embeddings, dim}, options));
    }

    torch::Tensor EmbeddingImpl::forward(const torch::Tensor &idx) { return at::embedding(weight, idx); }

    CausalSelfAttentionImpl::CausalSelfAttentionImpl(const GPTConfig &config, int64_t layer_idx,
                                                     const torch::TensorOptions &options) :
        n_head(config.n_head), n_kv_head(config.n_kv_head), head_dim(config.head_dim()) {
        if (config.n_embd % n_head != 0 || n_head % n_kv_head != 0)
            throw std::invalid_argument("n_embd must divide by n_head and n_head by n_kv_head");
        c_q = register_module("c_q", Linear(config.n_embd, n_head * head_dim, options));
        c_k = register_module("c_k", Linear(config.n_embd, n_kv_head * head_dim, options));
        c_v = register_module("c_v", Linear(config.n_embd, n_kv_head * head_dim, options));
        c_proj = register_module("c_proj", Linear(config.n_embd, config.n_embd, options));
        if (has_ve(layer_idx, config.n_layer))
            ve_gate = register_module("ve_gate", Linear(kVeGateChannels, n_kv_head, options));
    }

    torch::Tensor CausalSelfAttentionImpl::forward(const torch::Tensor &x, const torch::Tensor &ve,
                                                   const torch::Tensor &cos, const torch::Tensor &sin, int64_t window) {
        const int64_t B = x.size(0), T = x.size(1);
        // (B, T, H, D)
        auto q = c_q(x).view({B, T, n_head, head_dim});
        auto k = c_k(x).view({B, T, n_kv_head, head_dim});
        auto v = c_v(x).view({B, T, n_kv_head, head_dim});

        // value residual: mix in the value embedding with an input-dependent gate per head, range (0, 3)
        if (ve.defined()) {
            auto gate = 3 * torch::sigmoid(ve_gate(x.index({"...", Slice(None, kVeGateChannels)})));
            v = v + gate.unsqueeze(-1) * ve.view({B, T, n_kv_head, head_dim});
        }

        q = apply_rotary_emb(q, cos, sin);
        k = apply_rotary_emb(k, cos, sin);
        q = rms_norm(q) * 1.2; // QK norm, sharper attention split between q and k
        k = rms_norm(k) * 1.2;

        auto y = attention == Attention::FA2
                         ? fa2_attention(q, k, v, window)
                         : sdpa_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), window).transpose(1, 2);
        return c_proj(y.contiguous().view({B, T, -1}));
    }

    MLPImpl::MLPImpl(const GPTConfig &config, const torch::TensorOptions &options) {
        c_fc = register_module("c_fc", Linear(config.n_embd, 4 * config.n_embd, options));
        c_proj = register_module("c_proj", Linear(4 * config.n_embd, config.n_embd, options));
    }

    torch::Tensor MLPImpl::forward(const torch::Tensor &x) { return c_proj(torch::relu(c_fc(x)).square()); }

    BlockImpl::BlockImpl(const GPTConfig &config, int64_t layer_idx, const torch::TensorOptions &options) {
        attn = register_module("attn", CausalSelfAttention(config, layer_idx, options));
        mlp = register_module("mlp", MLP(config, options));
    }

    torch::Tensor BlockImpl::forward(const torch::Tensor &x, const torch::Tensor &ve, const torch::Tensor &cos,
                                     const torch::Tensor &sin, int64_t window) {
        auto y = x + attn(rms_norm(x), ve, cos, sin, window);
        return y + mlp(rms_norm(y));
    }

    TransformerImpl::TransformerImpl(const GPTConfig &config, int64_t padded_vocab, const torch::TensorOptions &options) {
        wte = register_module("wte", Embedding(padded_vocab, config.n_embd, options));
        h = register_module("h", torch::nn::ModuleList());
        for (int64_t i = 0; i < config.n_layer; ++i)
            h->push_back(Block(config, i, options));
    }

    // ---------------------------------------------------------------------------------------------------------------

    GPTImpl::GPTImpl(GPTConfig config, torch::Device device, int64_t pad_vocab_size_to) :
        config_(std::move(config)), windows_(window_sizes(config_)) {
        const auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
        const int64_t padded_vocab = (config_.vocab_size + pad_vocab_size_to - 1) / pad_vocab_size_to * pad_vocab_size_to;
        // Registration order follows Python, so parameters() (and optimizer groups) come out in the same order.
        transformer = register_module("transformer", Transformer(config_, padded_vocab, options));
        lm_head = register_module("lm_head", Linear(config_.n_embd, padded_vocab, options));
        resid_lambdas = register_parameter("resid_lambdas", torch::empty({config_.n_layer}, options));
        x0_lambdas = register_parameter("x0_lambdas", torch::empty({config_.n_layer}, options));
        smear_gate = register_module("smear_gate", Linear(24, 1, options));
        smear_lambda = register_parameter("smear_lambda", torch::empty({1}, options));
        backout_lambda = register_parameter("backout_lambda", torch::empty({1}, options));
        const int64_t kv_dim = config_.n_kv_head * config_.head_dim();
        torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> ves;
        for (int64_t i = 0; i < config_.n_layer; ++i)
            if (has_ve(i, config_.n_layer))
                ves.insert(std::to_string(i), Embedding(padded_vocab, kv_dim, options).ptr());
        value_embeds = register_module("value_embeds", torch::nn::ModuleDict(ves));
        precompute_rotary(config_.sequence_len * 10); // 10x over-compute, as Python
    }

    void GPTImpl::precompute_rotary(int64_t seq_len) {
        const int64_t head_dim = config_.head_dim();
        const auto opts = torch::TensorOptions().device(lm_head->weight.device()).dtype(torch::kFloat32);
        auto channel_range = torch::arange(0, head_dim, 2, opts);
        auto inv_freq = torch::pow(100000.0, channel_range / head_dim).reciprocal(); // Python: 1.0 / (base ** ...)
        auto t = torch::arange(seq_len, opts);
        auto freqs = torch::outer(t, inv_freq);
        cos_ = freqs.cos().to(kComputeDtype).unsqueeze(0).unsqueeze(2);
        sin_ = freqs.sin().to(kComputeDtype).unsqueeze(0).unsqueeze(2);
    }

    void GPTImpl::init_weights() {
        torch::NoGradGuard no_grad;
        transformer->wte->weight.normal_(0.0, 0.8);
        lm_head->weight.normal_(0.0, 0.001);

        // uniform with bound sqrt(3) * std gives the same std as normal, without outliers
        const double s = std::pow(3.0, 0.5) * std::pow(static_cast<double>(config_.n_embd), -0.5); // Python ** semantics
        for (const auto &m: *transformer->h) {
            const auto &block = m->as<BlockImpl>();
            block->attn->c_q->weight.uniform_(-s, s);
            block->attn->c_k->weight.uniform_(-s, s);
            block->attn->c_v->weight.uniform_(-s, s);
            block->attn->c_proj->weight.zero_();
            block->mlp->c_fc->weight.uniform_(-s * 0.4, s * 0.4);
            block->mlp->c_proj->weight.zero_();
        }

        // stronger residual and more x0 blending at early layers
        const int64_t n_layer = config_.n_layer;
        for (int64_t i = 0; i < n_layer; ++i)
            resid_lambdas[i].fill_(1.15 - 0.10 * static_cast<double>(i) / static_cast<double>(std::max<int64_t>(n_layer - 1, 1)));
        for (int64_t i = 0; i < n_layer; ++i)
            x0_lambdas[i].fill_(0.20 - 0.15 * static_cast<double>(i) / static_cast<double>(std::max<int64_t>(n_layer - 1, 1)));

        smear_lambda.zero_();
        backout_lambda.fill_(0.2);
        smear_gate->weight.uniform_(0.0, 0.02);

        for (const auto &item: value_embeds->items())
            item.second->as<EmbeddingImpl>()->weight.uniform_(-s, s);
        for (const auto &m: *transformer->h) {
            const auto &attn = m->as<BlockImpl>()->attn;
            if (attn->ve_gate)
                attn->ve_gate->weight.uniform_(0.0, 0.02);
        }

        precompute_rotary(config_.sequence_len * 10);

        // embeddings are stored in the compute dtype
        transformer->wte->weight.set_data(transformer->wte->weight.to(kComputeDtype));
        for (const auto &item: value_embeds->items()) {
            auto &w = item.second->as<EmbeddingImpl>()->weight;
            w.set_data(w.to(kComputeDtype));
        }
    }

    torch::Tensor GPTImpl::forward(const torch::Tensor &idx, const torch::Tensor &targets,
                                   F::CrossEntropyFuncOptions::reduction_t reduction) {
        const int64_t T = idx.size(1);
        if (T > cos_.size(1))
            throw std::invalid_argument("sequence length grew beyond the rotary embeddings cache");
        if (T < 2)
            throw std::invalid_argument("training forward needs T > 1");
        auto cos = cos_.index({Slice(), Slice(None, T)}), sin = sin_.index({Slice(), Slice(None, T)});

        auto x = rms_norm(transformer->wte(idx).to(kComputeDtype));

        // smear: mix the previous token's embedding into the current position (cheap bigram info)
        // (statement order = Python's evaluation order, see apply_rotary_emb)
        auto lambda = smear_lambda.to(x.scalar_type());
        auto gate = lambda * torch::sigmoid(smear_gate(x.index({Slice(), Slice(1, None), Slice(None, 24)})));
        auto first = x.index({Slice(), Slice(None, 1)});
        auto rest = x.index({Slice(), Slice(1, None)});
        auto prev = gate * x.index({Slice(), Slice(None, -1)});
        x = torch::cat({first, rest + prev}, 1);

        const auto x0 = x; // initial normalized embedding, blended back in at every layer
        const int64_t backout_layer = config_.n_layer / 2;
        torch::Tensor x_backout;
        for (int64_t i = 0; i < config_.n_layer; ++i) {
            auto scaled = resid_lambdas[i] * x;
            x = scaled + x0_lambdas[i] * x0;
            torch::Tensor ve;
            if (const auto key = std::to_string(i); value_embeds->contains(key))
                ve = value_embeds[key]->as<EmbeddingImpl>()->forward(idx).to(x.scalar_type());
            x = transformer->h[i]->as<BlockImpl>()->forward(x, ve, cos, sin, windows_[i]);
            if (i == backout_layer)
                x_backout = x;
        }
        // subtract the mid-layer residual to remove low-level features before the logits
        if (x_backout.defined()) {
            auto lambda_b = backout_lambda.to(x.scalar_type());
            x = x - lambda_b * x_backout;
        }
        x = rms_norm(x);

        const double softcap = 15;
        if (targets.defined() && loss_chunk_rows_ > 0) {
            const auto r = std::holds_alternative<torch::enumtype::kMean>(reduction) ? LossReduction::Mean
                           : std::holds_alternative<torch::enumtype::kSum>(reduction) ? LossReduction::Sum
                                                                                     : LossReduction::None;
            return softcap_cross_entropy(x.view({-1, x.size(-1)}), lm_head->weight, targets.view(-1),
                                         config_.vocab_size, softcap, loss_chunk_rows_, r);
        }
        auto logits = lm_head(x).index({"...", Slice(None, config_.vocab_size)}).to(torch::kFloat32);
        logits = softcap * torch::tanh(logits / softcap);
        if (!targets.defined())
            return logits;
        return F::cross_entropy(logits.view({-1, logits.size(-1)}), targets.view(-1),
                                F::CrossEntropyFuncOptions().ignore_index(-1).reduction(reduction));
    }

    void GPTImpl::set_attention(Attention attention) {
        for (const auto &m: *transformer->h)
            m->as<BlockImpl>()->attn->attention = attention;
    }

    safetensors::TensorMap GPTImpl::state_dict() const {
        safetensors::TensorMap state;
        for (const auto &item: named_parameters(true))
            state[item.key()] = item.value().detach();
        return state;
    }

    void GPTImpl::load_state(const safetensors::TensorMap &state) {
        torch::NoGradGuard no_grad;
        auto params = named_parameters(true);
        if (params.size() != state.size())
            throw std::runtime_error("state has " + std::to_string(state.size()) + " tensors, model has " +
                                     std::to_string(params.size()));
        for (auto &item: params) {
            const auto it = state.find(item.key());
            if (it == state.end())
                throw std::runtime_error("missing tensor in state: " + item.key());
            if (it->second.sizes() != item.value().sizes())
                throw std::runtime_error("shape mismatch for " + item.key());
            if (it->second.scalar_type() == item.value().scalar_type())
                item.value().copy_(it->second);
            else // e.g. bf16 embeddings into a model that wasn't initialized
                item.value().set_data(it->second.to(item.value().device()).clone());
        }
    }

    int64_t GPTImpl::num_matmul_params() const {
        int64_t n = 0;
        for (const auto &m: modules())
            if (const auto *linear = dynamic_cast<const LinearImpl *>(m.get()))
                n += linear->weight.numel();
        return n;
    }

    int64_t GPTImpl::estimate_flops() const {
        const int64_t h = config_.n_head, q = config_.head_dim(), t = config_.sequence_len;
        int64_t attn_flops = 0;
        for (const int64_t window: windows_)
            attn_flops += 12 * h * q * (window < 0 ? t : std::min(window, t));
        return 6 * num_matmul_params() + attn_flops;
    }

    ScalingParams GPTImpl::num_scaling_params() const {
        auto count = [](const std::vector<torch::Tensor> &params) {
            int64_t n = 0;
            for (const auto &p: params)
                n += p.numel();
            return n;
        };
        ScalingParams s;
        s.wte = transformer->wte->weight.numel();
        s.value_embeds = count(value_embeds->parameters());
        s.lm_head = lm_head->weight.numel();
        s.transformer_matrices = count(transformer->h->parameters());
        s.scalars = resid_lambdas.numel() + x0_lambdas.numel() + smear_gate->weight.numel() + smear_lambda.numel() +
                    backout_lambda.numel();
        s.total = s.wte + s.value_embeds + s.lm_head + s.transformer_matrices + s.scalars;
        if (s.total != count(parameters()))
            throw std::logic_error("parameter count mismatch");
        return s;
    }

} // namespace nanochat
