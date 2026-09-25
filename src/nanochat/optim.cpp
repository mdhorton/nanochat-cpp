#include "nanochat/optim.h"

#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nanochat {

    namespace {

        // Python passes hyperparameters as 0-D fp32 CPU tensors; using the same keeps the fp32 rounding identical.
        torch::Tensor cpu_scalar(double v) { return torch::tensor(static_cast<float>(v), torch::kFloat32); }

        // Polar Express coefficients (num_iters=5, safety_factor=2e-2, cushion=2), https://arxiv.org/pdf/2505.16932
        constexpr std::array<std::array<double, 3>, 5> kPolarExpressCoeffs = {{
                {8.156554524902461, -22.48329292557795, 15.878769915207462},
                {4.042929935166739, -2.808917465908714, 0.5000178451051316},
                {3.8916678022926607, -2.772484153217685, 0.5060648178503393},
                {3.285753657755655, -2.3681294933425376, 0.46449024233003106},
                {2.3465413258596377, -1.7097828382687081, 0.42323551169305323},
        }};

        torch::Tensor frobenius(const torch::Tensor &x) { return at::linalg_vector_norm(x, 2, {-2, -1}, true); }

        // AdamW, as adamw_step_fused: math in fp32, written back to the (possibly bf16) param and state.
        void adamw_update(const torch::Tensor &p, const torch::Tensor &grad, const torch::Tensor &exp_avg,
                          const torch::Tensor &exp_avg_sq, const torch::Tensor &step_t, const torch::Tensor &lr_t,
                          const torch::Tensor &beta1_t, const torch::Tensor &beta2_t, const torch::Tensor &eps_t,
                          const torch::Tensor &wd_t) {
            auto p32 = p.to(torch::kFloat32);
            auto exp_avg32 = exp_avg.to(torch::kFloat32);
            auto exp_avg_sq32 = exp_avg_sq.to(torch::kFloat32);
            auto grad32 = grad.to(torch::kFloat32);
            p32.mul_(1 - lr_t * wd_t); // decoupled weight decay
            exp_avg32.lerp_(grad32, 1 - beta1_t);
            exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t);
            auto bias1 = 1 - beta1_t.pow(step_t);
            auto bias2 = 1 - beta2_t.pow(step_t);
            auto denom = (exp_avg_sq32 / bias2).sqrt() + eps_t;
            auto step_size = lr_t / bias1;
            p32.add_(exp_avg32 / denom, (-step_size).item());
            p.copy_(p32); // no-ops when already fp32 (to() returned the same tensor)
            exp_avg.copy_(exp_avg32);
            exp_avg_sq.copy_(exp_avg_sq32);
        }

        // Muon, as muon_step_fused: Nesterov momentum -> MuonEq -> Polar Express -> Muon+ -> NorMuon -> cautious update.
        void muon_update(torch::Tensor stacked_grads, const torch::Tensor &stacked_params,
                         const torch::Tensor &momentum_buffer, const torch::Tensor &second_momentum_buffer,
                         const torch::Tensor &momentum_t, const torch::Tensor &lr_t, const torch::Tensor &wd_t,
                         const torch::Tensor &beta2_t, int ns_steps, int64_t red_dim) {
            // Nesterov momentum
            auto momentum = momentum_t.to(stacked_grads.scalar_type());
            momentum_buffer.lerp_(stacked_grads, 1 - momentum);
            auto g = stacked_grads.lerp_(momentum_buffer, momentum);

            auto X = g.to(torch::kBFloat16);

            // MuonEq row equilibration: rescale each row to the mean row norm
            auto target = frobenius(X.to(torch::kFloat32)) / std::pow(static_cast<double>(X.size(-2)), 0.5);
            auto row_norm = at::linalg_vector_norm(X.to(torch::kFloat32), 2, {-1}, true).clamp_min(1e-6);
            X = X * (target / row_norm).to(X.scalar_type());

            // Polar Express orthogonalization
            X = X / (frobenius(X) * 1.01 + 1e-6);
            const bool tall = g.size(-2) > g.size(-1);
            for (int i = 0; i < ns_steps; ++i) {
                const auto [a, b, c] = kPolarExpressCoeffs[i];
                if (tall) {
                    auto A = X.mT().matmul(X);
                    auto B = b * A + c * A.matmul(A);
                    X = a * X + X.matmul(B);
                }
                else {
                    auto A = X.matmul(X.mT());
                    auto B = b * A + c * A.matmul(A);
                    X = a * X + B.matmul(X);
                }
            }
            g = X.to(stacked_params.scalar_type());

            // Muon+: snap the Frobenius norm to sqrt(min(m, n)); Python's float / tensor is reciprocal() * float
            const double target_norm = std::pow(static_cast<double>(std::min(g.size(-2), g.size(-1))), 0.5);
            auto current_norm = frobenius(g.to(torch::kFloat32)).clamp_min(1e-6);
            g = g * (current_norm.reciprocal() * target_norm).to(g.scalar_type());

            // NorMuon variance reduction
            auto beta2 = beta2_t.to(g.scalar_type());
            auto v_mean = g.to(torch::kFloat32).square().mean(red_dim, true);
            const int64_t red_dim_size = g.size(red_dim);
            auto v_norm_sq = v_mean.sum({-2, -1}, true) * red_dim_size;
            auto v_norm = v_norm_sq.sqrt();
            second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.scalar_type()), 1 - beta2);
            auto step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt();
            auto scaled_sq_sum = (v_mean * red_dim_size) * step_size.to(torch::kFloat32).square();
            auto v_norm_new = scaled_sq_sum.sum({-2, -1}, true).sqrt();
            auto final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10));
            g = g * final_scale.to(g.scalar_type());

            // cautious weight decay + update
            auto lr = lr_t.to(g.scalar_type());
            auto wd = wd_t.to(g.scalar_type());
            auto mask = (g * stacked_params) >= 0;
            auto update = lr * g;
            auto decay = ((lr * wd) * stacked_params) * mask;
            stacked_params.sub_(update + decay);
        }

    } // namespace

    MuonAdamW::MuonAdamW(std::vector<OptimGroup> groups) : groups_(std::move(groups)) {
        adamw_states_.resize(groups_.size());
        muon_states_.resize(groups_.size());
        for (size_t i = 0; i < groups_.size(); ++i) {
            const auto &group = groups_[i];
            if (group.params.empty())
                throw std::invalid_argument("empty optimizer group " + group.name);
            if (group.kind == OptimGroup::Kind::AdamW)
                adamw_states_[i].resize(group.params.size());
            else
                for (const auto &p: group.params)
                    if (p.dim() != 2 || p.sizes() != group.params[0].sizes())
                        throw std::invalid_argument("Muon group " + group.name + " needs 2D params of one shape");
        }
    }

    void MuonAdamW::step() {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < groups_.size(); ++i) {
            if (groups_[i].kind == OptimGroup::Kind::AdamW)
                step_adamw(groups_[i], adamw_states_[i]);
            else
                step_muon(groups_[i], muon_states_[i]);
        }
    }

    void MuonAdamW::step_adamw(const OptimGroup &group, std::vector<AdamWState> &states) {
        const auto lr_t = cpu_scalar(group.lr), beta1_t = cpu_scalar(group.beta1), beta2_t = cpu_scalar(group.beta2);
        const auto eps_t = cpu_scalar(group.eps), wd_t = cpu_scalar(group.weight_decay);
        for (size_t j = 0; j < group.params.size(); ++j) {
            const auto &p = group.params[j];
            auto &state = states[j];
            if (!state.exp_avg.defined()) {
                state.exp_avg = torch::zeros_like(p);
                state.exp_avg_sq = torch::zeros_like(p);
            }
            ++state.step;
            adamw_update(p, p.grad(), state.exp_avg, state.exp_avg_sq, cpu_scalar(static_cast<double>(state.step)),
                         lr_t, beta1_t, beta2_t, eps_t, wd_t);
        }
    }

    void MuonAdamW::step_muon(const OptimGroup &group, MuonState &state) {
        const auto &params = group.params;
        const int64_t m = params[0].size(0), n = params[0].size(1), k = static_cast<int64_t>(params.size());
        const auto options = params[0].options();
        if (!state.momentum_buffer.defined()) {
            state.momentum_buffer = torch::zeros({k, m, n}, options);
            state.second_momentum_buffer = m >= n ? torch::zeros({k, m, 1}, options) : torch::zeros({k, 1, n}, options);
        }
        const int64_t red_dim = m >= n ? -1 : -2;

        std::vector<torch::Tensor> grads;
        for (const auto &p: params)
            grads.push_back(p.grad());
        auto stacked_grads = torch::stack(grads);
        auto stacked_params = torch::stack(params);
        // tall matrices get a larger lr
        const double lr = group.lr * std::pow(std::max(1.0, static_cast<double>(m) / static_cast<double>(n)), 0.5);
        muon_update(stacked_grads, stacked_params, state.momentum_buffer, state.second_momentum_buffer,
                    cpu_scalar(group.momentum), cpu_scalar(lr), cpu_scalar(group.weight_decay),
                    cpu_scalar(group.beta2), group.ns_steps, red_dim);
        for (int64_t j = 0; j < k; ++j)
            params[j].copy_(stacked_params[j]);
    }

    safetensors::TensorMap MuonAdamW::state_dict(safetensors::Metadata &metadata) const {
        safetensors::TensorMap state;
        auto groups = nlohmann::json::array();
        int64_t index = 0;
        for (size_t i = 0; i < groups_.size(); ++i) {
            const auto &group = groups_[i];
            nlohmann::json g = {{"name", group.name}, {"lr", group.lr}, {"initial_lr", group.initial_lr},
                                {"weight_decay", group.weight_decay}, {"params", nlohmann::json::array()}};
            for (size_t j = 0; j < group.params.size(); ++j)
                g["params"].push_back(index + static_cast<int64_t>(j));
            const auto prefix = [](int64_t idx) { return "state." + std::to_string(idx) + "."; };
            if (group.kind == OptimGroup::Kind::AdamW) {
                g.update({{"kind", "adamw"}, {"betas", {group.beta1, group.beta2}}, {"eps", group.eps}});
                for (size_t j = 0; j < group.params.size(); ++j) {
                    const auto &s = adamw_states_[i][j];
                    if (!s.exp_avg.defined())
                        continue;
                    const auto p = prefix(index + static_cast<int64_t>(j));
                    state[p + "step"] = torch::tensor(s.step, torch::kInt64);
                    state[p + "exp_avg"] = s.exp_avg;
                    state[p + "exp_avg_sq"] = s.exp_avg_sq;
                }
            }
            else {
                g.update({{"kind", "muon"}, {"momentum", group.momentum}, {"ns_steps", group.ns_steps},
                          {"beta2", group.beta2}});
                if (muon_states_[i].momentum_buffer.defined()) {
                    state[prefix(index) + "momentum_buffer"] = muon_states_[i].momentum_buffer;
                    state[prefix(index) + "second_momentum_buffer"] = muon_states_[i].second_momentum_buffer;
                }
            }
            groups.push_back(g);
            index += static_cast<int64_t>(group.params.size());
        }
        metadata["param_groups"] = groups.dump();
        return state;
    }

    void MuonAdamW::load_state_dict(const safetensors::TensorMap &state, const safetensors::Metadata &metadata) {
        const auto groups = nlohmann::json::parse(metadata.at("param_groups"));
        if (groups.size() != groups_.size())
            throw std::runtime_error("optimizer state has a different number of groups");
        torch::NoGradGuard no_grad;
        int64_t index = 0;
        for (size_t i = 0; i < groups_.size(); ++i) {
            auto &group = groups_[i];
            const auto &g = groups[i];
            if (g["params"].size() != group.params.size() ||
                (g["kind"] == "muon") != (group.kind == OptimGroup::Kind::Muon))
                throw std::runtime_error("optimizer group " + std::to_string(i) + " doesn't match the model");
            group.lr = g["lr"];
            group.initial_lr = g["initial_lr"];
            group.weight_decay = g["weight_decay"];
            const auto prefix = [](int64_t idx) { return "state." + std::to_string(idx) + "."; };
            // state tensors are copied onto each param's device
            const auto get = [&](const std::string &key, const torch::Tensor &like) {
                return state.at(key).to(like.device()).clone();
            };
            if (group.kind == OptimGroup::Kind::AdamW) {
                group.beta1 = g["betas"][0];
                group.beta2 = g["betas"][1];
                group.eps = g["eps"];
                for (size_t j = 0; j < group.params.size(); ++j) {
                    const auto p = prefix(index + static_cast<int64_t>(j));
                    if (!state.contains(p + "exp_avg"))
                        continue;
                    auto &s = adamw_states_[i][j];
                    s.step = state.at(p + "step").item<int64_t>();
                    s.exp_avg = get(p + "exp_avg", group.params[j]);
                    s.exp_avg_sq = get(p + "exp_avg_sq", group.params[j]);
                }
            }
            else {
                group.momentum = g["momentum"];
                group.ns_steps = g["ns_steps"];
                group.beta2 = g["beta2"];
                if (state.contains(prefix(index) + "momentum_buffer")) {
                    muon_states_[i].momentum_buffer = get(prefix(index) + "momentum_buffer", group.params[0]);
                    muon_states_[i].second_momentum_buffer =
                            get(prefix(index) + "second_momentum_buffer", group.params[0]);
                }
            }
            index += static_cast<int64_t>(group.params.size());
        }
    }

    MuonAdamW setup_optimizer(GPTImpl &model, double unembedding_lr, double embedding_lr, double matrix_lr,
                              double weight_decay, double scalar_lr) {
        using Kind = OptimGroup::Kind;
        const double dmodel_lr_scale = std::pow(static_cast<double>(model.config().n_embd) / 768.0, -0.5);
        auto adamw = [](std::string name, std::vector<torch::Tensor> params, double lr, double beta1, double beta2,
                        double wd) {
            return OptimGroup{.kind = Kind::AdamW, .name = std::move(name), .params = std::move(params), .lr = lr,
                              .initial_lr = lr, .weight_decay = wd, .beta1 = beta1, .beta2 = beta2, .eps = 1e-10};
        };
        std::vector<OptimGroup> groups = {
                adamw("lm_head", model.lm_head->parameters(), unembedding_lr * dmodel_lr_scale, 0.8, 0.96, 0.01),
                adamw("wte", model.transformer->wte->parameters(), embedding_lr * dmodel_lr_scale, 0.8, 0.995, 0.001),
                adamw("value_embeds", model.value_embeds->parameters(), embedding_lr * dmodel_lr_scale * 0.5, 0.8,
                      0.995, 0.01),
                adamw("resid_lambdas", {model.resid_lambdas}, scalar_lr * 0.01, 0.8, 0.95, 0.05),
                adamw("x0_lambdas", {model.x0_lambdas}, scalar_lr, 0.96, 0.95, 0.0), // higher beta1 for x0
                adamw("smear", {model.smear_gate->weight, model.smear_lambda, model.backout_lambda}, 0.2, 0.8, 0.95,
                      0.0),
        };
        // Muon groups: matrix params grouped by shape (sorted, as Python's sorted() of shape tuples)
        std::map<std::vector<int64_t>, std::vector<torch::Tensor>> by_shape;
        for (const auto &p: model.transformer->h->parameters())
            by_shape[p.sizes().vec()].push_back(p);
        for (auto &[shape, params]: by_shape)
            groups.push_back({.kind = Kind::Muon,
                              .name = "muon_" + std::to_string(shape[0]) + "x" + std::to_string(shape[1]),
                              .params = std::move(params), .lr = matrix_lr, .initial_lr = matrix_lr,
                              .weight_decay = weight_decay, .beta2 = 0.9, .momentum = 0.95, .ns_steps = 5});
        return MuonAdamW(std::move(groups));
    }

} // namespace nanochat
