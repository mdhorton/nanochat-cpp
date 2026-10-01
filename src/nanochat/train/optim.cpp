#include "nanochat/train/optim.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/train/adamw_kernel.h"
#include "nanochat/train/muon_kernel.h"

#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nanochat {

namespace {

// Python passes hyperparameters as 0-D fp32 CPU tensors; using the same keeps the fp32 rounding identical.
torch::Tensor cpu_scalar(double v) {
  return torch::tensor(static_cast<float>(v), torch::kFloat32);
}

// Polar Express coefficients (num_iters=5, safety_factor=2e-2, cushion=2), https://arxiv.org/pdf/2505.16932
constexpr std::array<std::array<double, 3>, 5> kPolarExpressCoeffs = {{
      {8.156554524902461, -22.48329292557795, 15.878769915207462},
      {4.042929935166739, -2.808917465908714, 0.5000178451051316},
      {3.8916678022926607, -2.772484153217685, 0.5060648178503393},
      {3.285753657755655, -2.3681294933425376, 0.46449024233003106},
      {2.3465413258596377, -1.7097828382687081, 0.42323551169305323},
}};

torch::Tensor frobenius(const torch::Tensor& x) {
  return at::linalg_vector_norm(x, 2, {-2, -1}, true);
}

// AdamW, as adamw_step_fused: math in fp32, written back to the (possibly bf16) param and state. The CUDA kernel
// replays these ops with the same rounding; this path serves other devices.
void adamw_update(
      const torch::Tensor& p, const torch::Tensor& grad, const torch::Tensor& exp_avg, const torch::Tensor& exp_avg_sq,
      const torch::Tensor& step_t, const torch::Tensor& lr_t, const torch::Tensor& beta1_t,
      const torch::Tensor& beta2_t, const torch::Tensor& eps_t, const torch::Tensor& wd_t) {
  const auto bias1 = 1 - beta1_t.pow(step_t), bias2 = 1 - beta2_t.pow(step_t);
  const auto step_size = lr_t / bias1;
  if (p.is_cuda()) {
    const kernels::AdamWScalars s{
          .decay = (1 - lr_t * wd_t).item<float>(),
          .w1 = (1 - beta1_t).item<float>(),
          .w2 = (1 - beta2_t).item<float>(),
          .inv_bias2 = 1.f / bias2.item<float>(),
          .eps = eps_t.item<float>(),
          .neg_step_size = (-step_size).item<float>()};
    TORCH_CHECK(
          p.is_contiguous() && grad.is_contiguous() && exp_avg.is_contiguous() && exp_avg_sq.is_contiguous() &&
                exp_avg.scalar_type() == p.scalar_type() && exp_avg_sq.scalar_type() == p.scalar_type() &&
                (p.scalar_type() == torch::kFloat32 || p.scalar_type() == torch::kBFloat16) &&
                (grad.scalar_type() == torch::kFloat32 || grad.scalar_type() == torch::kBFloat16),
          "adamw_update: expected contiguous fp32 or bf16 tensors");
    kernels::adamw_step(
          p.data_ptr(), grad.data_ptr(), exp_avg.data_ptr(), exp_avg_sq.data_ptr(), p.numel(),
          p.scalar_type() == torch::kBFloat16, grad.scalar_type() == torch::kBFloat16, s,
          at::cuda::getCurrentCUDAStream());
    // the kernel bypasses autograd's version counters, which the FP8 weight cache keys on
    for (const auto& t : {p, exp_avg, exp_avg_sq})
      t.unsafeGetTensorImpl()->bump_version();
    return;
  }
  auto p32 = p.to(torch::kFloat32);
  auto exp_avg32 = exp_avg.to(torch::kFloat32);
  auto exp_avg_sq32 = exp_avg_sq.to(torch::kFloat32);
  auto grad32 = grad.to(torch::kFloat32);
  p32.mul_(1 - lr_t * wd_t); // decoupled weight decay
  exp_avg32.lerp_(grad32, 1 - beta1_t);
  exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t);
  auto denom = (exp_avg_sq32 / bias2).sqrt() + eps_t;
  p32.add_(exp_avg32 / denom, (-step_size).item());
  p.copy_(p32); // no-ops when already fp32 (to() returned the same tensor)
  exp_avg.copy_(exp_avg32);
  exp_avg_sq.copy_(exp_avg_sq32);
}

// Muon, as muon_step_fused: Nesterov momentum -> MuonEq -> Polar Express -> Muon+ -> NorMuon -> cautious update.
void muon_update(
      torch::Tensor stacked_grads, const torch::Tensor& stacked_params, const torch::Tensor& momentum_buffer,
      const torch::Tensor& second_momentum_buffer, const torch::Tensor& momentum_t, const torch::Tensor& lr_t,
      const torch::Tensor& wd_t, const torch::Tensor& beta2_t, int ns_steps, int64_t red_dim) {
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

// muon_update in fused kernels (muon_kernel.h), Polar Express's polynomial in the GEMMs' alpha / beta. fp32 CUDA
// stacks, contiguous.
void muon_update_fused(
      const torch::Tensor& grads, const torch::Tensor& params, const torch::Tensor& momentum_buffer,
      const torch::Tensor& second_momentum_buffer, float momentum, float lr, float wd, float beta2, int ns_steps) {
  for (const auto& t : {grads, params, momentum_buffer, second_momentum_buffer})
    TORCH_CHECK(
          t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32,
          "muon_update_fused: expected contiguous fp32 CUDA tensors");
  const int64_t k = params.size(0), m = params.size(1), n = params.size(2);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto X = torch::empty({k, m, n}, params.options().dtype(torch::kBFloat16));
  auto scratch = torch::empty({std::max(k * m, kernels::muon_post_scratch(k, m, n))}, params.options());
  kernels::muon_pre(
        grads.data_ptr<float>(), momentum_buffer.data_ptr<float>(), X.data_ptr(), scratch.data_ptr<float>(), k, m, n,
        momentum, stream);
  const bool tall = m > n;
  for (int i = 0; i < ns_steps; ++i) {
    const auto [a, b, c] = kPolarExpressCoeffs[i];
    if (tall) {
      const auto A = at::bmm(X.mT(), X);
      X = at::baddbmm(X, X, at::baddbmm(A, A, A, b, c), a, 1);
    }
    else {
      const auto A = at::bmm(X, X.mT());
      X = at::baddbmm(X, at::baddbmm(A, A, A, b, c), X, a, 1);
    }
  }
  kernels::muon_post(
        X.data_ptr(), params.data_ptr<float>(), second_momentum_buffer.data_ptr<float>(), scratch.data_ptr<float>(), k,
        m, n, lr, wd, beta2, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for (const auto& t : {params, momentum_buffer, second_momentum_buffer})
    t.unsafeGetTensorImpl()->bump_version();
}

} // namespace

MuonAdamW::MuonAdamW(std::vector<OptimGroup> groups, Dist* dist)
    : dist_(dist)
    , groups_(std::move(groups)) {
  adamw_states_.resize(groups_.size());
  muon_states_.resize(groups_.size());
  muon_grads_.resize(groups_.size());
  const int world = world_size();
  for (size_t i = 0; i < groups_.size(); ++i) {
    const auto& group = groups_[i];
    if (group.params.empty())
      throw std::invalid_argument("empty optimizer group " + group.name);
    if (group.kind == OptimGroup::Kind::AdamW) {
      adamw_states_[i].resize(group.params.size());
      continue;
    }
    const auto& p = group.params[0];
    for (const auto& q : group.params)
      if (q.dim() != 2 || q.sizes() != p.sizes())
        throw std::invalid_argument("Muon group " + group.name + " needs 2D params of one shape");
    const auto k = static_cast<int64_t>(group.params.size());
    muon_grads_[i] = torch::zeros({(k + world - 1) / world * world, p.size(0), p.size(1)}, p.options());
  }
}

void MuonAdamW::zero_grad() {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < groups_.size(); ++i) {
    const auto& params = groups_[i].params;
    if (groups_[i].kind == OptimGroup::Kind::AdamW) {
      for (const auto& p : params)
        p.mutable_grad().reset();
      continue;
    }
    muon_grads_[i].zero_();
    for (size_t j = 0; j < params.size(); ++j)
      params[j].mutable_grad() = muon_grads_[i][static_cast<int64_t>(j)];
  }
}

// As Python: launch every group's reduce, then per group wait, update and launch the gathers, then finish them.
void MuonAdamW::step() {
  torch::NoGradGuard no_grad;
  std::vector<Pending> pending;
  pending.reserve(groups_.size());
  for (size_t i = 0; i < groups_.size(); ++i)
    pending.push_back(groups_[i].kind == OptimGroup::Kind::AdamW ? reduce_adamw(groups_[i]) : reduce_muon(i));
  std::vector<Gather> gathers;
  for (size_t i = 0; i < groups_.size(); ++i)
    if (groups_[i].kind == OptimGroup::Kind::AdamW)
      compute_adamw(groups_[i], pending[i], adamw_states_[i], gathers);
    else
      compute_muon(groups_[i], pending[i], muon_states_[i], gathers);
  for (auto& g : gathers) {
    Dist::wait(g.work);
    if (g.params != nullptr)
      for (size_t j = 0; j < g.params->size(); ++j)
        (*g.params)[j].copy_(g.stacked[static_cast<int64_t>(j)]);
  }
}

MuonAdamW::Pending MuonAdamW::reduce_adamw(const OptimGroup& group) {
  Pending pending;
  const int world = world_size();
  for (const auto& p : group.params) {
    auto grad = p.grad();
    if (world == 1) {
      pending.works.emplace_back();
      pending.grads.push_back(grad);
      pending.sharded.push_back(false);
    }
    else if (p.numel() < 1024) {
      pending.works.push_back(dist_->all_reduce(grad, Dist::Op::Avg));
      pending.grads.push_back(grad);
      pending.sharded.push_back(false);
    }
    else {
      if (grad.size(0) % world != 0)
        throw std::invalid_argument(group.name + ": dim 0 must be divisible by the world size");
      auto slice = torch::empty_like(grad.slice(0, 0, grad.size(0) / world));
      pending.works.push_back(dist_->reduce_scatter(slice, grad, Dist::Op::Avg));
      pending.grads.push_back(slice);
      pending.sharded.push_back(true);
    }
  }
  return pending;
}

MuonAdamW::Pending MuonAdamW::reduce_muon(size_t group_index) {
  const auto& params = groups_[group_index].params;
  const auto k = static_cast<int64_t>(params.size());
  auto& stacked = muon_grads_[group_index];
  // grads set up elsewhere (not zero_grad's views): copy them in
  for (int64_t j = 0; j < k; ++j) {
    const auto& g = params[j].grad();
    if (!g.defined())
      throw std::invalid_argument(groups_[group_index].name + ": param " + std::to_string(j) + " has no gradient");
    if (g.data_ptr() != stacked[j].data_ptr() || !g.is_contiguous())
      stacked[j].copy_(g);
  }
  if (k < stacked.size(0))
    stacked.slice(0, k).zero_();
  Pending pending;
  pending.chunk_size = stacked.size(0) / world_size();
  if (world_size() == 1) { // this rank owns every param: the stack is the chunk
    pending.works.emplace_back();
    pending.grads.push_back(stacked);
    return pending;
  }
  auto grad_chunk = torch::empty({pending.chunk_size, stacked.size(1), stacked.size(2)}, stacked.options());
  pending.works.push_back(dist_->reduce_scatter(grad_chunk, stacked, Dist::Op::Avg));
  pending.grads.push_back(grad_chunk);
  pending.stacked = stacked;
  return pending;
}

void MuonAdamW::compute_adamw(
      const OptimGroup& group, Pending& pending, std::vector<AdamWState>& states, std::vector<Gather>& gathers) {
  const auto lr_t = cpu_scalar(group.lr), beta1_t = cpu_scalar(group.beta1), beta2_t = cpu_scalar(group.beta2);
  const auto eps_t = cpu_scalar(group.eps), wd_t = cpu_scalar(group.weight_decay);
  for (size_t j = 0; j < group.params.size(); ++j) {
    Dist::wait(pending.works[j]);
    auto p = group.params[j];
    auto p_slice = p;
    if (pending.sharded[j]) {
      const int64_t rows = p.size(0) / world_size();
      p_slice = p.slice(0, rank() * rows, (rank() + 1) * rows);
    }
    auto& state = states[j];
    if (!state.exp_avg.defined()) {
      state.exp_avg = torch::zeros_like(p_slice);
      state.exp_avg_sq = torch::zeros_like(p_slice);
    }
    ++state.step;
    adamw_update(
          p_slice, pending.grads[j], state.exp_avg, state.exp_avg_sq, cpu_scalar(static_cast<double>(state.step)), lr_t,
          beta1_t, beta2_t, eps_t, wd_t);
    if (pending.sharded[j])
      gathers.push_back({dist_->all_gather(p, p_slice), {}, nullptr});
  }
}

void MuonAdamW::compute_muon(
      const OptimGroup& group, Pending& pending, MuonState& state, std::vector<Gather>& gathers) {
  Dist::wait(pending.works[0]);
  const auto& params = group.params;
  const int64_t m = params[0].size(0), n = params[0].size(1), k = static_cast<int64_t>(params.size());
  const int64_t chunk = pending.chunk_size;
  const auto options = params[0].options();
  if (!state.momentum_buffer.defined()) {
    state.momentum_buffer = torch::zeros({chunk, m, n}, options);
    state.second_momentum_buffer = m >= n ? torch::zeros({chunk, m, 1}, options) : torch::zeros({chunk, 1, n}, options);
  }
  const int64_t red_dim = m >= n ? -1 : -2;

  // this rank updates params [start, start + num_owned)
  const int64_t start = rank() * chunk;
  const int64_t num_owned = std::min(chunk, std::max<int64_t>(0, k - start));
  torch::Tensor owned;
  if (num_owned > 0) {
    owned = torch::stack(std::vector<torch::Tensor>(params.begin() + start, params.begin() + start + num_owned));
    // tall matrices get a larger lr
    const double lr = group.lr * std::pow(std::max(1.0, static_cast<double>(m) / static_cast<double>(n)), 0.5);
    if (fused_muon_ && owned.is_cuda())
      muon_update_fused(
            pending.grads[0].slice(0, 0, num_owned), owned, state.momentum_buffer.slice(0, 0, num_owned),
            state.second_momentum_buffer.slice(0, 0, num_owned), static_cast<float>(group.momentum),
            static_cast<float>(lr), static_cast<float>(group.weight_decay), static_cast<float>(group.beta2),
            group.ns_steps);
    else
      muon_update(
            pending.grads[0].slice(0, 0, num_owned), owned, state.momentum_buffer.slice(0, 0, num_owned),
            state.second_momentum_buffer.slice(0, 0, num_owned), cpu_scalar(group.momentum), cpu_scalar(lr),
            cpu_scalar(group.weight_decay), cpu_scalar(group.beta2), group.ns_steps, red_dim);
  }
  if (!pending.stacked.defined()) { // one rank: the updated stack maps onto the params
    gathers.push_back({{}, owned, &params});
    return;
  }
  auto updated = torch::empty({chunk, m, n}, options);
  if (num_owned > 0)
    updated.slice(0, 0, num_owned).copy_(owned);
  if (num_owned < chunk)
    updated.slice(0, num_owned).zero_();
  gathers.push_back({dist_->all_gather(pending.stacked, updated), pending.stacked, &params});
}

safetensors::TensorMap MuonAdamW::state_dict(safetensors::Metadata& metadata) const {
  safetensors::TensorMap state;
  auto groups = nlohmann::json::array();
  int64_t index = 0;
  for (size_t i = 0; i < groups_.size(); ++i) {
    const auto& group = groups_[i];
    nlohmann::json g = {
          {"name", group.name},
          {"lr", group.lr},
          {"initial_lr", group.initial_lr},
          {"weight_decay", group.weight_decay},
          {"params", nlohmann::json::array()}};
    for (size_t j = 0; j < group.params.size(); ++j)
      g["params"].push_back(index + static_cast<int64_t>(j));
    const auto prefix = [](int64_t idx) {
      return "state." + std::to_string(idx) + ".";
    };
    if (group.kind == OptimGroup::Kind::AdamW) {
      g.update({{"kind", "adamw"}, {"betas", {group.beta1, group.beta2}}, {"eps", group.eps}});
      for (size_t j = 0; j < group.params.size(); ++j) {
        const auto& s = adamw_states_[i][j];
        if (!s.exp_avg.defined())
          continue;
        const auto p = prefix(index + static_cast<int64_t>(j));
        state[p + "step"] = torch::tensor(s.step, torch::kInt64);
        state[p + "exp_avg"] = s.exp_avg;
        state[p + "exp_avg_sq"] = s.exp_avg_sq;
      }
    }
    else {
      g.update({{"kind", "muon"}, {"momentum", group.momentum}, {"ns_steps", group.ns_steps}, {"beta2", group.beta2}});
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

void MuonAdamW::load_state_dict(const safetensors::TensorMap& state, const safetensors::Metadata& metadata) {
  const auto groups = nlohmann::json::parse(metadata.at("param_groups"));
  if (groups.size() != groups_.size())
    throw std::runtime_error("optimizer state has a different number of groups");
  torch::NoGradGuard no_grad;
  int64_t index = 0;
  for (size_t i = 0; i < groups_.size(); ++i) {
    auto& group = groups_[i];
    const auto& g = groups[i];
    if (g["params"].size() != group.params.size() || (g["kind"] == "muon") != (group.kind == OptimGroup::Kind::Muon))
      throw std::runtime_error("optimizer group " + std::to_string(i) + " doesn't match the model");
    group.lr = g["lr"];
    group.initial_lr = g["initial_lr"];
    group.weight_decay = g["weight_decay"];
    const auto prefix = [](int64_t idx) {
      return "state." + std::to_string(idx) + ".";
    };
    // state tensors are copied onto each param's device
    const auto get = [&](const std::string& key, const torch::Tensor& like) {
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
        auto& s = adamw_states_[i][j];
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
        muon_states_[i].second_momentum_buffer = get(prefix(index) + "second_momentum_buffer", group.params[0]);
      }
    }
    index += static_cast<int64_t>(group.params.size());
  }
}

MuonAdamW setup_optimizer(
      GPTImpl& model, double unembedding_lr, double embedding_lr, double matrix_lr, double weight_decay,
      double scalar_lr, Dist* dist) {
  using Kind = OptimGroup::Kind;
  const double dmodel_lr_scale = std::pow(static_cast<double>(model.config().n_embd) / 768.0, -0.5);
  auto adamw = [](std::string name, std::vector<torch::Tensor> params, double lr, double beta1, double beta2,
                  double wd) {
    return OptimGroup{
          .kind = Kind::AdamW,
          .name = std::move(name),
          .params = std::move(params),
          .lr = lr,
          .initial_lr = lr,
          .weight_decay = wd,
          .beta1 = beta1,
          .beta2 = beta2,
          .eps = 1e-10};
  };
  std::vector<OptimGroup> groups = {
        adamw("lm_head", model.lm_head->parameters(), unembedding_lr * dmodel_lr_scale, 0.8, 0.96, 0.01),
        adamw("wte", model.transformer->wte->parameters(), embedding_lr * dmodel_lr_scale, 0.8, 0.995, 0.001),
        adamw("value_embeds", model.value_embeds->parameters(), embedding_lr * dmodel_lr_scale * 0.5, 0.8, 0.995, 0.01),
        adamw("resid_lambdas", {model.resid_lambdas}, scalar_lr * 0.01, 0.8, 0.95, 0.05),
        adamw("x0_lambdas", {model.x0_lambdas}, scalar_lr, 0.96, 0.95, 0.0), // higher beta1 for x0
        adamw("smear", {model.smear_gate->weight, model.smear_lambda, model.backout_lambda}, 0.2, 0.8, 0.95, 0.0),
  };
  // Muon groups: matrix params grouped by shape (sorted, as Python's sorted() of shape tuples)
  std::map<std::vector<int64_t>, std::vector<torch::Tensor>> by_shape;
  for (const auto& p : model.transformer->h->parameters())
    by_shape[p.sizes().vec()].push_back(p);
  for (auto& [shape, params] : by_shape)
    groups.push_back(
          {.kind = Kind::Muon,
           .name = "muon_" + std::to_string(shape[0]) + "x" + std::to_string(shape[1]),
           .params = std::move(params),
           .lr = matrix_lr,
           .initial_lr = matrix_lr,
           .weight_decay = weight_decay,
           .beta2 = 0.9,
           .momentum = 0.95,
           .ns_steps = 5});
  return MuonAdamW(std::move(groups), dist);
}

} // namespace nanochat
