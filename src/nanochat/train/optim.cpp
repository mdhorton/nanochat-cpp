#include "nanochat/train/optim.h"

#include <ATen/cuda/CUDAContext.h>

#include "nanochat/train/adamw_kernel.h"
#include "nanochat/train/muon_kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nanochat {

namespace {

constexpr int64_t kMuonSegments = 8; // gather overlap: the most all_gathers per Muon stack
constexpr size_t kNoGather = static_cast<size_t>(-1);

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

// The first half of an fp32 stack's bytes as a bf16 stack of the same shape: bf16 row j lies in fp32 row j / 2.
torch::Tensor bf16_alias(const torch::Tensor& stack) {
  return stack.view(torch::kBFloat16).flatten().narrow(0, 0, stack.numel()).view(stack.sizes());
}

} // namespace

MuonAdamW::MuonAdamW(std::vector<OptimGroup> groups, Dist* dist)
    : dist_(dist)
    , groups_(std::move(groups)) {
  adamw_states_.resize(groups_.size());
  muon_states_.resize(groups_.size());
  muon_grads_.resize(groups_.size());
  muon_segments_.resize(groups_.size());
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
    muon_segments_[i] = {muon_grads_[i].size(0) / world};
  }
  layout();
}

void MuonAdamW::set_gather_overlap(bool on) {
  const int world = world_size();
  for (size_t i = 0; i < groups_.size(); ++i) {
    if (groups_[i].kind != OptimGroup::Kind::Muon)
      continue;
    if (muon_states_[i].momentum_buffer.defined())
      throw std::logic_error("set_gather_overlap: the Muon state already has a layout");
    const int64_t chunk = muon_grads_[i].size(0) / world;
    const int64_t n = on && world > 1 ? std::min(kMuonSegments, chunk) : 1;
    muon_segments_[i].assign(static_cast<size_t>(n), chunk / n);
    for (int64_t s = 0; s < chunk % n; ++s)
      ++muon_segments_[i][static_cast<size_t>(s)];
  }
  gather_overlap_ = on;
  layout();
}

// The gathers of the current segments. Segment s of a Muon stack is rows [world * o, world * (o + c)) for its c
// params per rank and o the ones before it; rank r owns its rows [r * c, (r + 1) * c).
void MuonAdamW::layout() {
  gathers_.clear();
  gather_of_.clear();
  muon_order_.clear();
  group_gathers_.assign(groups_.size(), {});
  const int world = world_size();
  if (world == 1)
    return;
  for (size_t i = 0; i < groups_.size(); ++i) {
    const auto& params = groups_[i].params;
    if (groups_[i].kind == OptimGroup::Kind::AdamW) {
      for (size_t j = 0; j < params.size(); ++j) {
        if (params[j].numel() < 1024) { // all_reduced: no gather
          group_gathers_[i].push_back(kNoGather);
          continue;
        }
        gather_of_[params[j].unsafeGetTensorImpl()] = gathers_.size();
        group_gathers_[i].push_back(gathers_.size());
        gathers_.push_back({.group = i, .begin = static_cast<int64_t>(j), .end = static_cast<int64_t>(j) + 1});
      }
      continue;
    }
    int64_t offset = 0;
    for (const int64_t count : muon_segments_[i]) {
      Gather g{.group = i, .begin = world * offset, .end = world * (offset + count), .offset = offset};
      for (int64_t j = g.begin; j < std::min(g.end, static_cast<int64_t>(params.size())); ++j)
        gather_of_[params[static_cast<size_t>(j)].unsafeGetTensorImpl()] = gathers_.size();
      muon_order_.push_back(gathers_.size());
      group_gathers_[i].push_back(gathers_.size());
      gathers_.push_back(std::move(g));
      offset += count;
    }
  }
  // a stack holds its params in forward order: interleave the stacks' segments by where they start
  std::ranges::stable_sort(muon_order_, {}, [&](size_t g) {
    return static_cast<double>(gathers_[g].begin) / static_cast<double>(muon_grads_[gathers_[g].group].size(0));
  });
}

// The stack rows of this rank's chunk, ascending (padding rows last).
std::vector<int64_t> MuonAdamW::owned_rows(size_t group_index) const {
  std::vector<int64_t> rows;
  int64_t offset = 0;
  for (const int64_t count : muon_segments_[group_index]) {
    for (int64_t t = 0; t < count; ++t)
      rows.push_back(world_size() * offset + rank() * count + t);
    offset += count;
  }
  return rows;
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
    if (group_gathers_[i].empty())
      muon_grads_[i].zero_();
    // an in-flight gather into the stack: finish() zeroes its rows. bf16 gather: the updates in flight span the
    // stack's bytes, so none are zeroed here and the group's last finish() zeroes the whole stack.
    bool whole = false;
    for (const size_t gi : group_gathers_[i]) {
      auto& g = gathers_[gi];
      g.zero = g.pending;
      whole = whole || (g.pending && g.dst.defined());
    }
    for (const size_t gi : group_gathers_[i])
      if (const auto& g = gathers_[gi]; !g.pending && !whole)
        muon_grads_[i].slice(0, g.begin, g.end).zero_();
    for (size_t j = 0; j < params.size(); ++j)
      params[j].mutable_grad() = muon_grads_[i][static_cast<int64_t>(j)];
  }
}

// As Python: launch every group's reduce, then per group wait, update and launch the gathers, then finish them.
// Gather overlap: the gathers launch in forward order (embeddings, Muon segments, lm_head) and sync() finishes them.
void MuonAdamW::step() {
  torch::NoGradGuard no_grad;
  if (num_pending_ > 0)
    throw std::logic_error("MuonAdamW::step: sync() the previous step's gathers before backward");
  std::vector<Pending> pending;
  pending.reserve(groups_.size());
  for (size_t i = 0; i < groups_.size(); ++i)
    pending.push_back(groups_[i].kind == OptimGroup::Kind::AdamW ? reduce_adamw(groups_[i]) : reduce_muon(i));
  std::vector<torch::Tensor> updated(groups_.size()), dst(groups_.size());
  std::vector<size_t> last;
  for (size_t i = 0; i < groups_.size(); ++i) {
    if (groups_[i].kind == OptimGroup::Kind::AdamW) {
      compute_adamw(i, pending[i], last);
      continue;
    }
    updated[i] = compute_muon(i, pending[i]);
    if (muon_bf16_gather_) // the grads are reduced: the gathered updates go in the grad stack's bytes
      dst[i] = bf16_alias(muon_grads_[i]);
    if (!gather_overlap_)
      for (const size_t gi : group_gathers_[i])
        launch(gi, updated[i], dst[i]);
  }
  if (gather_overlap_)
    for (const size_t gi : muon_order_)
      launch(gi, updated[gathers_[gi].group], dst[gathers_[gi].group]);
  for (const size_t gi : last)
    launch(gi);
  if (!gather_overlap_)
    sync();
}

void MuonAdamW::launch(size_t gather_index, const torch::Tensor& updated, const torch::Tensor& dst) {
  auto& g = gathers_[gather_index];
  const auto& group = groups_[g.group];
  const int world = world_size();
  if (group.kind == OptimGroup::Kind::AdamW) { // in place: the param's other slices from their ranks
    auto p = group.params[static_cast<size_t>(g.begin)];
    const int64_t rows = p.size(0) / world;
    auto p_slice = p.slice(0, rank() * rows, (rank() + 1) * rows);
    g.work = dist_->all_gather(p, p_slice);
  }
  else { // the params into the grad stack, or their bf16 updates into dst: finish() applies them
    const int64_t count = (g.end - g.begin) / world;
    auto out = (dst.defined() ? dst : muon_grads_[g.group]).slice(0, g.begin, g.end);
    auto in = updated.slice(0, g.offset, g.offset + count);
    g.work = dist_->all_gather(out, in);
    g.src = updated;
    g.dst = dst;
  }
  g.pending = true;
  ++num_pending_;
}

void MuonAdamW::finish(Gather& g) {
  Dist::wait(g.work);
  g.work = {};
  g.src = torch::Tensor();
  if (groups_[g.group].kind == OptimGroup::Kind::Muon) {
    torch::NoGradGuard no_grad;
    const auto& params = groups_[g.group].params;
    const auto& stacked = muon_grads_[g.group];
    for (int64_t j = g.begin; j < std::min(g.end, static_cast<int64_t>(params.size())); ++j)
      if (g.dst.defined())
        params[static_cast<size_t>(j)].add_(g.dst[j]);
      else
        params[static_cast<size_t>(j)].copy_(stacked[j]);
    g.pending = false;
    if (g.zero && !g.dst.defined())
      stacked.slice(0, g.begin, g.end).zero_();
    else if (g.zero && std::ranges::none_of(group_gathers_[g.group], [&](size_t gi) { return gathers_[gi].pending; }))
      stacked.zero_(); // bf16 gather (see zero_grad): the group's last gather is done
    g.dst = torch::Tensor();
  }
  g.pending = g.zero = false;
  --num_pending_;
}

void MuonAdamW::sync(const std::vector<torch::Tensor>& params) {
  if (num_pending_ == 0)
    return;
  for (const auto& p : params)
    if (const auto it = gather_of_.find(p.unsafeGetTensorImpl());
        it != gather_of_.end() && gathers_[it->second].pending)
      finish(gathers_[it->second]);
}

void MuonAdamW::sync() {
  for (auto& g : gathers_)
    if (g.pending)
      finish(g);
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
  const int world = world_size();
  pending.chunk_size = stacked.size(0) / world;
  if (world == 1) { // this rank owns every param: the stack is the chunk
    pending.works.emplace_back();
    pending.grads.push_back(stacked);
    return pending;
  }
  const auto in_stack = muon_bf16_reduce_ ? stacked.to(torch::kBFloat16) : stacked;
  if (muon_bf16_reduce_)
    pending.bf16 = in_stack;
  auto grad_chunk = torch::empty({pending.chunk_size, stacked.size(1), stacked.size(2)}, in_stack.options());
  int64_t offset = 0;
  for (const int64_t count : muon_segments_[group_index]) { // each segment's slices land in chunk order
    auto in = in_stack.slice(0, world * offset, world * (offset + count));
    auto out = grad_chunk.slice(0, offset, offset + count);
    pending.works.push_back(dist_->reduce_scatter(out, in, Dist::Op::Avg));
    offset += count;
  }
  pending.grads.push_back(grad_chunk);
  return pending;
}

// Updates the group and launches its gathers (gather overlap: a gather_last group's indices go to `last` instead).
void MuonAdamW::compute_adamw(size_t group_index, Pending& pending, std::vector<size_t>& last) {
  const auto& group = groups_[group_index];
  auto& states = adamw_states_[group_index];
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
    if (!pending.sharded[j])
      continue;
    // the grad and its reduced slice are used up: free them for the rest of step() (the work can hold them too)
    pending.works[j] = {};
    pending.grads[j] = torch::Tensor();
    p.mutable_grad().reset();
    if (gather_overlap_ && group.gather_last)
      last.push_back(group_gathers_[group_index][j]);
    else
      launch(group_gathers_[group_index][j]);
  }
}

// Updates this rank's params. Several ranks: returns its chunk for the gathers (padding rows zero): the updated
// params, or with the bf16 gather their updates.
torch::Tensor MuonAdamW::compute_muon(size_t group_index, Pending& pending) {
  const auto& group = groups_[group_index];
  auto& state = muon_states_[group_index];
  for (const auto& work : pending.works)
    Dist::wait(work);
  const auto& params = group.params;
  const int64_t m = params[0].size(0), n = params[0].size(1), k = static_cast<int64_t>(params.size());
  const int64_t chunk = pending.chunk_size;
  const auto options = params[0].options();
  if (!state.momentum_buffer.defined()) {
    state.momentum_buffer = torch::zeros({chunk, m, n}, options);
    state.second_momentum_buffer = m >= n ? torch::zeros({chunk, m, 1}, options) : torch::zeros({chunk, 1, n}, options);
  }
  const int64_t red_dim = m >= n ? -1 : -2;
  const auto grads = pending.grads[0].to(params[0].scalar_type()); // bf16 reduce: back to the params' fp32
  // the reduce is done: free its bf16 stack and chunk for the rest of step() (the works can hold them too)
  pending.works.clear();
  pending.grads.clear();
  pending.bf16 = torch::Tensor();

  // this rank updates the params of its first num_owned rows
  const auto rows = owned_rows(group_index);
  const auto num_owned = static_cast<int64_t>(std::ranges::lower_bound(rows, k) - rows.begin());
  torch::Tensor owned;
  std::vector<torch::Tensor> owned_params;
  if (num_owned > 0) {
    for (int64_t t = 0; t < num_owned; ++t)
      owned_params.push_back(params[static_cast<size_t>(rows[static_cast<size_t>(t)])]);
    owned = torch::stack(owned_params);
    // tall matrices get a larger lr
    const double lr = group.lr * std::pow(std::max(1.0, static_cast<double>(m) / static_cast<double>(n)), 0.5);
    if (fused_muon_ && owned.is_cuda())
      muon_update_fused(
            grads.slice(0, 0, num_owned), owned, state.momentum_buffer.slice(0, 0, num_owned),
            state.second_momentum_buffer.slice(0, 0, num_owned), static_cast<float>(group.momentum),
            static_cast<float>(lr), static_cast<float>(group.weight_decay), static_cast<float>(group.beta2),
            group.ns_steps);
    else
      muon_update(
            grads.slice(0, 0, num_owned), owned, state.momentum_buffer.slice(0, 0, num_owned),
            state.second_momentum_buffer.slice(0, 0, num_owned), cpu_scalar(group.momentum), cpu_scalar(lr),
            cpu_scalar(group.weight_decay), cpu_scalar(group.beta2), group.ns_steps, red_dim);
  }
  if (world_size() == 1) { // the updated stack maps onto the params
    for (int64_t j = 0; j < k; ++j)
      params[static_cast<size_t>(j)].copy_(owned[j]);
    return {};
  }
  auto updated = torch::empty({chunk, m, n}, muon_bf16_gather_ ? options.dtype(torch::kBFloat16) : options);
  if (muon_bf16_gather_)
    for (int64_t t = 0; t < num_owned; ++t) // the params still hold the old values
      updated[t].copy_(owned[t] - owned_params[static_cast<size_t>(t)]);
  else if (num_owned > 0)
    updated.slice(0, 0, num_owned).copy_(owned);
  if (num_owned < chunk)
    updated.slice(0, num_owned).zero_();
  return updated;
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
      if (muon_segments_[i].size() > 1)
        g["segments"] = muon_segments_[i]; // the state's layout
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
      if (world_size() > 1) { // the state keeps the layout it was saved with (one segment before gather overlap)
        const int64_t chunk = muon_grads_[i].size(0) / world_size();
        const auto segments = g.value("segments", std::vector<int64_t>{chunk});
        if (std::accumulate(segments.begin(), segments.end(), int64_t{0}) != chunk)
          throw std::runtime_error("optimizer group " + std::to_string(i) + " was saved with another world size");
        muon_segments_[i] = segments;
      }
      if (state.contains(prefix(index) + "momentum_buffer")) {
        muon_states_[i].momentum_buffer = get(prefix(index) + "momentum_buffer", group.params[0]);
        muon_states_[i].second_momentum_buffer = get(prefix(index) + "second_momentum_buffer", group.params[0]);
      }
    }
    index += static_cast<int64_t>(group.params.size());
  }
  layout();
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
  groups[0].gather_last = true;
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
