#include "nanochat/train/nvfp4_osci.h"

#include <map>
#include <ranges>
#include <stdexcept>

#include "nanochat/model/nvfp4.h"

namespace nanochat {

Nvfp4OsciMonitor::Nvfp4OsciMonitor(
      std::vector<std::pair<std::string, torch::Tensor>> weights, int64_t every, int64_t window)
    : weights_(std::move(weights))
    , every_(every)
    , window_(window) {
  if (every_ <= 0 || window_ <= 0 || window_ >= every_ || window_ > 32767)
    throw std::invalid_argument("nvfp4_osci: 0 < window < every, window <= 32767");
  for (const auto& [kind, w] : weights_)
    TORCH_CHECK(
          w.dim() == 2 && w.size(0) % 128 == 0 && w.size(1) % 64 == 0 && w.scalar_type() == torch::kFloat32,
          "nvfp4_osci: ", kind, ": expected fp32 (R, C), R % 128, C % 64");
}

std::optional<nlohmann::json> Nvfp4OsciMonitor::after_step(int64_t step) {
  torch::NoGradGuard no_grad;
  if (first_ < 0) {
    if ((step + 1) % every_ != 0)
      return std::nullopt;
    first_ = step + 1;
    tracks_.clear();
    for (const auto& w : weights_ | std::views::values) {
      const auto q = nvfp4_weight_values(w).to(torch::kFloat32);
      tracks_.push_back(
            {w.clone(), w.clone(), q, q.clone(), torch::zeros_like(w), torch::zeros_like(w),
             torch::zeros(w.sizes(), w.options().dtype(torch::kInt16))});
    }
    return std::nullopt;
  }

  for (size_t i = 0; i < weights_.size(); ++i) {
    const auto& w = weights_[i].second;
    auto& t = tracks_[i];
    const auto q = nvfp4_weight_values(w).to(torch::kFloat32);
    const auto dq = q - t.q_prev;
    t.path_q.add_(dq.abs());
    t.flips.add_(dq != 0);
    t.path_w.add_((w - t.w_prev).abs());
    t.w_prev.copy_(w);
    t.q_prev.copy_(q);
  }
  if (step + 1 - first_ < window_)
    return std::nullopt;

  // per kind: n, moved, osc, r8, flips, path_q, net_q, path_w, net_w (fp64 on the device, one copy at the end)
  std::map<std::string, torch::Tensor> sums;
  for (size_t i = 0; i < weights_.size(); ++i) {
    const auto& w = weights_[i].second;
    const auto& t = tracks_[i];
    const auto net_q = (t.q_prev - t.q0).abs(), net_w = (w - t.w0).abs();
    const auto f64 = [](const torch::Tensor& x) {
      return x.sum(torch::kFloat64);
    };
    const auto s = torch::stack(
          {torch::full({}, static_cast<double>(w.numel()), w.options().dtype(torch::kFloat64)), f64(t.flips > 0),
           f64((t.flips >= 2) & (net_q < t.path_q * (1 - 1e-6))), f64((t.path_q > 0) & (t.path_q >= 8 * t.path_w)),
           f64(t.flips), f64(t.path_q), f64(net_q), f64(t.path_w), f64(net_w)});
    for (const auto& kind : {weights_[i].first, std::string("all")}) {
      auto& acc = sums[kind];
      acc = acc.defined() ? acc + s : s;
    }
  }
  tracks_.clear();

  nlohmann::json out{{"first", first_}, {"last", step}};
  for (const auto& [kind, acc] : sums) {
    const auto v = acc.cpu();
    const auto* a = v.data_ptr<double>();
    const auto frac = [](double x, double n) {
      return n > 0 ? x / n : 0.0;
    };
    const auto back = [](double net, double path) {
      return path > 0 ? 1 - net / path : 0.0;
    };
    out[kind] = {{"moved", frac(a[1], a[0])}, {"osc", frac(a[2], a[0])},    {"r8", frac(a[3], a[0])},
                 {"flips", frac(a[4], a[1])}, {"back_q", back(a[6], a[5])}, {"back_w", back(a[8], a[7])}};
  }
  first_ = -1;
  return out;
}

} // namespace nanochat
