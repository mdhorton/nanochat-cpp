// Oscillation of the NVFP4 forward weights (TetraJet-v2, arXiv 2510.27527), measured only: over windows of optimizer
// steps, how the weights' NVFP4 values (nvfp4_weight_values) move compared with their master copies.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

namespace nanochat {

class Nvfp4OsciMonitor {
public:
  // weights: (kind, e.g. mlp.c_fc; fp32 (R, C), R % 128, C % 64). The `window` steps after every `every`-th are
  // tracked (window < every, <= 32767).
  Nvfp4OsciMonitor(std::vector<std::pair<std::string, torch::Tensor>> weights, int64_t every, int64_t window);

  // After optimizer step `step`. When a window ends: per kind and "all", fractions of the elements that
  //   moved: changed NVFP4 value at least once
  //   osc:   changed at least twice and ended closer to the start than the path walked (went back)
  //   r8:    NVFP4 path >= 8x master path (TetraJet's reset criterion)
  // and back_q / back_w: the share of the summed NVFP4 / master path that cancelled out, 1 - sum|end - start| / sum
  // path; flips: NVFP4 changes per moved element. Also first / last: the window's steps.
  std::optional<nlohmann::json> after_step(int64_t step);

private:
  struct Track {
    torch::Tensor w0, w_prev, q0, q_prev, path_w, path_q, flips;
  };

  std::vector<std::pair<std::string, torch::Tensor>> weights_;
  int64_t every_, window_;
  int64_t first_ = -1; // the window's first tracked step; -1: none open
  std::vector<Track> tracks_;
};

} // namespace nanochat
