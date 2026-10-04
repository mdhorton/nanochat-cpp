// MuonAdamW: Muon for the transformer matrices, AdamW for embeddings, lm_head and scalars. Port of nanochat/optim.py.
// With several ranks it syncs the gradients itself (no DDP) and shards the optimizer state ZeRO-2 style:
// - AdamW params < 1024 elements: all_reduce the grad, update the whole param on every rank.
// - other AdamW params: reduce_scatter the grad along dim 0, update this rank's slice, all_gather the param.
// - Muon groups: the K grads are the rows of one stacked buffer (zero_grad installs them as .grad views, so backward
//   accumulates straight into the reduce_scatter input), each rank owns ceil(K / world) of them (zero padded),
//   reduce_scatter, update the owned params, all_gather the stack.
// With gather overlap the all_gathers stay in flight after step() and sync() finishes them where the next forward
// first reads each param. For that a Muon stack is gathered in up to 8 segments in forward order, and a rank owns a
// slice of every segment instead of one contiguous chunk (so the Muon state's layout differs).
// AdamW is one fused kernel per param on CUDA (train/adamw_kernel.cu), the op path elsewhere; both round as Python.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "nanochat/model/gpt.h"
#include "nanochat/train/dist.h"
#include "nanochat/train/safetensors.h"

namespace nanochat {

struct OptimGroup {
  enum class Kind { AdamW, Muon };
  Kind kind = Kind::AdamW;
  std::string name; // for logging and checkpoints
  std::vector<torch::Tensor> params;
  double lr = 0, initial_lr = 0, weight_decay = 0;
  bool gather_last = false; // AdamW with gather overlap: the forward reads these last (lm_head)
  // AdamW
  double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  // Muon (beta2 above is NorMuon's second-moment beta)
  double momentum = 0.95;
  int ns_steps = 5;
};

class MuonAdamW {
public:
  // dist: null (or world size 1) = one GPU. Must outlive the optimizer.
  explicit MuonAdamW(std::vector<OptimGroup> groups, Dist* dist = nullptr);

  void step();

  // Several ranks: leave step()'s all_gathers in flight (false: step() finishes them). Until sync() the gathered
  // params are stale and the Muon .grads are not zero. Set before the first step.
  void set_gather_overlap(bool on);
  // Finishes the in-flight all_gathers of these params (all without arguments): the current stream waits for them.
  // Call before reading a param, and for all before backward.
  void sync(const std::vector<torch::Tensor>& params);
  void sync();

  size_t num_gathers() const {
    return gathers_.size();
  }

  // Muon's update in fused CUDA kernels (muon_kernel.h; not bit-identical to the op path, which is Python's)
  void set_fused_muon(bool on) {
    fused_muon_ = on;
  }

  // Instead of Module::zero_grad(true): AdamW grads to none, Muon grads the zeroed rows of their group's stack.
  void zero_grad();

  std::vector<OptimGroup>& groups() {
    return groups_;
  }

  const std::vector<OptimGroup>& groups() const {
    return groups_;
  }

  // Same structure as Python's Optimizer.state_dict(): per-param state as "state.<i>.<field>", i indexing params
  // across groups in order (Muon state lives on each group's first param), and the group hyperparameters as
  // JSON in metadata["param_groups"]. With several ranks, each rank's state is its shard.
  safetensors::TensorMap state_dict(safetensors::Metadata& metadata) const;
  void load_state_dict(const safetensors::TensorMap& state, const safetensors::Metadata& metadata);

private:
  struct AdamWState {
    int64_t step = 0;
    torch::Tensor exp_avg, exp_avg_sq;
  };

  struct MuonState {
    torch::Tensor momentum_buffer, second_momentum_buffer;
  };

  // in-flight communication of one group
  struct Pending {
    std::vector<Dist::Work> works;    // AdamW: per param; Muon: per segment
    std::vector<torch::Tensor> grads; // AdamW: this rank's grad (slice); Muon: the owned chunk
    std::vector<bool> sharded;        // AdamW: reduce_scattered (else all_reduced)
    int64_t chunk_size = 0;
  };

  // an all_gather of updated params: one per sharded AdamW param and per Muon segment
  struct Gather {
    size_t group = 0;
    int64_t begin = 0, end = 0; // AdamW: the param's index; Muon: rows of the group's stack
    int64_t offset = 0;         // Muon: the segment's start in a rank's chunk
    Dist::Work work;
    torch::Tensor src;    // Muon: this rank's updated chunk, alive until the gather is done
    bool pending = false; // launched, not finished
    bool zero = false;    // Muon: zero_grad() came first, so finish() zeroes the rows
  };

  void layout();
  std::vector<int64_t> owned_rows(size_t group_index) const;
  Pending reduce_adamw(const OptimGroup& group);
  Pending reduce_muon(size_t group_index);
  void compute_adamw(size_t group_index, Pending& pending, std::vector<size_t>& last);
  torch::Tensor compute_muon(size_t group_index, Pending& pending);
  void launch(size_t gather_index, const torch::Tensor& updated = {});
  void finish(Gather& gather);

  int rank() const {
    return dist_ ? dist_->rank() : 0;
  }

  int world_size() const {
    return dist_ ? dist_->world_size() : 1;
  }

  Dist* dist_;
  bool fused_muon_ = false, gather_overlap_ = false;
  std::vector<OptimGroup> groups_;
  std::vector<std::vector<AdamWState>> adamw_states_; // per group, per param
  std::vector<MuonState> muon_states_;                // per group
  std::vector<torch::Tensor> muon_grads_;             // per Muon group: (ceil(K / world) * world, m, n) grad stack
  std::vector<std::vector<int64_t>> muon_segments_;   // per Muon group: each segment's params per rank
  std::vector<Gather> gathers_;                       // several ranks only
  std::vector<std::vector<size_t>> group_gathers_;    // per group: its gathers (AdamW: by param, -1 = none)
  std::vector<size_t> muon_order_;                    // the Muon gathers in forward order
  std::unordered_map<const torch::TensorImpl*, size_t> gather_of_;
  size_t num_pending_ = 0;
};

// GPT.setup_optimizer: AdamW groups (lm_head, wte, value_embeds, resid, x0, smear/backout), then one Muon group
// per matrix shape in sorted order. AdamW LRs scale with 1/sqrt(n_embd / 768).
MuonAdamW setup_optimizer(
      GPTImpl& model, double unembedding_lr = 0.004, double embedding_lr = 0.2, double matrix_lr = 0.02,
      double weight_decay = 0.0, double scalar_lr = 0.5, Dist* dist = nullptr);

} // namespace nanochat
