// MuonAdamW: Muon for the transformer matrices, AdamW for embeddings, lm_head and scalars. Port of nanochat/optim.py.
// With several ranks it syncs the gradients itself (no DDP) and shards the optimizer state ZeRO-2 style:
// - AdamW params < 1024 elements: all_reduce the grad, update the whole param on every rank.
// - other AdamW params: reduce_scatter the grad along dim 0, update this rank's slice, all_gather the param.
// - Muon groups: the K grads are the rows of one stacked buffer (zero_grad installs them as .grad views, so backward
//   accumulates straight into the reduce_scatter input), each rank owns ceil(K / world) of them (zero padded),
//   reduce_scatter, update the owned params, all_gather the stack.
// AdamW is one fused kernel per param on CUDA (train/adamw_kernel.cu), the op path elsewhere; both round as Python.
#pragma once

#include <string>
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
    std::vector<Dist::Work> works;    // AdamW: per param; Muon: one
    std::vector<torch::Tensor> grads; // AdamW: this rank's grad (slice); Muon: the owned chunk
    std::vector<bool> sharded;        // AdamW: reduce_scattered (else all_reduced)
    torch::Tensor stacked;            // Muon: the group's grad stack, reduce_scatter input and all_gather output
    int64_t chunk_size = 0;
  };

  struct Gather {
    Dist::Work work;
    torch::Tensor stacked; // Muon: gathered params, copied back after the wait
    const std::vector<torch::Tensor>* params = nullptr;
  };

  Pending reduce_adamw(const OptimGroup& group);
  Pending reduce_muon(size_t group_index);
  void compute_adamw(
        const OptimGroup& group, Pending& pending, std::vector<AdamWState>& states, std::vector<Gather>& gathers);
  void compute_muon(const OptimGroup& group, Pending& pending, MuonState& state, std::vector<Gather>& gathers);

  int rank() const {
    return dist_ ? dist_->rank() : 0;
  }

  int world_size() const {
    return dist_ ? dist_->world_size() : 1;
  }

  Dist* dist_;
  bool fused_muon_ = false;
  std::vector<OptimGroup> groups_;
  std::vector<std::vector<AdamWState>> adamw_states_; // per group, per param
  std::vector<MuonState> muon_states_;                // per group
  std::vector<torch::Tensor> muon_grads_;             // per Muon group: (ceil(K / world) * world, m, n) grad stack
};

// GPT.setup_optimizer: AdamW groups (lm_head, wte, value_embeds, resid, x0, smear/backout), then one Muon group
// per matrix shape in sorted order. AdamW LRs scale with 1/sqrt(n_embd / 768).
MuonAdamW setup_optimizer(
      GPTImpl& model, double unembedding_lr = 0.004, double embedding_lr = 0.2, double matrix_lr = 0.02,
      double weight_decay = 0.0, double scalar_lr = 0.5, Dist* dist = nullptr);

} // namespace nanochat
