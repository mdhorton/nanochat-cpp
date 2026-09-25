// MuonAdamW: Muon for the transformer matrices, AdamW for embeddings, lm_head and scalars. Port of nanochat/optim.py.
// Single GPU for now; the ZeRO-2 distributed version comes with multi-GPU training.
#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "nanochat/gpt.h"
#include "nanochat/safetensors.h"

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
        explicit MuonAdamW(std::vector<OptimGroup> groups);

        void step();

        std::vector<OptimGroup> &groups() { return groups_; }
        const std::vector<OptimGroup> &groups() const { return groups_; }

        // Same structure as Python's Optimizer.state_dict(): per-param state as "state.<i>.<field>", i indexing params
        // across groups in order (Muon state lives on each group's first param), and the group hyperparameters as
        // JSON in metadata["param_groups"].
        safetensors::TensorMap state_dict(safetensors::Metadata &metadata) const;
        void load_state_dict(const safetensors::TensorMap &state, const safetensors::Metadata &metadata);

    private:
        struct AdamWState {
            int64_t step = 0;
            torch::Tensor exp_avg, exp_avg_sq;
        };
        struct MuonState {
            torch::Tensor momentum_buffer, second_momentum_buffer;
        };

        void step_adamw(const OptimGroup &group, std::vector<AdamWState> &states);
        void step_muon(const OptimGroup &group, MuonState &state);

        std::vector<OptimGroup> groups_;
        std::vector<std::vector<AdamWState>> adamw_states_; // per group, per param
        std::vector<MuonState> muon_states_; // per group
    };

    // GPT.setup_optimizer: AdamW groups (lm_head, wte, value_embeds, resid, x0, smear/backout), then one Muon group
    // per matrix shape in sorted order. AdamW LRs scale with 1/sqrt(n_embd / 768).
    MuonAdamW setup_optimizer(GPTImpl &model, double unembedding_lr = 0.004, double embedding_lr = 0.2,
                              double matrix_lr = 0.02, double weight_decay = 0.0, double scalar_lr = 0.5);

} // namespace nanochat
