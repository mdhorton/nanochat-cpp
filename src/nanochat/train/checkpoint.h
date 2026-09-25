// Training checkpoints, laid out like nanochat/checkpoint_manager.py but in safetensors:
//   model_<step>.safetensors, optim_<step>_rank<r>.safetensors (optimizer state is per rank), meta_<step>.json.
// tools/convert_checkpoint.py converts to and from Python's .pt files.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

#include <nlohmann/json.hpp>

#include "nanochat/model/gpt.h"
#include "nanochat/train/optim.h"

namespace nanochat {

    // model_000123.safetensors etc.
    std::filesystem::path checkpoint_file(const std::filesystem::path &dir, const std::string &kind, int64_t step,
                                          std::optional<int> rank = std::nullopt);

    // Rank 0 writes the model and meta; every rank writes its optimizer state (when given).
    void save_checkpoint(const std::filesystem::path &dir, int64_t step, const GPTImpl &model,
                         const MuonAdamW *optimizer, const nlohmann::json &meta, int rank = 0);

    struct Checkpoint {
        safetensors::TensorMap model, optimizer;
        safetensors::Metadata optimizer_metadata;
        nlohmann::json meta;
    };

    Checkpoint load_checkpoint(const std::filesystem::path &dir, int64_t step, torch::Device device,
                               bool load_optimizer, int rank = 0);

    // Largest step with a model file in dir, if any.
    std::optional<int64_t> last_checkpoint_step(const std::filesystem::path &dir);

    // GPTConfig as Python's asdict(GPTConfig), for meta["model_config"].
    nlohmann::json config_to_json(const GPTConfig &config);
    GPTConfig config_from_json(const nlohmann::json &j);

} // namespace nanochat
