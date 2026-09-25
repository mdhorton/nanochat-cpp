#include "nanochat/train/checkpoint.h"

#include <cstdio>
#include <fstream>
#include <regex>

namespace nanochat {

    namespace fs = std::filesystem;

    fs::path checkpoint_file(const fs::path &dir, const std::string &kind, int64_t step, std::optional<int> rank) {
        char name[64];
        const auto ext = kind == "meta" ? "json" : "safetensors";
        if (rank)
            std::snprintf(name, sizeof(name), "%s_%06ld_rank%d.%s", kind.c_str(), step, *rank, ext);
        else
            std::snprintf(name, sizeof(name), "%s_%06ld.%s", kind.c_str(), step, ext);
        return dir / name;
    }

    void save_checkpoint(const fs::path &dir, int64_t step, const GPTImpl &model, const MuonAdamW *optimizer,
                         const nlohmann::json &meta, int rank) {
        fs::create_directories(dir);
        if (rank == 0) {
            safetensors::save(checkpoint_file(dir, "model", step), model.state_dict());
            std::ofstream(checkpoint_file(dir, "meta", step)) << meta.dump(2);
        }
        if (optimizer) {
            safetensors::Metadata metadata;
            const auto state = optimizer->state_dict(metadata);
            safetensors::save(checkpoint_file(dir, "optim", step, rank), state, metadata);
        }
    }

    Checkpoint load_checkpoint(const fs::path &dir, int64_t step, torch::Device device, bool load_optimizer, int rank) {
        Checkpoint c;
        c.model = safetensors::load(checkpoint_file(dir, "model", step), device);
        if (load_optimizer) {
            const auto path = checkpoint_file(dir, "optim", step, rank);
            c.optimizer = safetensors::load(path, device);
            c.optimizer_metadata = safetensors::load_metadata(path);
        }
        std::ifstream in(checkpoint_file(dir, "meta", step));
        c.meta = nlohmann::json::parse(in);
        return c;
    }

    std::optional<int64_t> last_checkpoint_step(const fs::path &dir) {
        if (!fs::is_directory(dir))
            return std::nullopt;
        static const std::regex pattern(R"(model_(\d+)\.safetensors)");
        std::optional<int64_t> last;
        for (const auto &entry: fs::directory_iterator(dir)) {
            std::smatch m;
            const auto name = entry.path().filename().string();
            if (std::regex_match(name, m, pattern))
                last = std::max(last.value_or(-1), std::stol(m[1].str()));
        }
        return last;
    }

    nlohmann::json config_to_json(const GPTConfig &c) {
        return {{"sequence_len", c.sequence_len}, {"vocab_size", c.vocab_size}, {"n_layer", c.n_layer},
                {"n_head", c.n_head}, {"n_kv_head", c.n_kv_head}, {"n_embd", c.n_embd},
                {"window_pattern", c.window_pattern}};
    }

    GPTConfig config_from_json(const nlohmann::json &j) {
        GPTConfig c;
        c.sequence_len = j.at("sequence_len");
        c.vocab_size = j.at("vocab_size");
        c.n_layer = j.at("n_layer");
        c.n_head = j.at("n_head");
        c.n_kv_head = j.at("n_kv_head");
        c.n_embd = j.at("n_embd");
        c.window_pattern = j.value("window_pattern", "L"); // missing in old checkpoints
        return c;
    }

} // namespace nanochat
