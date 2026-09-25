// Checkpoints: C++ save/load resumes exactly, and a converted Python checkpoint resumes like Python.
#include <gtest/gtest.h>

#include <unistd.h>

#include "nanochat/checkpoint.h"
#include "test_env.h"

using namespace nanochat;
namespace fs = std::filesystem;
namespace st = nanochat::safetensors;

namespace {

    const GPTConfig kTiny{.sequence_len = 256, .vocab_size = 1000, .n_layer = 4, .n_head = 4, .n_kv_head = 4,
                          .n_embd = 256, .window_pattern = "SSSL"};

    MuonAdamW make_optimizer(GPTImpl &model) { return setup_optimizer(model, 0.008, 0.3, 0.02, 0.2, 0.5); }

    void train_step(GPTImpl &model, MuonAdamW &opt, int64_t seed) {
        auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
        auto ids = torch::randint(0, kTiny.vocab_size, {2, kTiny.sequence_len + 1}, gen).to(torch::kCUDA);
        model.forward(ids.slice(1, 0, -1).contiguous(), ids.slice(1, 1).contiguous()).backward();
        opt.step();
        model.zero_grad(true);
    }

    void expect_same_params(const GPTImpl &a, const GPTImpl &b) {
        const auto sa = a.state_dict(), sb = b.state_dict();
        for (const auto &[name, t]: sa)
            EXPECT_TRUE(torch::equal(t, sb.at(name))) << name;
    }

} // namespace

TEST(Checkpoint, FileNames) {
    EXPECT_EQ(checkpoint_file("d", "model", 12).string(), "d/model_000012.safetensors");
    EXPECT_EQ(checkpoint_file("d", "optim", 12, 1).string(), "d/optim_000012_rank1.safetensors");
    EXPECT_EQ(checkpoint_file("d", "meta", 5).string(), "d/meta_000005.json");
}

// Save after 2 steps, reload into a fresh model + optimizer: the 3rd step must be identical.
TEST(Checkpoint, ResumeIsExact) {
    const auto dir = fs::temp_directory_path() / ("nanochat_ckpt_" + std::to_string(::getpid()));
    torch::manual_seed(42);
    GPT model(kTiny);
    model->init_weights();
    model->set_attention(Attention::SDPA); // deterministic backward (FA2's may not be)
    auto opt = make_optimizer(*model);
    train_step(*model, opt, 1);
    train_step(*model, opt, 2);
    save_checkpoint(dir, 2, *model, &opt, {{"step", 2}, {"model_config", config_to_json(kTiny)}});
    EXPECT_EQ(last_checkpoint_step(dir), 2);

    const auto ckpt = load_checkpoint(dir, 2, torch::kCUDA, true);
    GPT resumed(config_from_json(ckpt.meta["model_config"]));
    resumed->load_state(ckpt.model);
    resumed->set_attention(Attention::SDPA);
    auto resumed_opt = make_optimizer(*resumed);
    resumed_opt.load_state_dict(ckpt.optimizer, ckpt.optimizer_metadata);
    expect_same_params(*model, *resumed);

    train_step(*model, opt, 3);
    train_step(*resumed, resumed_opt, 3);
    expect_same_params(*model, *resumed);
    fs::remove_all(dir);
}

TEST(CheckpointGolden, ResumesPythonCheckpoint) {
    const auto dir = test_env().golden_dir / "train";
    if (!fs::exists(dir / "ckpt_params.safetensors"))
        GTEST_SKIP() << "missing " << dir / "ckpt_params.safetensors" << " (run tools/export_train_golden.py)";
    const auto ckpt = load_checkpoint(dir / "ckpt", 2, torch::kCUDA, true);
    GPT model(config_from_json(ckpt.meta["model_config"]));
    model->load_state(ckpt.model);
    auto opt = make_optimizer(*model);
    opt.load_state_dict(ckpt.optimizer, ckpt.optimizer_metadata);

    const auto grads = st::load(dir / "ckpt_grads.safetensors", torch::kCUDA);
    for (auto &item: model->named_parameters(true))
        item.value().mutable_grad() = grads.at(item.key()).clone();
    opt.step();
    const auto want = st::load(dir / "ckpt_params.safetensors", torch::kCUDA);
    for (const auto &[name, t]: model->state_dict())
        EXPECT_TRUE(torch::equal(t, want.at(name))) << name;
}
