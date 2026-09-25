// base_train: scaling rules, schedules, and a tiny training run vs Python (tools/export_train_golden.py).
#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "nanochat/train/safetensors.h"
#include "nanochat/train/trainer.h"
#include "test_env.h"

using namespace nanochat;
namespace fs = std::filesystem;

namespace {

    fs::path golden(const std::string &name) { return test_env().golden_dir / "train" / name; }

    TrainOptions options_from_json(const nlohmann::json &j) {
        TrainOptions o;
        o.depth = j["depth"];
        o.aspect_ratio = j["aspect_ratio"];
        o.head_dim = j["head_dim"];
        o.max_seq_len = j["max_seq_len"];
        o.window_pattern = j["window_pattern"];
        o.num_iterations = j["num_iterations"];
        o.target_flops = j["target_flops"];
        o.target_param_data_ratio = j["target_param_data_ratio"];
        o.device_batch_size = j["device_batch_size"];
        o.total_batch_size = j["total_batch_size"];
        o.embedding_lr = j["embedding_lr"];
        o.unembedding_lr = j["unembedding_lr"];
        o.weight_decay = j["weight_decay"];
        o.matrix_lr = j["matrix_lr"];
        o.scalar_lr = j["scalar_lr"];
        o.warmup_steps = j["warmup_steps"];
        o.warmdown_ratio = j["warmdown_ratio"];
        o.final_lr_frac = j["final_lr_frac"];
        o.eval_every = j["eval_every"];
        o.eval_tokens = j["eval_tokens"];
        return o;
    }

    class TrainGolden : public testing::Test {
    protected:
        void SetUp() override {
            if (!fs::exists(golden("train.json")))
                GTEST_SKIP() << "missing " << golden("train.json") << " (run tools/export_train_golden.py)";
            std::ifstream in(golden("train.json"));
            golden_ = nlohmann::json::parse(in);
            options_ = options_from_json(golden_["options"]);
            options_.base_dir = test_env().base_dir;
            options_.save = false;
            options_.attention = "sdpa"; // as Python on this GPU
            options_.loss_chunk_rows = 0;
            options_.verbose = std::getenv("NANOCHAT_TEST_VERBOSE") != nullptr;
        }
        nlohmann::json golden_;
        TrainOptions options_;
    };

} // namespace

TEST(Train, CountParamsMatchesModel) {
    for (const auto &config: {config_from_depth(4, 32768, 64, 64, 256), config_from_depth(12, 32768)}) {
        GPT model(config);
        const auto a = count_params(config), b = model->num_scaling_params();
        EXPECT_EQ(a.total, b.total);
        EXPECT_EQ(a.transformer_matrices, b.transformer_matrices);
        EXPECT_EQ(a.value_embeds, b.value_embeds);
    }
}

// speedrun.sh: --depth=24 --target-param-data-ratio=8 --device-batch-size=16 on 8 GPUs
TEST(Train, SpeedrunPlan) {
    TrainOptions o;
    o.depth = 24;
    o.target_param_data_ratio = 8;
    o.device_batch_size = 16;
    o.world_size = 8;
    const auto plan = plan_training(o, 32768, 0);
    EXPECT_EQ(plan.total_batch_size, 1 << 20);
    EXPECT_NEAR(plan.batch_lr_scale, std::sqrt(2.0), 1e-12);
    EXPECT_EQ(plan.grad_accum_steps, 4);
    EXPECT_EQ(plan.num_iterations, plan.target_tokens / (1 << 20));
}

TEST_F(TrainGolden, PlanAndSchedulesMatchPython) {
    const auto &p = golden_["plan"];
    const auto plan = plan_training(options_, 32768, 0);
    EXPECT_EQ(plan.num_scaling_params, p["num_scaling_params"].get<int64_t>());
    EXPECT_EQ(plan.target_tokens, p["target_tokens"].get<int64_t>());
    EXPECT_EQ(plan.d_ref, p["d_ref"].get<double>());
    EXPECT_EQ(plan.total_batch_size, p["total_batch_size"].get<int64_t>());
    EXPECT_EQ(plan.batch_lr_scale, p["batch_lr_scale"].get<double>());
    EXPECT_EQ(plan.weight_decay_scaled, p["weight_decay_scaled"].get<double>());
    EXPECT_EQ(plan.num_iterations, p["num_iterations"].get<int64_t>());
    EXPECT_EQ(plan.grad_accum_steps, p["grad_accum_steps"].get<int64_t>());
    const Schedules s{plan.num_iterations, options_.warmup_steps, options_.warmdown_ratio, options_.final_lr_frac,
                      plan.weight_decay_scaled};
    for (int64_t it = 0; it < plan.num_iterations; ++it) {
        const auto &want = golden_["schedule"][it];
        EXPECT_EQ(s.lr_multiplier(it), want[0].get<double>()) << "step " << it;
        EXPECT_EQ(s.muon_momentum(it), want[1].get<double>()) << "step " << it;
        EXPECT_EQ(s.weight_decay(it), want[2].get<double>()) << "step " << it;
    }
}

TEST_F(TrainGolden, LossCurveMatchesPython) {
    std::vector<double> losses;
    std::map<int64_t, double> evals;
    safetensors::TensorMap final_state;
    train(options_, {.on_step = [&](const StepInfo &s) { losses.push_back(s.train_loss); },
                     .on_eval = [&](int64_t step, double bpb) { evals[step] = bpb; },
                     .on_end = [&](GPTImpl &model) { final_state = model.state_dict(); }});

    const auto &want_losses = golden_["losses"];
    ASSERT_EQ(losses.size(), want_losses.size());
    for (size_t i = 0; i < losses.size(); ++i)
        EXPECT_NEAR(losses[i], want_losses[i].get<double>(), 1e-3) << "step " << i;
    for (const auto &[step, bpb]: golden_["evals"].items())
        EXPECT_NEAR(evals.at(std::stol(step)), bpb.get<double>(), 1e-4) << "val bpb at step " << step;

    const auto want = safetensors::load(golden("train_final.safetensors"), torch::kCUDA);
    int exact = 0;
    for (const auto &[name, t]: want) {
        exact += torch::equal(final_state.at(name), t);
        const double scale = t.to(torch::kFloat64).abs().max().item<double>() + 1e-12;
        const double diff = (final_state.at(name).to(torch::kFloat64) - t.to(torch::kFloat64)).abs().max().item<double>();
        EXPECT_LE(diff / scale, 1e-2) << name;
    }
    RecordProperty("exact_params", std::to_string(exact) + "/" + std::to_string(want.size()));
    if (options_.verbose)
        std::cout << exact << "/" << want.size() << " final params bit-identical to Python\n";
}
