// MuonAdamW: AdamW math vs torch::optim::AdamW, group setup, and parity with Python steps.
#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "nanochat/train/optim.h"
#include "test_env.h"

using namespace nanochat;
namespace fs = std::filesystem;
namespace st = nanochat::safetensors;

namespace {

fs::path golden(const std::string& name) {
  return test_env().golden_dir / "train" / name;
}

nlohmann::json read_json(const fs::path& path) {
  std::ifstream in(path);
  return nlohmann::json::parse(in);
}

GPTConfig golden_config() {
  const auto j = read_json(golden("gpt_config.json"));
  return {
        .sequence_len = j["sequence_len"],
        .vocab_size = j["vocab_size"],
        .n_layer = j["n_layer"],
        .n_head = j["n_head"],
        .n_kv_head = j["n_kv_head"],
        .n_embd = j["n_embd"],
        .window_pattern = j["window_pattern"]};
}

double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  return (a.to(torch::kFloat64) - b.to(torch::kFloat64)).abs().max().item<double>();
}

} // namespace

// Same check as nanochat tests/test_optim.py: the fused AdamW matches torch's AdamW on fp32 params.
TEST(Optim, AdamWMatchesTorch) {
  torch::manual_seed(0);
  auto opts = torch::TensorOptions().device(torch::kCUDA);
  auto p = torch::randn({64, 32}, opts).requires_grad_();
  auto ref = p.detach().clone().requires_grad_();
  MuonAdamW ours(
        {{.kind = OptimGroup::Kind::AdamW,
          .name = "p",
          .params = {p},
          .lr = 0.01,
          .initial_lr = 0.01,
          .weight_decay = 0.1,
          .beta1 = 0.8,
          .beta2 = 0.95,
          .eps = 1e-10}});
  torch::optim::AdamW theirs({ref}, torch::optim::AdamWOptions(0.01).betas({0.8, 0.95}).eps(1e-10).weight_decay(0.1));
  for (int i = 0; i < 5; ++i) {
    auto g = torch::randn({64, 32}, opts);
    p.mutable_grad() = g.clone();
    ref.mutable_grad() = g.clone();
    ours.step();
    theirs.step();
  }
  EXPECT_TRUE(torch::allclose(p, ref, 1e-5, 1e-6)) << max_abs_diff(p, ref);
}

TEST(Optim, Groups) {
  GPT model(
        GPTConfig{.sequence_len = 256, .vocab_size = 1000, .n_layer = 4, .n_head = 4, .n_kv_head = 4, .n_embd = 256});
  auto opt = setup_optimizer(*model);
  std::vector<std::string> names;
  size_t num_params = 0;
  for (const auto& g : opt.groups()) {
    names.push_back(g.name);
    num_params += g.params.size();
  }
  // Muon shapes sorted as tuples: ve_gate (4, 12), c_q/k/v/proj (256, 256), c_proj (256, 1024), c_fc (1024, 256)
  const std::vector<std::string> want = {"lm_head", "wte",       "value_embeds", "resid_lambdas", "x0_lambdas",
                                         "smear",   "muon_4x12", "muon_256x256", "muon_256x1024", "muon_1024x256"};
  EXPECT_EQ(names, want);
  EXPECT_EQ(num_params, model->parameters().size());
}

// Feeds Python's gradients into the C++ optimizer and compares every parameter after each step.
TEST(OptimGolden, StepsMatchPython) {
  if (!fs::exists(golden("optim_config.json")))
    GTEST_SKIP() << "missing " << golden("optim_config.json") << " (run tools/export_train_golden.py)";
  const auto config = read_json(golden("optim_config.json"));
  GPT model(golden_config());
  model->load_state(st::load(golden("gpt_perturbed.safetensors"), torch::kCUDA));
  auto opt = setup_optimizer(
        *model, config["unembedding_lr"], config["embedding_lr"], config["matrix_lr"], config["weight_decay"],
        config["scalar_lr"]);
  auto params = model->named_parameters(true);
  for (int step = 0; step < config["steps"].get<int>(); ++step) {
    const auto grads = st::load(golden("optim_grads_" + std::to_string(step) + ".safetensors"), torch::kCUDA);
    for (auto& item : params)
      item.value().mutable_grad() = grads.at(item.key()).clone();
    opt.step();
    const auto want = st::load(golden("optim_params_" + std::to_string(step) + ".safetensors"), torch::kCUDA);
    for (const auto& item : params) {
      const double diff = max_abs_diff(item.value(), want.at(item.key()));
      if (std::getenv("NANOCHAT_TEST_VERBOSE") && diff > 0)
        std::cout << "step " << step << " " << item.key() << " max diff " << diff << "\n";
      EXPECT_LE(diff, 1e-5) << "step " << step << " " << item.key();
    }
  }
}
