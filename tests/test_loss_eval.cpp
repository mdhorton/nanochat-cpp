// Validation bits per byte vs Python (tools/export_train_golden.py).
#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "nanochat/train/loss_eval.h"
#include "test_env.h"

using namespace nanochat;
namespace fs = std::filesystem;

TEST(LossEvalGolden, BpbMatchesPython) {
  const auto dir = test_env().golden_dir / "train";
  if (!fs::exists(dir / "bpb.json"))
    GTEST_SKIP() << "missing " << dir / "bpb.json" << " (run tools/export_train_golden.py)";
  std::ifstream in(dir / "bpb.json");
  const auto j = nlohmann::json::parse(in);
  const auto& c = j["config"];
  GPT model(
        GPTConfig{
              .sequence_len = c["sequence_len"],
              .vocab_size = c["vocab_size"],
              .n_layer = c["n_layer"],
              .n_head = c["n_head"],
              .n_kv_head = c["n_kv_head"],
              .n_embd = c["n_embd"],
              .window_pattern = c["window_pattern"]});
  model->set_attention(Attention::SDPA); // as Python on this GPU
  model->load_state(safetensors::load(dir / "bpb_model.safetensors", torch::kCUDA));

  const auto tokenizer = Tokenizer::load(test_env().golden_dir / "tokenizer.json");
  DataLoader val(tokenizer, j["B"], c["sequence_len"], Split::Val, test_env().base_dir / "base_data_climbmix");
  const auto token_bytes = load_token_bytes(test_env().golden_dir / "token_bytes.bin");
  const double bpb = evaluate_bpb(*model, val, j["steps"], token_bytes);
  EXPECT_NEAR(bpb, j["bpb"].get<double>(), 1e-6 * j["bpb"].get<double>());
}
