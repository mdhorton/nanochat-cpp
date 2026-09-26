// GPT model: structure tests, plus parity with Python from tools/export_train_golden.py (skipped when missing).
#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "nanochat/model/gpt.h"
#include "test_env.h"

using namespace nanochat;
namespace fs = std::filesystem;
namespace st = nanochat::safetensors;

namespace {

fs::path golden(const std::string& name) {
  return test_env().golden_dir / "train" / name;
}

#define REQUIRE_TRAIN_GOLDEN(name)                                                                                     \
  if (!fs::exists(golden(name)))                                                                                       \
  GTEST_SKIP() << "missing " << golden(name) << " (run tools/export_train_golden.py)"

// The parity config (see tools/export_train_golden.py).
GPTConfig tiny_config() {
  return {
        .sequence_len = 256,
        .vocab_size = 1000,
        .n_layer = 4,
        .n_head = 4,
        .n_kv_head = 4,
        .n_embd = 256,
        .window_pattern = "SSSL"};
}

GPTConfig golden_config() {
  std::ifstream in(golden("gpt_config.json"));
  const auto j = nlohmann::json::parse(in);
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

TEST(GPT, ConfigFromDepth) {
  auto c = config_from_depth(12, 32768);
  EXPECT_EQ(c.n_embd, 768);
  EXPECT_EQ(c.n_head, 6);
  EXPECT_EQ(config_from_depth(20, 32768).n_embd, 1280);
  EXPECT_EQ(config_from_depth(24, 32768).n_head, 12);
  c = config_from_depth(6, 32768, 64, 64, 512, "L"); // runcpu.sh
  EXPECT_EQ(c.n_embd, 384);
  EXPECT_EQ(c.n_head, 6);
  EXPECT_EQ(c.head_dim(), 64);
}

TEST(GPT, WindowSizes) {
  GPTConfig c; // 12 layers, T = 2048, SSSL
  const auto w = window_sizes(c);
  const std::vector<int64_t> want = {512, 512, 512, 2048, 512, 512, 512, 2048, 512, 512, 512, 2048};
  EXPECT_EQ(w, want);
  c.window_pattern = "S";
  EXPECT_EQ(window_sizes(c).back(), 2048); // last layer always long
  c.sequence_len = 256;
  EXPECT_EQ(window_sizes(c).front(), 128); // rounded up to 128
}

TEST(GPT, ParameterNames) {
  GPT model(tiny_config());
  std::vector<std::string> names;
  for (const auto& item : model->named_parameters(true))
    names.push_back(item.key());
  for (const std::string want :
       {"transformer.wte.weight", "transformer.h.0.attn.c_q.weight", "transformer.h.1.attn.ve_gate.weight",
        "transformer.h.3.mlp.c_proj.weight", "lm_head.weight", "resid_lambdas", "x0_lambdas", "smear_gate.weight",
        "smear_lambda", "backout_lambda", "value_embeds.1.weight", "value_embeds.3.weight"})
    EXPECT_NE(std::find(names.begin(), names.end(), want), names.end()) << want;
  EXPECT_EQ(
        std::count_if(
              names.begin(), names.end(),
              [](const auto& n) {
                return n.find("ve_gate") != n.npos;
              }),
        2);
  EXPECT_EQ(model->lm_head->weight.size(0), 1024); // vocab padded to a multiple of 64
}

TEST(GPT, ForwardBackward) {
  torch::manual_seed(0);
  GPT model(tiny_config());
  model->init_weights();
  EXPECT_EQ(model->transformer->wte->weight.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(model->lm_head->weight.scalar_type(), torch::kFloat32);
  auto idx = torch::randint(0, 1000, {2, 256}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  auto logits = model->forward(idx);
  EXPECT_EQ(logits.sizes(), (std::vector<int64_t>{2, 256, 1000}));
  EXPECT_LE(logits.abs().max().item<float>(), 15.0f); // softcap
  auto loss = model->forward(idx, idx.roll(-1, 1));
  EXPECT_NEAR(loss.item<float>(), std::log(1000.0), 0.5); // near-uniform at init
  loss.backward();
  for (const auto& item : model->named_parameters(true))
    ASSERT_TRUE(item.value().grad().defined()) << item.key();
}

TEST(GPT, FlopsAndScalingParams) {
  GPT model(config_from_depth(12, 32768), torch::kCUDA);
  const auto s = model->num_scaling_params();
  EXPECT_EQ(s.wte, 32768 * 768);
  EXPECT_EQ(s.lm_head, 32768 * 768);
  EXPECT_EQ(s.value_embeds, 6 * 32768 * 768);
  EXPECT_EQ(s.scalars, 12 + 12 + 24 + 1 + 1);
  // 12 blocks of q, k, v, proj (4 d^2) and mlp (8 d^2), plus 6 ve_gates (6 x 12)
  EXPECT_EQ(s.transformer_matrices, 12 * 12 * 768 * 768 + 6 * 6 * 12);
  const int64_t matmul = s.transformer_matrices + s.lm_head + 24;
  const int64_t attn = 12 * 6 * 128 * (9 * 512 + 3 * 2048);
  EXPECT_EQ(model->estimate_flops(), 6 * matmul + attn);
}

// FlashAttention-2 (sliding windows in S layers) vs the SDPA path, within bf16 tolerance.
TEST(GPT, Fa2MatchesSdpa) {
  torch::manual_seed(0);
  GPT model(tiny_config()); // T = 256, S layers use a 128 window
  model->init_weights();
  {
    torch::NoGradGuard no_grad; // perturb so projections and gates are non-zero
    for (auto& p : model->parameters())
      p.add_((0.02 * torch::randn(p.sizes(), p.options().dtype(torch::kFloat32))).to(p.scalar_type()));
  }
  auto ids = torch::randint(0, 1000, {2, 257}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  auto x = ids.slice(1, 0, -1).contiguous(), y = ids.slice(1, 1).contiguous();
  auto run = [&](Attention attention) {
    model->set_attention(attention);
    model->zero_grad(true);
    auto loss = model->forward(x, y);
    loss.backward();
    std::map<std::string, torch::Tensor> grads;
    for (const auto& item : model->named_parameters(true))
      grads[item.key()] = item.value().grad().clone();
    return std::pair{loss.item<double>(), grads};
  };
  const auto [loss_fa2, grads_fa2] = run(Attention::FA2);
  const auto [loss_sdpa, grads_sdpa] = run(Attention::SDPA);
  if (std::getenv("NANOCHAT_TEST_VERBOSE")) {
    const auto [loss_fa2b, grads_fa2b] = run(Attention::FA2);
    for (const auto& [name, g] : grads_sdpa) {
      const double scale = g.to(torch::kFloat64).abs().max().item<double>() + 1e-12;
      std::cout << name << " fa2-sdpa " << max_abs_diff(grads_fa2.at(name), g) / scale << " fa2-fa2 "
                << max_abs_diff(grads_fa2.at(name), grads_fa2b.at(name)) / scale << "\n";
    }
  }
  EXPECT_NEAR(loss_fa2, loss_sdpa, 1e-3);
  for (const auto& [name, g] : grads_sdpa) {
    const double scale = g.to(torch::kFloat64).abs().max().item<double>() + 1e-12;
    // scalar grads sum over every position and layer with cancellation, so bf16 differences show up more
    const double tol = g.numel() < 100 ? 0.2 : 3e-2;
    EXPECT_LE(max_abs_diff(grads_fa2.at(name), g) / scale, tol) << name;
  }
}

// Same seed, same draw order as Python init_weights => identical weights.
TEST(GPTGolden, InitMatchesPython) {
  REQUIRE_TRAIN_GOLDEN("gpt_init.safetensors");
  torch::manual_seed(42);
  GPT model(golden_config());
  model->init_weights();
  const auto want = st::load(golden("gpt_init.safetensors"), torch::kCUDA);
  const auto got = model->state_dict();
  ASSERT_EQ(got.size(), want.size());
  for (const auto& [name, t] : want) {
    ASSERT_TRUE(got.contains(name)) << name;
    EXPECT_EQ(got.at(name).scalar_type(), t.scalar_type()) << name;
    EXPECT_TRUE(torch::equal(got.at(name), t)) << name << " max diff " << max_abs_diff(got.at(name), t);
  }
}

TEST(GPTGolden, ForwardBackwardMatchPython) {
  REQUIRE_TRAIN_GOLDEN("gpt_outputs.safetensors");
  GPT model(golden_config());
  model->set_attention(Attention::SDPA); // as Python on this GPU
  model->load_state(st::load(golden("gpt_perturbed.safetensors"), torch::kCUDA));
  const auto batch = st::load(golden("gpt_batch.safetensors"), torch::kCUDA);
  const auto want = st::load(golden("gpt_outputs.safetensors"), torch::kCUDA);

  {
    torch::NoGradGuard no_grad;
    const auto logits = model->forward(batch.at("idx"));
    EXPECT_LE(max_abs_diff(logits, want.at("logits")), 1e-2);
    if (std::getenv("NANOCHAT_TEST_VERBOSE"))
      std::cout << "logits max diff " << max_abs_diff(logits, want.at("logits")) << "\n";
    RecordProperty("logits_max_diff", std::to_string(max_abs_diff(logits, want.at("logits"))));
  }
  auto loss = model->forward(batch.at("idx"), batch.at("targets"));
  EXPECT_NEAR(loss.item<double>(), want.at("loss").item<double>(), 1e-4);
  if (std::getenv("NANOCHAT_TEST_VERBOSE"))
    std::cout << "loss " << loss.item<double>() << " vs " << want.at("loss").item<double>() << "\n";
  loss.backward();
  for (const auto& item : model->named_parameters(true)) {
    const auto& g = want.at("grad." + item.key());
    const auto& got = item.value().grad();
    ASSERT_TRUE(got.defined()) << item.key();
    // bf16 matmuls: compare relative to the gradient's scale
    const double scale = g.to(torch::kFloat64).abs().max().item<double>() + 1e-12;
    const double rel = max_abs_diff(got, g) / scale;
    if (std::getenv("NANOCHAT_TEST_VERBOSE"))
      std::cout << item.key() << " rel diff " << rel << " (scale " << scale << ")\n";
    EXPECT_LE(rel, 2e-2) << item.key();
  }
}

TEST(GPTGolden, Fp8ForwardBackwardMatchPython) {
  REQUIRE_TRAIN_GOLDEN("fp8_outputs.safetensors");
  GPT model(golden_config());
  model->set_attention(Attention::SDPA);
  model->load_state(st::load(golden("gpt_perturbed.safetensors"), torch::kCUDA));
  EXPECT_EQ(model->set_fp8(true), 4 * 6 + 1); // c_q, c_k, c_v, c_proj, c_fc, mlp c_proj per layer, and lm_head
  const auto batch = st::load(golden("gpt_batch.safetensors"), torch::kCUDA);
  const auto want = st::load(golden("fp8_outputs.safetensors"), torch::kCUDA);
  auto loss = model->forward(batch.at("idx"), batch.at("targets"));
  EXPECT_EQ(loss.item<float>(), want.at("loss").item<float>());
  loss.backward();
  int exact = 0, total = 0;
  for (const auto& item : model->named_parameters(true)) {
    const auto& g = want.at("grad." + item.key());
    const auto& got = item.value().grad();
    ASSERT_TRUE(got.defined()) << item.key();
    EXPECT_EQ(got.scalar_type(), item.value().scalar_type()) << item.key();
    exact += torch::equal(got, g);
    ++total;
    const double scale = g.to(torch::kFloat64).abs().max().item<double>() + 1e-12;
    EXPECT_LE(max_abs_diff(got, g) / scale, 2e-2) << item.key();
  }
  RecordProperty("exact_grads", std::to_string(exact) + "/" + std::to_string(total));
  if (std::getenv("NANOCHAT_TEST_VERBOSE"))
    std::cout << exact << "/" << total << " grads bit-identical to Python\n";
}
