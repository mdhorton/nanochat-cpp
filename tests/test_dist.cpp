// 2-GPU training (ZeRO-2 MuonAdamW, strided data, all-reduced eval) vs Python under torchrun
// (tools/export_train_golden.py train_ddp2). Runs dist_worker, one process per GPU.
#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

#include "nanochat/train/safetensors.h"
#include "test_env.h"

using namespace nanochat;
namespace fs = std::filesystem;

TEST(DistGolden, TwoGpusMatchPython) {
  const auto golden = test_env().golden_dir / "train" / "train_ddp2.json";
  if (!fs::exists(golden))
    GTEST_SKIP() << "missing " << golden << " (pixi run export-train-golden-ddp)";
  if (torch::cuda::device_count() < 2)
    GTEST_SKIP() << "needs 2 GPUs";
  const auto out = fs::temp_directory_path() / ("nanochat_dist_" + std::to_string(getpid()));
  const int port = 20000 + getpid() % 20000;
  const auto cmd = std::string(NANOCHAT_DIST_WORKER) + " --nproc 2 --golden " + golden.string() + " --base-dir " +
                   test_env().base_dir.string() + " --out " + out.string() + " --master-port " + std::to_string(port);
  ASSERT_EQ(std::system(cmd.c_str()), 0) << cmd;

  std::ifstream want_in(golden), got_in(out.string() + ".json");
  const auto want = nlohmann::json::parse(want_in), got = nlohmann::json::parse(got_in);
  ASSERT_EQ(got["losses"].size(), want["losses"].size());
  for (size_t i = 0; i < got["losses"].size(); ++i)
    EXPECT_NEAR(got["losses"][i].get<double>(), want["losses"][i].get<double>(), 1e-3) << "step " << i;
  for (const auto& [step, bpb] : want["evals"].items())
    EXPECT_NEAR(got["evals"].at(step).get<double>(), bpb.get<double>(), 1e-4) << "val bpb at step " << step;

  const auto final_state = safetensors::load(out.string() + ".safetensors", torch::kCPU);
  const auto want_state = safetensors::load(golden.parent_path() / "train_ddp2_final.safetensors", torch::kCPU);
  int exact = 0;
  for (const auto& [name, t] : want_state) {
    exact += torch::equal(final_state.at(name), t);
    const double scale = t.to(torch::kFloat64).abs().max().item<double>() + 1e-12;
    const double diff = (final_state.at(name).to(torch::kFloat64) - t.to(torch::kFloat64)).abs().max().item<double>();
    EXPECT_LE(diff / scale, 1e-2) << name;
  }
  RecordProperty("exact_params", std::to_string(exact) + "/" + std::to_string(want_state.size()));
  if (std::getenv("NANOCHAT_TEST_VERBOSE"))
    std::cout << exact << "/" << want_state.size() << " final params bit-identical to Python\n";
  fs::remove(out.string() + ".json");
  fs::remove(out.string() + ".safetensors");
}
