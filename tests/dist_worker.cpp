// The tiny golden training run on several GPUs, for test_dist.cpp. Rank 0 writes <out>.json (losses, val bpb) and
// <out>.safetensors (final weights).
#include <fstream>
#include <iostream>
#include <map>

#include "nanochat/common.h"
#include "nanochat/flags.h"
#include "nanochat/train/dist.h"
#include "nanochat/train/safetensors.h"
#include "train_golden.h"

using namespace nanochat;

static int run(int argc, char** argv) {
  Flags flags(argc, argv, "Tiny multi-GPU training run for the dist golden test");
  const auto golden = flags.str("golden", "", "train_ddp<N>.json from tools/export_train_golden.py");
  const auto base_dir = flags.str("base-dir", default_base_dir().string(), "nanochat data directory");
  const auto out = flags.str("out", "", "output path prefix");
  const auto nproc = static_cast<int>(flags.i64("nproc", 2, "number of GPUs"));
  const auto rank = static_cast<int>(flags.i64("rank", -1, "internal: set by the launcher"));
  const auto master_port = static_cast<int>(flags.i64("master-port", 29500, "rendezvous port"));
  flags.done();
  if (rank < 0)
    return launch_ranks({argv + 1, argv + argc}, nproc);

  std::ifstream in(golden);
  const auto j = nlohmann::json::parse(in);
  auto o = options_from_json(j["options"]);
  o.base_dir = base_dir;
  o.save = false;
  o.attention = "sdpa"; // as Python on this GPU
  o.loss_chunk_rows = 0;
  o.fused = false;
  o.cublaslt_workspace_mb = 0; // torch's default: the same GEMM algorithms as Python
  o.verbose = false;
  o.rank = rank;
  o.world_size = nproc;
  o.master_port = master_port;

  std::vector<double> losses;
  std::map<std::string, double> evals;
  safetensors::TensorMap final_state;
  train(o, {.on_step =
                  [&](const StepInfo& s) {
                    losses.push_back(s.train_loss);
                  },
            .on_eval =
                  [&](int64_t step, double bpb) {
                    evals[std::to_string(step)] = bpb;
                  },
            .on_end =
                  [&](GPTImpl& model) {
                    final_state = model.state_dict();
                  }});
  if (rank == 0) {
    std::ofstream(out + ".json") << nlohmann::json{{"losses", losses}, {"evals", evals}}.dump(1);
    safetensors::save(out + ".safetensors", final_state);
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  }
  catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
