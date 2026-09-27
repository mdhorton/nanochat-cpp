// Pre-training loop. Port of scripts/base_train.py (without torch.compile, wandb, CORE eval and sampling).
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "nanochat/model/gpt.h"

namespace nanochat {

struct TrainOptions {
  // model
  int64_t depth = 20, aspect_ratio = 64, head_dim = 128, max_seq_len = 2048;
  std::string window_pattern = "SSSL";
  std::string attention = "fa2";    // fa2, fa2-torch (PyTorch's copy) or sdpa (bit-identical to Python nanochat)
  int64_t loss_chunk_rows = 4096;   // > 0: fused chunked lm_head + loss (never all logits at once); 0: as Python
  bool fp8 = true;                  // FP8 matmuls for training (eval stays bf16)
  std::string fp8_recipe = "mxfp8"; // mxfp8 or tensorwise (as Python)
  bool fused = true;                // fused elementwise kernels; false: Python's op-by-op path
  // training horizon: the first one set wins
  int64_t num_iterations = -1;
  double target_flops = -1, target_param_data_ratio = 8; // Python: 12; speedrun.sh uses 8
  // optimization
  int64_t device_batch_size = 32, total_batch_size = -1; // -1: from scaling laws
  double embedding_lr = 0.3, unembedding_lr = 0.008, weight_decay = 0.28, matrix_lr = 0.02, scalar_lr = 0.5;
  int64_t warmup_steps = 40;
  double warmdown_ratio = 0.65, final_lr_frac = 0.05;
  int64_t resume_from_step = -1;
  // evaluation and checkpoints
  int64_t eval_every = 250, eval_tokens = 80 * 524288, save_every = -1;
  bool save = true; // save at the end (and every save_every)
  // run name: checkpoint dir and wandb name; metrics go to <base_dir>/metrics/<run>/metrics-<timestamp>.jsonl.
  // "dummy" (as base_train.py): checkpoint dir d<depth>, no metrics
  std::string run = "dummy";
  // nsys: NVTX range "profile" around steps [profile_start, profile_start + profile_steps), -1 = off
  int64_t profile_start = -1, profile_steps = 3;
  // paths
  std::filesystem::path base_dir; // data, tokenizer and checkpoints
  // multi-GPU: this process's rank (= GPU index), rank 0 hosts the TCPStore at master_addr:master_port
  int rank = 0, world_size = 1;
  std::string master_addr = "127.0.0.1";
  int master_port = 29500;
  bool verbose = true;
};

// What the scaling laws give for a config (base_train.py "Scaling laws and muP extrapolations").
struct TrainPlan {
  int64_t num_scaling_params = 0; // transformer matrices + lm_head
  int64_t target_tokens = 0;
  double d_ref = 0; // compute-optimal tokens of the d12 reference
  int64_t total_batch_size = 0;
  double batch_lr_scale = 1, weight_decay_scaled = 0;
  int64_t num_iterations = 0, grad_accum_steps = 0;
};

// Parameter counts from the config alone (no allocation), as GPT::num_scaling_params.
ScalingParams count_params(const GPTConfig& config, int64_t pad_vocab_size_to = 64);

TrainPlan plan_training(const TrainOptions& options, int64_t vocab_size, int64_t flops_per_token);

// LR multiplier (warmup, constant, linear warmdown), Muon momentum and Muon weight decay per iteration.
struct Schedules {
  int64_t num_iterations, warmup_steps;
  double warmdown_ratio, final_lr_frac, weight_decay_scaled;

  double lr_multiplier(int64_t it) const;
  double muon_momentum(int64_t it) const;
  double weight_decay(int64_t it) const;
};

// BF16 peak FLOPs by GPU name (nanochat common.get_peak_flops); infinity when unknown (MFU shows 0).
double peak_flops(const std::string& device_name);

// One line per optimizer step, and one per validation.
struct StepInfo {
  int64_t step;
  double train_loss; // last micro-batch loss
  double lr_multiplier;
  double dt; // seconds
};

struct TrainCallbacks {
  std::function<void(const StepInfo&)> on_step;
  std::function<void(int64_t step, double val_bpb)> on_eval;
  std::function<void(GPTImpl& model)> on_end;
};

// Trains per options; returns the final validation bpb (if evaluated).
std::optional<double> train(const TrainOptions& options, const TrainCallbacks& callbacks = {});

} // namespace nanochat
