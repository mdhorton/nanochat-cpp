// Pre-training loop. Port of scripts/base_train.py (without torch.compile, CORE eval and sampling; wandb via
// tools/wandb_upload.py).
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "nanochat/model/gpt.h"

namespace nanochat {

struct TrainOptions {
  // model
  int64_t depth = 20, aspect_ratio = 64, head_dim = 128, max_seq_len = 2048;
  int64_t seed = 42; // weight init and NVFP4 stochastic rounding
  std::string window_pattern = "SSSL";
  std::string attention = "mx";       // mx (MXFP8 fwd + bwd), bf16 (ours), bf16mx (MXFP8 bwd), fa2, sdpa (as Python)
  int64_t loss_chunk_rows = 4096;     // > 0: fused chunked lm_head + loss (never all logits at once); 0: as Python
  bool fp8 = true;                    // FP8 matmuls for training (eval stays bf16)
  std::string fp8_recipe = "mxfp8";   // mxfp8 or tensorwise (as Python)
  std::string gemm = "cutlass";       // MXFP8 GEMMs: cutlass or cublas (cuBLASLt)
  bool fused = true;                  // fused elementwise kernels; false: Python's op-by-op path
  int64_t cublaslt_workspace_mb = 32; // cuBLASLt workspace per stream; its split-K GEMMs need a few MB (torch: 1)
  // simulated NVFP4 (nvfp4_sim.h; slow, for numerics) for these GEMMs of the blocks' Linears: comma list of fwd, dgrad,
  // wgrad; empty: off. Needs fp8 with mxfp8, which the other GEMMs, blocks and lm_head keep.
  std::string nvfp4;
  std::string nvfp4_rht = "wgrad";      // GEMMs with a random Hadamard transform along K
  std::string nvfp4_sr = "dgrad,wgrad"; // GEMMs whose gradient operand rounds stochastically
  std::string nvfp4_eden; // GEMMs with MS-EDEN operands (dgrad, wgrad), instead of their rht / sr; real: dgrad
  // simulated MS-EDEN ablations: the rotation / correction group, signs fixed per step (per Linear and GEMM) instead
  // of per GEMM call, and Linears (comma list: c_fc, mlp.c_proj, ...) kept on rht / sr
  int64_t nvfp4_eden_group = 128;
  bool nvfp4_eden_fixed_signs = false;
  std::string nvfp4_eden_skip;
  bool nvfp4_weight_2d = true;
  int64_t nvfp4_skip_first = 0, nvfp4_skip_last = 0; // blocks kept MXFP8 (sim: all GEMMs; nvfp4_fwd: forward)
  // real NVFP4 weight / input gradients for the Linears but lm_head (nvfp4.h), with nvfp4_rht / nvfp4_sr's wgrad /
  // dgrad (no rht) and seed; forward GEMMs for the blocks' Linears (16x16 weight blocks, round to nearest)
  bool nvfp4_wgrad = false, nvfp4_dgrad = false, nvfp4_fwd = false;
  // MXFP8 from a step on: the NVFP4 forward GEMMs (nvfp4_fwd_until) or all NVFP4 GEMMs (nvfp4_until), real or
  // simulated. "warmdown": its first step; a fraction of the steps; empty: never
  std::string nvfp4_fwd_until, nvfp4_until;
  // training horizon: the first one set wins
  int64_t num_iterations = -1;
  double target_flops = -1, target_param_data_ratio = 8; // Python: 12; speedrun.sh uses 8
  // optimization
  int64_t device_batch_size = 32, total_batch_size = -1; // -1: from scaling laws
  // micro-steps per step for each rank (sum = world_size * grad_accum_steps), to balance unequal GPUs; empty: even
  std::vector<int64_t> rank_micro_steps;
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
  bool wandb = false; // rank 0 uploads the metrics live with tools/wandb_upload.py (not with run "dummy")
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
  int64_t rank_accum_steps = 0; // this rank's micro-steps: grad_accum_steps unless rank_micro_steps
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
