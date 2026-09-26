// Pretrain a base model. Port of scripts/base_train.py (no CORE eval or sampling yet).
// --nproc N runs one process per GPU (as torchrun --nproc_per_node=N).
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "nanochat/common.h"
#include "nanochat/flags.h"
#include "nanochat/train/dist.h"
#include "nanochat/train/trainer.h"

using namespace nanochat;

static int run(int argc, char** argv) {
  Flags flags(argc, argv, "Pretrain a base model");
  TrainOptions o;
  o.base_dir = flags.str("base-dir", default_base_dir().string(), "nanochat data directory");
  // model
  o.depth = flags.i64("depth", o.depth, "depth of the Transformer model");
  o.aspect_ratio = flags.i64("aspect-ratio", o.aspect_ratio, "model_dim = depth * aspect_ratio");
  o.head_dim = flags.i64("head-dim", o.head_dim, "target head dimension for attention");
  o.max_seq_len = flags.i64("max-seq-len", o.max_seq_len, "max context length");
  o.window_pattern = flags.str(
        "window-pattern", o.window_pattern, "sliding window pattern tiled across layers: L=full, S=quarter context");
  o.attention = flags.str("attention", o.attention, "fa2 (FlashAttention-2) or sdpa (bit-identical to Python)");
  o.loss_chunk_rows = flags.i64(
        "loss-chunk-rows", o.loss_chunk_rows, "rows per chunk of the fused lm_head + loss (0 = unchunked, as Python)");
  o.fp8 = flags.boolean("fp8", o.fp8, "FP8 training (tensorwise scaling; eval stays bf16)");
  o.fused = flags.boolean("fused", o.fused, "fused elementwise CUDA kernels (false = op by op, as Python)");
  // horizon
  o.num_iterations = flags.i64("num-iterations", o.num_iterations, "explicit number of steps (-1 = disable)");
  o.target_flops = flags.f64("target-flops", o.target_flops, "steps to reach target FLOPs (-1 = disable)");
  o.target_param_data_ratio = flags.f64(
        "target-param-data-ratio", o.target_param_data_ratio, "steps for this data:param ratio (-1 = disable)");
  // optimization
  o.device_batch_size = flags.i64("device-batch-size", o.device_batch_size, "per-device batch size (reduce on OOM)");
  o.total_batch_size = flags.i64("total-batch-size", o.total_batch_size, "total batch size in tokens (-1 = auto)");
  o.embedding_lr = flags.f64("embedding-lr", o.embedding_lr, "learning rate for embeddings (AdamW)");
  o.unembedding_lr = flags.f64("unembedding-lr", o.unembedding_lr, "learning rate for lm_head (AdamW)");
  o.weight_decay = flags.f64("weight-decay", o.weight_decay, "cautious weight decay for Muon");
  o.matrix_lr = flags.f64("matrix-lr", o.matrix_lr, "learning rate for matrices (Muon)");
  o.scalar_lr = flags.f64("scalar-lr", o.scalar_lr, "learning rate for scalars (AdamW)");
  o.warmup_steps = flags.i64("warmup-steps", o.warmup_steps, "LR warmup steps");
  o.warmdown_ratio = flags.f64("warmdown-ratio", o.warmdown_ratio, "fraction of steps for LR warmdown");
  o.final_lr_frac = flags.f64("final-lr-frac", o.final_lr_frac, "final LR as a fraction of the initial LR");
  o.resume_from_step = flags.i64("resume-from-step", o.resume_from_step, "resume from this checkpoint (-1 = no)");
  // evaluation and output
  o.eval_every = flags.i64(
        "eval-every", o.eval_every, "evaluate val bpb every N steps and at the end (-1 = end only, 0 = never)");
  o.eval_tokens = flags.i64("eval-tokens", o.eval_tokens, "tokens to evaluate val bpb on");
  o.profile_start = flags.i64(
        "profile-start", o.profile_start, "first step in the NVTX range 'profile', for nsys --nvtx-capture (-1 = off)");
  o.profile_steps = flags.i64("profile-steps", o.profile_steps, "number of steps captured");
  o.save_every = flags.i64("save-every", o.save_every, "save a checkpoint every N steps (-1 = only at the end)");
  o.save = flags.boolean("save", o.save, "save checkpoints");
  o.run = flags.str(
        "run", o.run,
        "run name: checkpoint dir, metrics/<run>/metrics-<timestamp>.jsonl, wandb name ('dummy' = d<depth>, no "
        "metrics)");
  // distributed
  const auto nproc = static_cast<int>(flags.i64("nproc", 1, "number of GPUs (one process each)"));
  const auto rank = static_cast<int>(flags.i64("rank", -1, "internal: set by --nproc for each process"));
  o.master_addr = flags.str("master-addr", o.master_addr, "rank 0's address, for the rendezvous");
  o.master_port = static_cast<int>(flags.i64("master-port", o.master_port, "rank 0's port, for the rendezvous"));
  flags.done();
  if (nproc > 1 && rank < 0)
    return launch_ranks({argv + 1, argv + argc}, nproc);
  o.rank = std::max(rank, 0);
  o.world_size = nproc;
  train(o);
  return 0;
}

int main(int argc, char** argv) {
  // as base_train.py; read when the CUDA allocator initializes
  setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True", 0);
  try {
    return run(argc, argv);
  }
  catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
