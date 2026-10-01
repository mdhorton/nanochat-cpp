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
  o.seed = flags.i64("seed", o.seed, "weight init and nvfp4 stochastic rounding seed");
  o.window_pattern = flags.str(
        "window-pattern", o.window_pattern, "sliding window pattern tiled across layers: L=full, S=quarter context");
  o.attention = flags.str(
        "attention", o.attention,
        "bf16 (our sm120 flash attention, FA2 backward), bf16mx (bf16 forward, our MXFP8 backward), mx (MXFP8 "
        "forward and backward), fa2 (FlashAttention-2) or sdpa (bit-identical to Python)");
  o.loss_chunk_rows = flags.i64(
        "loss-chunk-rows", o.loss_chunk_rows, "rows per chunk of the fused lm_head + loss (0 = unchunked, as Python)");
  o.fp8 = flags.boolean("fp8", o.fp8, "FP8 training (eval stays bf16); false = bf16, as Python's default");
  o.fp8_recipe = flags.str("fp8-recipe", o.fp8_recipe, "mxfp8 (32-value block scales) or tensorwise (as Python)");
  o.gemm = flags.str("gemm", o.gemm, "MXFP8 GEMMs: cutlass or cublas (cuBLASLt)");
  o.nvfp4_gemm = flags.str(
        "nvfp4-gemm", o.nvfp4_gemm,
        "NVFP4 GEMMs: cutlass, cublas (cuBLASLt) or auto (per shape, whichever timed faster on its first call)");
  o.fused = flags.boolean("fused", o.fused, "fused elementwise CUDA kernels (false = op by op, as Python)");
  o.muon_fused = flags.boolean(
        "muon-fused", o.muon_fused, "Muon's update in fused CUDA kernels (not bit-identical to Python's op path)");
  o.nvfp4 = flags.str(
        "nvfp4", o.nvfp4,
        "simulated NVFP4 (slow, for numerics) for these GEMMs of the blocks' Linears: comma list of fwd, dgrad, wgrad "
        "(empty = off; needs --fp8-recipe=mxfp8)");
  o.nvfp4_rht = flags.str(
        "nvfp4-rht", o.nvfp4_rht,
        "nvfp4: GEMMs with a random Hadamard transform along K (fwd, dgrad, wgrad; empty = none)");
  o.nvfp4_sr = flags.str(
        "nvfp4-sr", o.nvfp4_sr,
        "nvfp4: GEMMs whose gradient operand rounds stochastically (dgrad, wgrad; empty = none)");
  o.nvfp4_eden = flags.str(
        "nvfp4-eden", o.nvfp4_eden,
        "nvfp4: GEMMs with Quartet II's MS-EDEN operands, replacing their rht / sr (dgrad, wgrad; empty = none); "
        "nvfp4-dgrad: dgrad (real)");
  o.nvfp4_eden_group = flags.i64(
        "nvfp4-eden-group", o.nvfp4_eden_group, "nvfp4-eden (simulated): rotation / correction group (power of 2)");
  o.nvfp4_eden_fixed_signs = flags.boolean(
        "nvfp4-eden-fixed-signs", o.nvfp4_eden_fixed_signs,
        "nvfp4-eden (simulated): rotation signs fixed per step (per Linear and GEMM) instead of per GEMM call");
  o.nvfp4_eden_skip = flags.str(
        "nvfp4-eden-skip", o.nvfp4_eden_skip,
        "nvfp4-eden (simulated): Linears kept on rht / sr, comma list of module names or their last parts (c_fc, "
        "mlp.c_proj, ...; empty = none)");
  o.nvfp4_weight_2d = flags.boolean("nvfp4-weight-2d", o.nvfp4_weight_2d, "nvfp4: 16x16 weight blocks (without rht)");
  o.nvfp4_wgrad = flags.boolean(
        "nvfp4-wgrad", o.nvfp4_wgrad,
        "real NVFP4 weight gradients (CUTLASS) for the Linears but lm_head, with --nvfp4-rht/--nvfp4-sr's wgrad");
  o.nvfp4_dgrad = flags.boolean(
        "nvfp4-dgrad", o.nvfp4_dgrad,
        "real NVFP4 input gradients (CUTLASS) for the Linears but lm_head, with --nvfp4-sr's dgrad (no rht)");
  o.nvfp4_fwd = flags.boolean(
        "nvfp4-fwd", o.nvfp4_fwd,
        "real NVFP4 forward GEMMs (CUTLASS) for the blocks' Linears: 16x16 weight blocks (dgrad sees the same weight), "
        "round to nearest");
  o.nvfp4_four_six = flags.boolean(
        "nvfp4-4over6", o.nvfp4_four_six,
        "nvfp4-fwd: 4/6 block scales (each 16-block's max to 6 or 4, the lower error) for the weights and the inputs "
        "written as rows (norm, attention; not relu^2's)");
  o.nvfp4_fwd_until = flags.str(
        "nvfp4-fwd-until", o.nvfp4_fwd_until,
        "nvfp4, nvfp4-fwd: forward GEMMs MXFP8 from this step on: warmdown (its start) or a fraction of the steps "
        "(empty = never)");
  o.nvfp4_until = flags.str(
        "nvfp4-until", o.nvfp4_until,
        "nvfp4, nvfp4-*: all GEMMs MXFP8 from this step on: warmdown (its start) or a fraction of the steps (empty = "
        "never)");
  o.nvfp4_skip_first = flags.i64(
        "nvfp4-skip-first", o.nvfp4_skip_first, "nvfp4, nvfp4-fwd: first blocks kept MXFP8 (nvfp4-fwd: forward only)");
  o.nvfp4_skip_last = flags.i64(
        "nvfp4-skip-last", o.nvfp4_skip_last, "nvfp4, nvfp4-fwd: last blocks kept MXFP8 (nvfp4-fwd: forward only)");
  o.cublaslt_workspace_mb = flags.i64(
        "cublaslt-workspace-mb", o.cublaslt_workspace_mb,
        "cuBLASLt workspace in MB; split-K GEMMs need a few (<= 0 = torch's 1 MB default)");
  // horizon
  o.num_iterations = flags.i64("num-iterations", o.num_iterations, "explicit number of steps (-1 = disable)");
  o.target_flops = flags.f64("target-flops", o.target_flops, "steps to reach target FLOPs (-1 = disable)");
  o.target_param_data_ratio = flags.f64(
        "target-param-data-ratio", o.target_param_data_ratio, "steps for this data:param ratio (-1 = disable)");
  // optimization
  o.device_batch_size = flags.i64("device-batch-size", o.device_batch_size, "per-device batch size (reduce on OOM)");
  o.total_batch_size = flags.i64("total-batch-size", o.total_batch_size, "total batch size in tokens (-1 = auto)");
  const auto micro_steps = flags.str(
        "rank-micro-steps", "",
        "micro-steps per step for each rank, e.g. 126,130 to give a slower GPU 0 less (sum = nproc x grad accum "
        "steps; empty = even)");
  for (size_t pos = 0; pos < micro_steps.size();) {
    const size_t comma = std::min(micro_steps.find(',', pos), micro_steps.size());
    o.rank_micro_steps.push_back(std::stoll(micro_steps.substr(pos, comma - pos)));
    pos = comma + 1;
  }
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
  o.wandb = flags.boolean(
        "wandb", o.wandb, "upload the metrics live to wandb (project nanochat) with tools/wandb_upload.py");
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
