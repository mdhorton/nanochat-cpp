#include "nanochat/train/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <nvtx3/nvToolsExt.h>

#include "nanochat/tokenizer/tokenizer.h"
#include "nanochat/train/checkpoint.h"
#include "nanochat/train/dataloader.h"
#include "nanochat/train/dist.h"
#include "nanochat/train/loss_eval.h"
#include "nanochat/train/optim.h"

namespace nanochat {

namespace fs = std::filesystem;

ScalingParams count_params(const GPTConfig& c, int64_t pad_vocab_size_to) {
  const int64_t vocab = (c.vocab_size + pad_vocab_size_to - 1) / pad_vocab_size_to * pad_vocab_size_to;
  const int64_t d = c.n_embd, q_dim = c.n_head * c.head_dim(), kv_dim = c.n_kv_head * c.head_dim();
  int64_t num_ve = 0;
  for (int64_t i = 0; i < c.n_layer; ++i)
    num_ve += has_ve(i, c.n_layer);
  ScalingParams s;
  s.wte = vocab * d;
  s.value_embeds = num_ve * vocab * kv_dim;
  s.lm_head = vocab * d;
  s.transformer_matrices = c.n_layer * (q_dim * d + 2 * kv_dim * d + d * d + 8 * d * d) +
                           num_ve * CausalSelfAttentionImpl::kVeGateChannels * c.n_kv_head;
  s.scalars = 2 * c.n_layer + 24 + 1 + 1;
  s.total = s.wte + s.value_embeds + s.lm_head + s.transformer_matrices + s.scalars;
  return s;
}

TrainPlan plan_training(const TrainOptions& o, int64_t vocab_size, int64_t flops_per_token) {
  auto scaling_params = [&](int64_t depth) {
    const auto p = count_params(
          config_from_depth(depth, vocab_size, o.aspect_ratio, o.head_dim, o.max_seq_len, o.window_pattern));
    return p.transformer_matrices + p.lm_head; // cleanest scaling laws (nanochat dev/LOG.md, Jan 27 2026)
  };
  TrainPlan plan;
  plan.num_scaling_params = scaling_params(o.depth);
  plan.target_tokens = static_cast<int64_t>(o.target_param_data_ratio * static_cast<double>(plan.num_scaling_params));
  // d12 is the reference where hyperparameters were tuned
  plan.d_ref = o.target_param_data_ratio * static_cast<double>(scaling_params(12));
  constexpr double B_REF = 1 << 19;

  // optimal batch size grows as D^0.383 (Power Lines, arxiv 2505.13738), rounded to a power of 2
  plan.total_batch_size = o.total_batch_size;
  if (plan.total_batch_size == -1) {
    const double ratio = static_cast<double>(plan.target_tokens) / plan.d_ref;
    const double predicted = B_REF * std::pow(ratio, 0.383);
    plan.total_batch_size = int64_t{1} << static_cast<int>(std::nearbyint(std::log2(predicted))); // round half even
  }
  // lr ~ sqrt(B / B_ref); weight decay keeps T_epoch = B / (lr * wd * D) constant (arxiv 2405.13698)
  const double batch_ratio = static_cast<double>(plan.total_batch_size) / B_REF;
  if (batch_ratio != 1.0)
    plan.batch_lr_scale = std::pow(batch_ratio, 0.5);
  plan.weight_decay_scaled = o.weight_decay * std::sqrt(static_cast<double>(plan.total_batch_size) / B_REF) *
                             (plan.d_ref / static_cast<double>(plan.target_tokens));

  if (o.num_iterations > 0)
    plan.num_iterations = o.num_iterations;
  else if (o.target_flops > 0)
    plan.num_iterations = static_cast<int64_t>(
          std::nearbyint(o.target_flops / static_cast<double>(flops_per_token * plan.total_batch_size)));
  else if (o.target_param_data_ratio > 0)
    plan.num_iterations = plan.target_tokens / plan.total_batch_size;
  else
    throw std::invalid_argument("no training horizon: set num_iterations, target_flops or the data:param ratio");

  const int64_t world_tokens = o.device_batch_size * o.max_seq_len * o.world_size;
  if (plan.total_batch_size % world_tokens != 0)
    throw std::invalid_argument(
          std::format("total_batch_size ({}) must be a multiple of {}", plan.total_batch_size, world_tokens));
  plan.grad_accum_steps = plan.total_batch_size / world_tokens;
  return plan;
}

// Same expressions as base_train.py, so the doubles come out identical (Python round() is round-half-even).
double Schedules::lr_multiplier(int64_t it) const {
  const double warmdown_iters = std::nearbyint(warmdown_ratio * static_cast<double>(num_iterations));
  if (it < warmup_steps)
    return static_cast<double>(it + 1) / static_cast<double>(warmup_steps);
  if (static_cast<double>(it) <= static_cast<double>(num_iterations) - warmdown_iters)
    return 1.0;
  const double progress = static_cast<double>(num_iterations - it) / warmdown_iters;
  return progress * 1.0 + (1 - progress) * final_lr_frac;
}

double Schedules::muon_momentum(int64_t it) const {
  const double warmdown_iters = std::nearbyint(warmdown_ratio * static_cast<double>(num_iterations));
  const double warmdown_start = static_cast<double>(num_iterations) - warmdown_iters;
  if (it < 400) {
    const double frac = static_cast<double>(it) / 400;
    return (1 - frac) * 0.85 + frac * 0.97;
  }
  if (static_cast<double>(it) >= warmdown_start) {
    const double progress = (static_cast<double>(it) - warmdown_start) / warmdown_iters;
    return 0.97 * (1 - progress) + 0.90 * progress;
  }
  return 0.97;
}

double Schedules::weight_decay(int64_t it) const {
  return weight_decay_scaled * 0.5 *
         (1 + std::cos(M_PI * static_cast<double>(it) / static_cast<double>(num_iterations)));
}

double peak_flops(const std::string& device_name) {
  std::string name = device_name;
  std::ranges::transform(name, name.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  // more specific patterns first
  static const std::vector<std::pair<std::vector<std::string>, double>> table = {
        {{"gb200"}, 2.5e15},
        {{"grace blackwell"}, 2.5e15},
        {{"b200"}, 2.25e15},
        {{"b100"}, 1.8e15},
        {{"h200", "nvl"}, 836e12},
        {{"h200", "pcie"}, 836e12},
        {{"h200"}, 989e12},
        {{"h100", "nvl"}, 835e12},
        {{"h100", "pcie"}, 756e12},
        {{"h100"}, 989e12},
        {{"h800", "nvl"}, 989e12},
        {{"h800"}, 756e12},
        {{"a100"}, 312e12},
        {{"a800"}, 312e12},
        {{"a40"}, 149.7e12},
        {{"a30"}, 165e12},
        {{"l40s"}, 362e12},
        {{"l40-s"}, 362e12},
        {{"l40 s"}, 362e12},
        {{"l4"}, 121e12},
        // 70 SMs x 1024 dense BF16 FLOPs/clk x 2.25 GHz boost (from the datasheet's 1290 sparse FP4 TOPS)
        {{"rtx pro 4000"}, 161.3e12},
        {{"5090"}, 209.5e12},
        {{"4090"}, 165.2e12},
        {{"3090"}, 71e12}};
  for (const auto& [patterns, flops] : table)
    if (std::ranges::all_of(patterns, [&](const auto& p) {
          return name.find(p) != std::string::npos;
        }))
      return flops;
  return std::numeric_limits<double>::infinity();
}

namespace {

std::string with_commas(int64_t n) {
  auto s = std::to_string(n);
  for (int i = static_cast<int>(s.size()) - 3; i > (n < 0 ? 1 : 0); i -= 3)
    s.insert(static_cast<size_t>(i), ",");
  return s;
}

nlohmann::json options_to_json(const TrainOptions& o) {
  return {
        {"depth", o.depth},
        {"aspect_ratio", o.aspect_ratio},
        {"head_dim", o.head_dim},
        {"max_seq_len", o.max_seq_len},
        {"window_pattern", o.window_pattern},
        {"attention", o.attention},
        {"loss_chunk_rows", o.loss_chunk_rows},
        {"fp8", o.fp8},
        {"num_iterations", o.num_iterations},
        {"target_flops", o.target_flops},
        {"target_param_data_ratio", o.target_param_data_ratio},
        {"device_batch_size", o.device_batch_size},
        {"total_batch_size", o.total_batch_size},
        {"embedding_lr", o.embedding_lr},
        {"unembedding_lr", o.unembedding_lr},
        {"weight_decay", o.weight_decay},
        {"matrix_lr", o.matrix_lr},
        {"scalar_lr", o.scalar_lr},
        {"warmup_steps", o.warmup_steps},
        {"warmdown_ratio", o.warmdown_ratio},
        {"final_lr_frac", o.final_lr_frac},
        {"resume_from_step", o.resume_from_step},
        {"eval_every", o.eval_every},
        {"eval_tokens", o.eval_tokens},
        {"save_every", o.save_every},
        {"run", o.run}};
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// NVTX range for nsys timelines (no-op without a profiler)
// Process-wide NVTX start/end range: unlike push/pop it also covers other threads (autograd runs backward on its own)
struct NvtxProcessRange {
  explicit NvtxProcessRange(const char* name)
      : id(nvtxRangeStartA(name)) {}

  ~NvtxProcessRange() {
    nvtxRangeEnd(id);
  }

  NvtxProcessRange(const NvtxProcessRange&) = delete;
  NvtxProcessRange& operator=(const NvtxProcessRange&) = delete;

  nvtxRangeId_t id;
};

struct NvtxRange {
  explicit NvtxRange(const char* name) {
    nvtxRangePushA(name);
  }

  explicit NvtxRange(const std::string& name)
      : NvtxRange(name.c_str()) {}

  ~NvtxRange() {
    nvtxRangePop();
  }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

// JSON lines: a header (config), then one line per step and eval, keyed as base_train.py's wandb logs.
// Resuming keeps the lines before the resume step.
class MetricsLog {
public:
  MetricsLog(const fs::path& path, const nlohmann::json& header, int64_t resume_step) {
    std::vector<std::string> kept;
    if (resume_step >= 0 && fs::exists(path)) {
      std::ifstream in(path);
      for (std::string line; std::getline(in, line);) {
        const auto j = nlohmann::json::parse(line, nullptr, false);
        if (!j.is_discarded() && (!j.contains("step") || j["step"].get<int64_t>() < resume_step))
          kept.push_back(line);
      }
    }
    fs::create_directories(path.parent_path());
    out_.open(path, std::ios::trunc);
    if (kept.empty())
      kept.push_back(header.dump());
    for (const auto& line : kept)
      out_ << line << '\n';
    out_.flush();
  }

  void log(const nlohmann::json& j) {
    out_ << j.dump() << '\n' << std::flush;
  } // flushed for live readers

private:
  std::ofstream out_;
};

} // namespace

std::optional<double> train(const TrainOptions& o, const TrainCallbacks& callbacks) {
  const auto start = std::chrono::steady_clock::now();
  const bool master = o.rank == 0;
  auto print = [&](const std::string& line) {
    if (o.verbose && master)
      std::cout << line << std::endl;
  };
  const torch::Device device(torch::kCUDA, static_cast<c10::DeviceIndex>(o.rank));
  Dist dist(o.rank, o.world_size, o.master_addr, o.master_port); // also makes `device` the current GPU
  print(std::format("Distributed world size: {}", o.world_size));
  torch::manual_seed(42);                                // every rank initializes the same weights
  at::globalContext().setFloat32MatmulPrecision("high"); // TF32
  const auto device_name = std::string(at::cuda::getDeviceProperties(device.index())->name);
  const double gpu_peak_flops = peak_flops(device_name);
  print(std::format("GPU: {} | Peak FLOPS (BF16): {:.2e}", device_name, gpu_peak_flops));
  const auto attention = attention_from_string(o.attention);
  print("Attention: " + o.attention);
  if (attention == Attention::SDPA && o.window_pattern != "L")
    print("NOTE: SDPA sliding windows use an explicit mask, which is slow (see --attention fa2)");

  const auto tokenizer = Tokenizer::load(o.base_dir / "tokenizer" / "tokenizer.json");
  const auto token_bytes = load_token_bytes(o.base_dir / "tokenizer" / "token_bytes.bin", device);
  const int64_t vocab_size = tokenizer.vocab_size();
  print(std::format("Vocab size: {}", with_commas(vocab_size)));

  const auto config = config_from_depth(
        o.depth, vocab_size, o.aspect_ratio, o.head_dim, o.max_seq_len, o.window_pattern);
  print("Model config:\n" + config_to_json(config).dump(2));
  GPT model(config, device);
  model->init_weights();
  model->set_attention(attention);
  model->set_loss_chunk_rows(o.loss_chunk_rows);
  if (o.fp8) {
    const int num_linear = model->num_linears(), num_fp8 = model->set_fp8(true);
    print(std::format(
          "FP8 training enabled (tensorwise scaling) - converted {}/{} linear layers, skipped {} "
          "(too small){}",
          num_fp8, num_linear, num_linear - num_fp8,
          o.loss_chunk_rows > 0 ? "; lm_head stays bf16 in the chunked loss" : ""));
  }

  const auto checkpoint_dir = o.base_dir / "base_checkpoints" /
                              (o.run == "dummy" ? "d" + std::to_string(o.depth) : o.run);
  const bool resuming = o.resume_from_step != -1;
  std::optional<Checkpoint> ckpt;
  if (resuming) {
    print(std::format("Resuming optimization from step {}", o.resume_from_step));
    ckpt = load_checkpoint(checkpoint_dir, o.resume_from_step, device, true, o.rank);
    model->load_state(ckpt->model);
    ckpt->model.clear();
  }

  const auto counts = model->num_scaling_params();
  print(std::format(
        "Parameter counts:\nwte: {}\nvalue_embeds: {}\nlm_head: {}\ntransformer_matrices: {}\n"
        "scalars: {}\ntotal: {}",
        with_commas(counts.wte), with_commas(counts.value_embeds), with_commas(counts.lm_head),
        with_commas(counts.transformer_matrices), with_commas(counts.scalars), with_commas(counts.total)));
  const int64_t flops_per_token = model->estimate_flops();
  print(std::format("Estimated FLOPs per token: {:e}", static_cast<double>(flops_per_token)));

  const auto plan = plan_training(o, vocab_size, flops_per_token);
  std::optional<MetricsLog> metrics;
  if (master && o.run != "dummy")
    metrics.emplace(
          checkpoint_dir / "metrics.jsonl",
          nlohmann::json{
                {"run", o.run},
                {"user_config", options_to_json(o)},
                {"model_config", config_to_json(config)},
                {"world_size", o.world_size},
                {"device_name", device_name},
                {"num_scaling_params", plan.num_scaling_params},
                {"num_iterations", plan.num_iterations},
                {"total_batch_size", plan.total_batch_size}},
          o.resume_from_step);
  print(std::format(
        "Total batch size: {} tokens | LR scale: {:.4f} | weight decay: {:.6f}", with_commas(plan.total_batch_size),
        plan.batch_lr_scale, plan.weight_decay_scaled));
  const int64_t total_tokens = plan.total_batch_size * plan.num_iterations;
  print(std::format(
        "Iterations: {} | tokens: {} | tokens:scaling params: {:.2f} | FLOPs: {:e}", with_commas(plan.num_iterations),
        with_commas(total_tokens), static_cast<double>(total_tokens) / static_cast<double>(plan.num_scaling_params),
        static_cast<double>(flops_per_token) * static_cast<double>(total_tokens)));
  print(std::format(
        "Tokens / micro-batch / rank: {} x {} | gradient accumulation steps: {}", o.device_batch_size, o.max_seq_len,
        plan.grad_accum_steps));

  const double bs = plan.batch_lr_scale;
  auto optimizer = setup_optimizer(
        *model, o.unembedding_lr * bs, o.embedding_lr * bs, o.matrix_lr * bs, plan.weight_decay_scaled,
        o.scalar_lr * bs, &dist);
  if (resuming) {
    optimizer.load_state_dict(ckpt->optimizer, ckpt->optimizer_metadata);
    ckpt->optimizer.clear();
  }

  const auto data_dir = o.base_dir / "base_data_climbmix";
  DataLoaderOptions train_opts{.rank = o.rank, .world_size = o.world_size};
  if (resuming) {
    const auto& s = ckpt->meta["dataloader_state_dict"];
    train_opts.resume = DataLoaderState{.pq_idx = s["pq_idx"], .rg_idx = s["rg_idx"], .epoch = s.value("epoch", 1)};
  }
  train_opts.device = device;
  DataLoader train_loader(tokenizer, o.device_batch_size, o.max_seq_len, Split::Train, data_dir, train_opts);
  torch::Tensor x, y;
  std::tie(x, y) = train_loader.next(); // kick off the first batch

  const Schedules schedules{
        plan.num_iterations, o.warmup_steps, o.warmdown_ratio, o.final_lr_frac, plan.weight_decay_scaled};

  // loop state
  int64_t step = 0;
  std::optional<double> val_bpb;
  double min_val_bpb = std::numeric_limits<double>::infinity(), smooth_train_loss = 0, total_training_time = 0;
  if (resuming) {
    step = ckpt->meta["step"];
    if (!ckpt->meta["val_bpb"].is_null())
      val_bpb = ckpt->meta["val_bpb"].get<double>();
    const auto& ls = ckpt->meta["loop_state"];
    if (!ls["min_val_bpb"].is_null())
      min_val_bpb = ls["min_val_bpb"];
    smooth_train_loss = ls["smooth_train_loss"];
    total_training_time = ls["total_training_time"];
  }

  const int64_t N = plan.num_iterations;
  std::optional<NvtxProcessRange> profile_range; // nsys --nvtx-capture=profile, ncu --nvtx-include profile
  while (true) {
    const bool last_step = step == N;
    const double flops_so_far = static_cast<double>(flops_per_token * plan.total_batch_size) *
                                static_cast<double>(step);

    // -1: final step only (Python: never), 0: never
    if ((o.eval_every != 0 && last_step) || (o.eval_every > 0 && step % o.eval_every == 0)) {
      DataLoaderOptions val_opts{.rank = o.rank, .world_size = o.world_size};
      val_opts.device = device;
      DataLoader val_loader(tokenizer, o.device_batch_size, o.max_seq_len, Split::Val, data_dir, val_opts);
      const int64_t eval_steps = o.eval_tokens / (o.device_batch_size * o.max_seq_len * o.world_size);
      model->set_fp8(false); // evaluate in bf16, as disable_fp8
      {
        NvtxRange range("eval");
        val_bpb = evaluate_bpb(*model, val_loader, eval_steps, token_bytes, &dist);
      }
      model->set_fp8(o.fp8);
      print(std::format("Step {:05d} | Validation bpb: {:.6f}", step, *val_bpb));
      min_val_bpb = std::min(min_val_bpb, *val_bpb);
      if (metrics)
        metrics->log(
              {{"step", step},
               {"total_training_flops", flops_so_far},
               {"total_training_time", total_training_time},
               {"val/bpb", *val_bpb}});
      if (callbacks.on_eval)
        callbacks.on_eval(step, *val_bpb);
    }

    // save at the end, or every save_every steps (except the first step or the resume step)
    if (o.save &&
        (last_step || (step > 0 && step != o.resume_from_step && o.save_every > 0 && step % o.save_every == 0))) {
      const auto& ls = train_loader.state();
      const nlohmann::json meta = {
            {"step", step},
            {"val_bpb", val_bpb ? nlohmann::json(*val_bpb) : nlohmann::json()},
            {"model_config", config_to_json(config)},
            {"user_config", options_to_json(o)},
            {"device_batch_size", o.device_batch_size},
            {"max_seq_len", o.max_seq_len},
            {"total_batch_size", plan.total_batch_size},
            {"dataloader_state_dict", {{"pq_idx", ls.pq_idx}, {"rg_idx", ls.rg_idx}, {"epoch", ls.epoch}}},
            {"loop_state",
             {{"min_val_bpb", std::isinf(min_val_bpb) ? nlohmann::json() : nlohmann::json(min_val_bpb)},
              {"smooth_train_loss", smooth_train_loss},
              {"total_training_time", total_training_time}}}};
      save_checkpoint(checkpoint_dir, step, *model, &optimizer, meta, o.rank);
      print(std::format("Saved checkpoint to {} (step {})", checkpoint_dir.string(), step));
    }

    if (last_step)
      break;

    if (o.profile_start >= 0 && (step == o.profile_start || step == o.profile_start + o.profile_steps)) {
      torch::cuda::synchronize();
      if (step == o.profile_start)
        profile_range.emplace("profile");
      else
        profile_range.reset();
    }

    // one optimization step
    NvtxRange step_range(std::format("step {}", step));
    torch::cuda::synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    torch::Tensor train_loss;
    for (int64_t micro_step = 0; micro_step < plan.grad_accum_steps; ++micro_step) {
      torch::Tensor loss;
      {
        NvtxRange range("forward");
        loss = model->forward(x, y);
      }
      train_loss = loss.detach();
      {
        NvtxRange range("backward");
        (loss / plan.grad_accum_steps).backward(); // each backward sums grads, so normalize here
      }
      std::tie(x, y) = train_loader.next(); // prefetch while the GPU is busy
    }
    const double lrm = schedules.lr_multiplier(step);
    const double muon_momentum = schedules.muon_momentum(step);
    const double muon_weight_decay = schedules.weight_decay(step);
    for (auto& group : optimizer.groups()) {
      group.lr = group.initial_lr * lrm;
      if (group.kind == OptimGroup::Kind::Muon) {
        group.momentum = muon_momentum;
        group.weight_decay = muon_weight_decay;
      }
    }
    {
      NvtxRange range("optimizer");
      optimizer.step();
    }
    model->zero_grad(true);
    const double train_loss_f = train_loss.item<double>(); // CPU-GPU sync
    torch::cuda::synchronize();
    const double dt = seconds_since(t0);

    // logging
    const double ema_beta = 0.9;
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f;
    const double debiased = smooth_train_loss / (1 - std::pow(ema_beta, static_cast<double>(step + 1)));
    const double pct_done = 100.0 * static_cast<double>(step) / static_cast<double>(N);
    const auto tok_per_sec = static_cast<int64_t>(static_cast<double>(plan.total_batch_size) / dt);
    const double flops_per_sec = static_cast<double>(flops_per_token * plan.total_batch_size) / dt;
    const double mfu = 100 * flops_per_sec / (gpu_peak_flops * o.world_size);
    if (step > 10)
      total_training_time += dt; // only count the time after the first 10 steps
    std::string eta;
    if (const int64_t steps_done = step - 10; steps_done > 0)
      eta = std::format(
            " | eta: {:.1f}m",
            static_cast<double>(N - step) * (total_training_time / static_cast<double>(steps_done)) / 60);
    const auto& ls = train_loader.state();
    print(std::format(
          "step {:05d}/{:05d} ({:.2f}%) | loss: {:.6f} | lrm: {:.2f} | dt: {:.2f}ms | tok/sec: {} | "
          "bf16_mfu: {:.2f} | epoch: {} pq: {} rg: {} | total time: {:.2f}m{}",
          step, N, pct_done, debiased, lrm, dt * 1000, with_commas(tok_per_sec), mfu, ls.epoch, ls.pq_idx, ls.rg_idx,
          total_training_time / 60, eta));
    if (metrics)
      metrics->log(
            {{"step", step},
             {"total_training_flops", flops_so_far},
             {"total_training_time", total_training_time},
             {"train/loss", debiased},
             {"train/lrm", lrm},
             {"train/dt", dt},
             {"train/tok_per_sec", tok_per_sec},
             {"train/mfu", mfu},
             {"train/epoch", ls.epoch}});
    if (callbacks.on_step)
      callbacks.on_step({step, train_loss_f, lrm, dt});
    ++step;
  }

  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
  const auto peak = stats.allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].peak;
  print(std::format("Peak memory usage: {:.2f}MiB", static_cast<double>(peak) / 1024 / 1024));
  print(std::format("Total training time: {:.2f}m", total_training_time / 60));
  if (val_bpb)
    print(std::format("Minimum validation bpb: {:.6f}", min_val_bpb));
  print(std::format("Total wall-clock time: {:.2f}m (incl. setup, evals, checkpoints)", seconds_since(start) / 60));
  if (callbacks.on_end)
    callbacks.on_end(*model);
  return val_bpb;
}

} // namespace nanochat
