#include "nanochat/train/trainer.h"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <nvtx3/nvToolsExt.h>

#include "nanochat/common.h"
#include "nanochat/model/mx_gemm.h"
#include "nanochat/model/nvfp4.h"
#include "nanochat/tokenizer/tokenizer.h"
#include "nanochat/train/checkpoint.h"
#include "nanochat/train/dataloader.h"
#include "nanochat/train/dist.h"
#include "nanochat/train/loss_eval.h"
#include "nanochat/train/nvfp4_osci.h"
#include "nanochat/train/optim.h"

extern char** environ;

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
  plan.rank_accum_steps = plan.grad_accum_steps;
  if (!o.rank_micro_steps.empty()) {
    const auto& split = o.rank_micro_steps;
    int64_t sum = 0;
    for (const int64_t n : split)
      sum += n;
    if (std::ssize(split) != o.world_size || std::ranges::min(split) < 1 || sum != o.world_size * plan.grad_accum_steps)
      throw std::invalid_argument(
            std::format(
                  "rank_micro_steps needs {} values >= 1 summing to {}", o.world_size,
                  o.world_size * plan.grad_accum_steps));
    plan.rank_accum_steps = split[static_cast<size_t>(o.rank)];
  }
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
        {"seed", o.seed},
        {"window_pattern", o.window_pattern},
        {"attention", o.attention},
        {"loss_chunk_rows", o.loss_chunk_rows},
        {"fp8", o.fp8},
        {"fp8_recipe", o.fp8_recipe},
        {"gemm", o.gemm},
        {"fused", o.fused},
        {"muon_fused", o.muon_fused},
        {"nvfp4", o.nvfp4},
        {"nvfp4_rht", o.nvfp4_rht},
        {"nvfp4_sr", o.nvfp4_sr},
        {"nvfp4_eden", o.nvfp4_eden},
        {"nvfp4_eden_group", o.nvfp4_eden_group},
        {"nvfp4_eden_fixed_signs", o.nvfp4_eden_fixed_signs},
        {"nvfp4_eden_skip", o.nvfp4_eden_skip},
        {"nvfp4_weight_2d", o.nvfp4_weight_2d},
        {"nvfp4_wgrad", o.nvfp4_wgrad},
        {"nvfp4_dgrad", o.nvfp4_dgrad},
        {"nvfp4_fwd", o.nvfp4_fwd},
        {"nvfp4_four_six", o.nvfp4_four_six},
        {"nvfp4_fwd_until", o.nvfp4_fwd_until},
        {"nvfp4_until", o.nvfp4_until},
        {"nvfp4_osci_every", o.nvfp4_osci_every},
        {"nvfp4_osci_window", o.nvfp4_osci_window},
        {"nvfp4_skip_first", o.nvfp4_skip_first},
        {"nvfp4_skip_last", o.nvfp4_skip_last},
        {"num_iterations", o.num_iterations},
        {"target_flops", o.target_flops},
        {"target_param_data_ratio", o.target_param_data_ratio},
        {"device_batch_size", o.device_batch_size},
        {"total_batch_size", o.total_batch_size},
        {"rank_micro_steps", o.rank_micro_steps},
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
        {"run", o.run},
        {"wandb", o.wandb}};
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

// <dir>/metrics-YYYYmmdd-HHMMSS.jsonl (local time). Resuming continues the newest existing file.
fs::path metrics_path(const fs::path& dir, bool resuming) {
  if (resuming && fs::is_directory(dir)) {
    fs::path newest;
    for (const auto& entry : fs::directory_iterator(dir)) {
      const auto name = entry.path().filename().string();
      if (name.starts_with("metrics-") && name.ends_with(".jsonl") && entry.path() > newest)
        newest = entry.path(); // timestamps sort as strings
    }
    if (!newest.empty())
      return newest;
  }
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
  return dir / std::format("metrics-{}.jsonl", stamp);
}

// JSON lines: a header (config), then one line per step and eval, keyed as base_train.py's wandb logs, then
// {"step", "event": "end"}. Resuming keeps the lines before the resume step.
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

// tools/wandb_upload.py --follow on the metrics file, in the background. It stops at the metrics end line, or when
// this process exits.
class WandbUpload {
public:
  explicit WandbUpload(const fs::path& metrics) {
    const auto script = default_base_dir().parent_path() / "tools" / "wandb_upload.py"; // <project>/cache's parent
    const auto pid = std::to_string(getpid());
    std::vector<std::string> argv{"python", script.string(), "--file", metrics.string(), "--follow", "--pid", pid};
    std::vector<char*> cargv;
    for (auto& a : argv)
      cargv.push_back(a.data());
    cargv.push_back(nullptr);
    if (const int err = posix_spawnp(&pid_, "python", nullptr, nullptr, cargv.data(), environ))
      throw std::runtime_error(std::string("cannot start wandb_upload.py: ") + std::strerror(err));
  }

  // after the end line: lets the upload finish, up to timeout_s
  void wait(double timeout_s) const {
    const auto t0 = std::chrono::steady_clock::now();
    int status = 0;
    while (waitpid(pid_, &status, WNOHANG) == 0) {
      if (seconds_since(t0) > timeout_s) {
        std::cerr << "wandb_upload.py still running, not waiting for it" << std::endl;
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

private:
  pid_t pid_ = -1;
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
  torch::manual_seed(static_cast<uint64_t>(o.seed));     // every rank initializes the same weights
  at::globalContext().setFloat32MatmulPrecision("high"); // TF32
  if (o.cublaslt_workspace_mb > 0)
    at::cuda::setCUDABlasLtWorkspaceSize(static_cast<size_t>(o.cublaslt_workspace_mb) << 20);
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
  Nvfp4Options nvfp4{.fwd = false, .dgrad = false, .wgrad = false}; // outlives the model
  GPT model(config, device);
  model->init_weights();
  model->set_attention(attention);
  model->set_loss_chunk_rows(o.loss_chunk_rows);
  model->set_fused(o.fused);
  if (o.fp8) {
    if (o.fp8_recipe != "tensorwise" && o.fp8_recipe != "mxfp8")
      throw std::invalid_argument("unknown fp8 recipe: " + o.fp8_recipe + " (use tensorwise or mxfp8)");
    model->set_fp8_recipe(o.fp8_recipe == "mxfp8" ? Fp8Recipe::Mx : Fp8Recipe::Tensorwise);
    if (o.gemm != "cublas" && o.gemm != "cutlass")
      throw std::invalid_argument("unknown gemm: " + o.gemm + " (use cublas or cutlass)");
    set_mx_gemm_backend(o.gemm == "cutlass" ? MxGemmBackend::Cutlass : MxGemmBackend::Cublas);
    if (o.nvfp4_gemm != "auto" && o.nvfp4_gemm != "cublas" && o.nvfp4_gemm != "cutlass")
      throw std::invalid_argument("unknown nvfp4_gemm: " + o.nvfp4_gemm + " (use auto, cublas or cutlass)");
    set_nvfp4_gemm_backend(
          o.nvfp4_gemm == "auto"      ? Nvfp4GemmBackend::Auto
          : o.nvfp4_gemm == "cutlass" ? Nvfp4GemmBackend::Cutlass
                                      : Nvfp4GemmBackend::Cublas);
    const int num_linear = model->num_linears(), num_fp8 = model->set_fp8(true);
    print(std::format(
          "FP8 training enabled ({} scaling{}) - converted {}/{} linear layers, skipped {} (too small)", o.fp8_recipe,
          o.fp8_recipe == "mxfp8" ? ", " + o.gemm + " GEMMs" : "", num_fp8, num_linear, num_linear - num_fp8));
  }
  // a comma list of GEMMs -> {fwd, dgrad, wgrad}
  const auto gemms = [](const std::string& list, const char* what, bool fwd_ok) {
    std::array<bool, 3> on{};
    for (size_t pos = 0; !list.empty() && pos <= list.size();) {
      const size_t comma = std::min(list.find(',', pos), list.size());
      const auto gemm = list.substr(pos, comma - pos);
      if (gemm == "fwd" && fwd_ok)
        on[0] = true;
      else if (gemm == "dgrad")
        on[1] = true;
      else if (gemm == "wgrad")
        on[2] = true;
      else
        throw std::invalid_argument(
              std::format("unknown {} GEMM: {} (use {}dgrad, wgrad)", what, gemm, fwd_ok ? "fwd, " : ""));
      pos = comma + 1;
    }
    return on;
  };
  const auto rht = gemms(o.nvfp4_rht, "nvfp4-rht", true), sr = gemms(o.nvfp4_sr, "nvfp4-sr", false);
  const bool nvfp4_real = o.nvfp4_wgrad || o.nvfp4_dgrad || o.nvfp4_fwd;
  if ((!o.nvfp4.empty() || nvfp4_real) && (!o.fp8 || o.fp8_recipe != "mxfp8"))
    throw std::invalid_argument("nvfp4 needs fp8 with the mxfp8 recipe");
  if (!o.nvfp4.empty() && nvfp4_real)
    throw std::invalid_argument("nvfp4 (simulated) and nvfp4_wgrad / nvfp4_dgrad / nvfp4_fwd (real) are exclusive");
  if (o.nvfp4_dgrad && rht[1])
    throw std::invalid_argument("nvfp4_dgrad: no rht (simulated only)");
  const auto eden = gemms(o.nvfp4_eden, "nvfp4-eden", false);
  if (o.nvfp4.empty() && (eden[2] || (eden[1] && !o.nvfp4_dgrad)))
    throw std::invalid_argument("nvfp4_eden: simulated (--nvfp4), or real for dgrad (--nvfp4-dgrad)");
  const bool eden_sim = !o.nvfp4.empty() && (eden[1] || eden[2]);
  if (!eden_sim && (o.nvfp4_eden_group != 128 || o.nvfp4_eden_fixed_signs || !o.nvfp4_eden_skip.empty()))
    throw std::invalid_argument("nvfp4_eden_group / nvfp4_eden_fixed_signs / nvfp4_eden_skip: simulated MS-EDEN only");
  if (o.nvfp4_eden_group < 16 || (o.nvfp4_eden_group & (o.nvfp4_eden_group - 1)) != 0)
    throw std::invalid_argument("nvfp4_eden_group: a power of 2, >= 16");
  std::vector<std::string> eden_skip;
  for (size_t pos = 0; !o.nvfp4_eden_skip.empty() && pos <= o.nvfp4_eden_skip.size();) {
    const size_t comma = std::min(o.nvfp4_eden_skip.find(',', pos), o.nvfp4_eden_skip.size());
    eden_skip.push_back(o.nvfp4_eden_skip.substr(pos, comma - pos));
    pos = comma + 1;
  }
  if (!o.nvfp4.empty()) {
    const auto on = gemms(o.nvfp4, "nvfp4", true);
    nvfp4 = {
          .fwd = on[0],
          .dgrad = on[1],
          .wgrad = on[2],
          .rht_fwd = rht[0],
          .rht_dgrad = rht[1],
          .rht_wgrad = rht[2],
          .sr_dgrad = sr[1],
          .sr_wgrad = sr[2],
          .eden_dgrad = eden[1],
          .eden_wgrad = eden[2],
          .eden_group = o.nvfp4_eden_group,
          .eden_fixed_signs = o.nvfp4_eden_fixed_signs,
          .weight_2d = o.nvfp4_weight_2d,
          .seed = static_cast<uint64_t>(o.seed)};
    const int n = model->set_nvfp4(&nvfp4, o.nvfp4_skip_first, o.nvfp4_skip_last, eden_skip);
    print(std::format(
          "Simulated NVFP4 for {} ({} linear layers; rht {}, sr {}, eden {}, 2d weights {}, seed {})", o.nvfp4, n,
          o.nvfp4_rht.empty() ? "none" : o.nvfp4_rht, o.nvfp4_sr.empty() ? "none" : o.nvfp4_sr,
          o.nvfp4_eden.empty() ? "none" : o.nvfp4_eden, nvfp4.weight_2d, nvfp4.seed));
    if (eden_sim)
      print(std::format(
            "  MS-EDEN group {}, signs {}, skipped Linears: {}", nvfp4.eden_group,
            nvfp4.eden_fixed_signs ? "per step" : "per GEMM", o.nvfp4_eden_skip.empty() ? "none" : o.nvfp4_eden_skip));
  }

  // real NVFP4 backward GEMMs, process-wide until train returns
  const Nvfp4Backward backward{
        .wgrad = o.nvfp4_wgrad,
        .dgrad = o.nvfp4_dgrad,
        .rht = rht[2],
        .sr = sr[2],
        .sr_dgrad = sr[1],
        .eden_dgrad = o.nvfp4_dgrad && eden[1],
        .seed = static_cast<uint64_t>(o.seed)};

  struct BackwardScope {
    explicit BackwardScope(const Nvfp4Backward* b) {
      set_nvfp4_backward(b);
    }

    ~BackwardScope() {
      set_nvfp4_backward(nullptr);
    }
  } backward_scope(o.nvfp4_wgrad || o.nvfp4_dgrad ? &backward : nullptr);

  if (o.nvfp4_four_six && !o.nvfp4_fwd)
    throw std::invalid_argument("nvfp4_four_six: needs nvfp4_fwd");

  struct FourSixScope {
    explicit FourSixScope(bool on) {
      set_nvfp4_four_six(on);
    }

    ~FourSixScope() {
      set_nvfp4_four_six(false);
    }
  } four_six_scope(o.nvfp4_four_six);

  if (o.nvfp4_fwd) {
    const int n = model->set_nvfp4_fwd(true, o.nvfp4_skip_first, o.nvfp4_skip_last);
    print(std::format(
          "NVFP4 forward GEMMs for {} linear layers (16x16 weight blocks{})", n,
          o.nvfp4_four_six ? ", 4/6 scales" : ""));
  }

  if (o.nvfp4_wgrad)
    print(std::format("NVFP4 weight gradients (rht {}, sr {}, seed {})", backward.rht, backward.sr, backward.seed));
  if (o.nvfp4_dgrad)
    print(std::format(
          "NVFP4 input gradients ({}, seed {})",
          backward.eden_dgrad ? "MS-EDEN" : std::format("sr {}", backward.sr_dgrad), backward.seed));

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
        "Parameter counts:\n  wte: {}\n  value_embeds: {}\n  lm_head: {}\n  transformer_matrices: {}\n"
        "  scalars: {}\n  total: {}",
        with_commas(counts.wte), with_commas(counts.value_embeds), with_commas(counts.lm_head),
        with_commas(counts.transformer_matrices), with_commas(counts.scalars), with_commas(counts.total)));
  const int64_t flops_per_token = model->estimate_flops();
  print("Estimated FLOPs per token: " + with_commas(flops_per_token));

  const auto plan = plan_training(o, vocab_size, flops_per_token);
  std::optional<MetricsLog> metrics;
  std::optional<WandbUpload> wandb;
  if (master && o.run != "dummy") {
    const auto path = metrics_path(o.base_dir / "metrics" / o.run, resuming);
    print("Metrics: " + path.string());
    metrics.emplace(
          path,
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
    if (o.wandb)
      wandb.emplace(path);
  }
  print(std::format(
        "Total batch size: {} tokens | LR scale: {:.4f} | weight decay: {:.6f}", with_commas(plan.total_batch_size),
        plan.batch_lr_scale, plan.weight_decay_scaled));
  const int64_t total_tokens = plan.total_batch_size * plan.num_iterations;
  print(std::format(
        "Iterations: {} | tokens: {} | tokens:scaling params: {:.2f} | FLOPs: {:e}", with_commas(plan.num_iterations),
        with_commas(total_tokens), static_cast<double>(total_tokens) / static_cast<double>(plan.num_scaling_params),
        static_cast<double>(flops_per_token) * static_cast<double>(total_tokens)));
  print(std::format(
        "Tokens / micro-batch / rank: {} x {} | gradient accumulation steps: {}{}", o.device_batch_size, o.max_seq_len,
        plan.grad_accum_steps,
        o.rank_micro_steps.empty() ? "" : " (per rank: " + nlohmann::json(o.rank_micro_steps).dump() + ")"));

  const double bs = plan.batch_lr_scale;
  auto optimizer = setup_optimizer(
        *model, o.unembedding_lr * bs, o.embedding_lr * bs, o.matrix_lr * bs, plan.weight_decay_scaled,
        o.scalar_lr * bs, &dist);
  optimizer.set_fused_muon(o.muon_fused);
  if (o.muon_fused)
    print("Muon update: fused kernels");
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
  // nvfp4_fwd_until / nvfp4_until -> the first MXFP8 step (N: never)
  const auto until_step = [&](const std::string& until, const char* what) -> int64_t {
    if (until.empty())
      return N;
    if (o.nvfp4.empty() && !nvfp4_real)
      throw std::invalid_argument(std::format("{}: needs nvfp4 or nvfp4_wgrad / nvfp4_dgrad / nvfp4_fwd", what));
    if (until == "warmdown")
      return N - static_cast<int64_t>(std::nearbyint(o.warmdown_ratio * static_cast<double>(N)));
    size_t end = 0;
    const double f = std::stod(until, &end);
    if (end != until.size() || f < 0 || f > 1)
      throw std::invalid_argument(std::format("{}: {} (use warmdown or a fraction in [0, 1])", what, until));
    return static_cast<int64_t>(std::nearbyint(f * static_cast<double>(N)));
  };
  const int64_t nvfp4_off = until_step(o.nvfp4_until, "nvfp4_until");
  const int64_t nvfp4_fwd_off = std::min(until_step(o.nvfp4_fwd_until, "nvfp4_fwd_until"), nvfp4_off);
  bool nvfp4_fwd_on = o.nvfp4_fwd || nvfp4.fwd, nvfp4_on = nvfp4_real || !o.nvfp4.empty();
  if (!o.nvfp4_fwd_until.empty() && !nvfp4_fwd_on)
    throw std::invalid_argument("nvfp4_fwd_until: needs NVFP4 forward GEMMs");
  // the blocks' NVFP4-forward Linears (all FP8 ones without nvfp4_fwd, as a control), rank 0
  std::optional<Nvfp4OsciMonitor> osci;
  if (o.nvfp4_osci_every > 0 && master) {
    std::vector<std::pair<std::string, torch::Tensor>> ws;
    for (const auto& block : *model->transformer->h)
      for (const auto& item : block->named_modules("", false))
        if (const auto* linear = dynamic_cast<const LinearImpl*>(item.value().get());
            linear != nullptr && linear->fp8 && (linear->nvfp4_fwd || !o.nvfp4_fwd))
          ws.emplace_back(item.key(), linear->weight);
    osci.emplace(std::move(ws), o.nvfp4_osci_every, o.nvfp4_osci_window);
  }
  double total_dt = 0; // every step of this run, incl. the first 11 that total_training_time skips
  std::optional<NvtxProcessRange> profile_range; // nsys --nvtx-capture=profile, ncu --nvtx-include profile
  optimizer.zero_grad();                         // the Muon grads are views of the optimizer's stacks
  while (true) {
    const bool last_step = step == N;
    nvfp4.step = step;
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

    // NVFP4 GEMMs back to MXFP8 (between steps: the operands' formats are chosen per GEMM)
    if (nvfp4_fwd_on && step >= nvfp4_fwd_off) {
      nvfp4.fwd = nvfp4_fwd_on = false;
      model->set_nvfp4_fwd(false);
      print(std::format("Step {:05d} | NVFP4 forward GEMMs off (MXFP8)", step));
    }
    if (nvfp4_on && step >= nvfp4_off) {
      nvfp4_on = false;
      model->set_nvfp4(nullptr);
      set_nvfp4_backward(nullptr);
      print(std::format("Step {:05d} | NVFP4 GEMMs off (MXFP8)", step));
    }

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
    for (int64_t micro_step = 0; micro_step < plan.rank_accum_steps; ++micro_step) {
      torch::Tensor loss;
      {
        NvtxRange range("forward");
        loss = model->forward(x, y);
      }
      train_loss = loss.detach();
      {
        NvtxRange range("backward");
        // each backward sums grads, so normalize here. The mean micro-steps per rank, not this rank's: with the
        // ranks' average, the step's grad is the mean over all micro-steps even when rank_micro_steps is uneven.
        (loss / plan.grad_accum_steps).backward();
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
    optimizer.zero_grad();
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
    total_dt += dt;
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
    if (osci)
      if (const auto r = osci->after_step(step)) {
        const auto pct = [](double x) {
          return 100 * x;
        };
        const auto& all = (*r)["all"];
        std::string kinds;
        for (const auto& [kind, v] : r->items())
          if (v.is_object() && kind != "all")
            kinds += std::format(" {} {:.2f}%", kind, pct(v["osc"]));
        print(std::format(
              "Step {:05d} | NVFP4 weight oscillation, steps {}-{} (lrm {:.2f}-{:.2f}): moved {:.2f}% | osc {:.2f}% | "
              "r8 {:.3f}% | flips {:.2f} | back q {:.1f}% w {:.1f}% | osc by kind:{}",
              step, (*r)["first"].get<int64_t>(), step, schedules.lr_multiplier((*r)["first"].get<int64_t>()), lrm,
              pct(all["moved"]), pct(all["osc"]), pct(all["r8"]), all["flips"].get<double>(), pct(all["back_q"]),
              pct(all["back_w"]), kinds));
        if (metrics) {
          nlohmann::json m{{"step", step}};
          for (const auto& [kind, v] : r->items())
            if (v.is_object())
              for (const auto& [stat, x] : v.items())
                m["osci/" + kind + "/" + stat] = x;
          metrics->log(m);
        }
      }
    ++step;
  }
  if (metrics)
    metrics->log({{"step", step}, {"event", "end"}}); // resuming drops it with the steps >= resume step

  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
  const auto peak = stats.allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].peak;
  print(std::format("Peak memory usage: {:.2f}MiB", static_cast<double>(peak) / 1024 / 1024));
  print(std::format("Total training time: {:.2f}m", total_training_time / 60));
  print(std::format("Total DT: {:.2f}m", total_dt / 60));
  if (val_bpb)
    print(std::format("Minimum validation bpb: {:.6f}", min_val_bpb));
  print(std::format("Total wall-clock time: {:.2f}m (incl. setup, evals, checkpoints)", seconds_since(start) / 60));
  if (wandb)
    wandb->wait(60);
  if (callbacks.on_end)
    callbacks.on_end(*model);
  return val_bpb;
}

} // namespace nanochat
