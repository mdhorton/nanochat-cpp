// Shared by the train golden tests: TrainOptions from tools/export_train_golden.py's "options".
#pragma once

#include <nlohmann/json.hpp>

#include "nanochat/train/trainer.h"

namespace nanochat {

inline TrainOptions options_from_json(const nlohmann::json& j) {
  TrainOptions o;
  o.depth = j["depth"];
  o.aspect_ratio = j["aspect_ratio"];
  o.head_dim = j["head_dim"];
  o.max_seq_len = j["max_seq_len"];
  o.window_pattern = j["window_pattern"];
  o.num_iterations = j["num_iterations"];
  o.target_flops = j["target_flops"];
  o.target_param_data_ratio = j["target_param_data_ratio"];
  o.device_batch_size = j["device_batch_size"];
  o.total_batch_size = j["total_batch_size"];
  o.embedding_lr = j["embedding_lr"];
  o.unembedding_lr = j["unembedding_lr"];
  o.weight_decay = j["weight_decay"];
  o.matrix_lr = j["matrix_lr"];
  o.scalar_lr = j["scalar_lr"];
  o.warmup_steps = j["warmup_steps"];
  o.warmdown_ratio = j["warmdown_ratio"];
  o.final_lr_frac = j["final_lr_frac"];
  o.eval_every = j["eval_every"];
  o.eval_tokens = j["eval_tokens"];
  o.fp8 = j.value("fp8", false);
  return o;
}

} // namespace nanochat
