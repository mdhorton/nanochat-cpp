// Bits per byte: a vocab-size independent loss. Port of nanochat/loss_eval.py.
#pragma once

#include <cstdint>
#include <filesystem>

#include <torch/torch.h>

#include "nanochat/model/gpt.h"
#include "nanochat/train/dataloader.h"
#include "nanochat/train/dist.h"

namespace nanochat {

// token_bytes.bin (from tok_train / export_tokenizer.py): int32 byte length per token id, 0 for special tokens.
torch::Tensor load_token_bytes(const std::filesystem::path& path, torch::Device device = torch::kCUDA);

// Sum of per-token losses (nats) over the sum of target token bytes, in bits, on `steps` batches. Special tokens
// (0 bytes) and ignored targets (-1) are excluded. With several ranks, the sums are over all of them.
double evaluate_bpb(
      GPTImpl& model, DataLoader& batches, int64_t steps, const torch::Tensor& token_bytes, Dist* dist = nullptr);

} // namespace nanochat
