// GPT-4 style BPE tokenizer. Port of nanochat/tokenizer.py (training as in rustbpe, encoding as in tiktoken).
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "nanochat/tokenizer/splitter.h"

namespace nanochat {

using token_t = uint32_t;

inline constexpr std::array<std::string_view, 9> SPECIAL_TOKENS = {
      "<|bos|>", // every document begins with this
      // used during finetuning to render conversations
      "<|user_start|>",
      "<|user_end|>",
      "<|assistant_start|>",
      "<|assistant_end|>",
      "<|python_start|>",
      "<|python_end|>",
      "<|output_start|>",
      "<|output_end|>",
};

// Fills docs with the next batch of training text; returns false when exhausted.
using TextSource = std::function<bool(std::vector<std::string>&)>;

struct RenderedConversation {
  std::vector<token_t> ids;
  std::vector<uint8_t> mask; // 1 = assistant is trained on this token
};

class Tokenizer {
public:
  // mergeable_ranks[i] holds the bytes of token i (rank == id).
  Tokenizer(
        std::vector<std::string> mergeable_ranks, const std::vector<std::pair<std::string, token_t>>& special_tokens,
        std::string pattern = std::string(SPLIT_PATTERN), std::string_view bos_token = "<|bos|>");
  Tokenizer(Tokenizer&&) = default;
  Tokenizer(const Tokenizer&) = delete; // encoder_ views into decoder_

  // Appends SPECIAL_TOKENS after the mergeable ranks.
  static Tokenizer from_ranks(std::vector<std::string> mergeable_ranks);
  static Tokenizer train(const TextSource& source, uint32_t vocab_size, int num_threads);
  // Pretrained tiktoken encodings (gpt2, cl100k_base) use bos_token "<|endoftext|>".
  static Tokenizer load(const std::filesystem::path& json_path, std::string_view bos_token = "<|bos|>");
  void save(const std::filesystem::path& json_path) const;

  uint32_t vocab_size() const {
    return static_cast<uint32_t>(decoder_.size());
  }

  const std::string& pattern() const {
    return pattern_;
  }

  std::vector<std::string> special_tokens() const;
  token_t encode_special(std::string_view name) const;

  token_t bos_token_id() const {
    return bos_;
  }

  std::vector<token_t> encode(
        std::string_view text, std::optional<token_t> prepend = {}, std::optional<token_t> append = {}) const;
  std::vector<std::vector<token_t>> encode_batch(
        std::span<const std::string> texts, std::optional<token_t> prepend = {}, std::optional<token_t> append = {},
        int num_threads = 8) const;
  // Appends ids of text, treating special token strings as ordinary text.
  void encode_ordinary(std::string_view text, std::vector<token_t>& out) const;

  // UTF-8 text; invalid sequences become U+FFFD (like tiktoken's decode).
  std::string decode(std::span<const token_t> ids) const;
  std::string decode_bytes(std::span<const token_t> ids) const;
  const std::string& decode_single_token_bytes(token_t id) const;

  std::string id_to_token(token_t id) const {
    return decode(std::span(&id, 1));
  }

  RenderedConversation render_conversation(const nlohmann::json& conversation, size_t max_tokens = 2048) const;
  // Conversation minus the last (assistant) message, primed with <|assistant_start|>.
  std::vector<token_t> render_for_completion(const nlohmann::json& conversation) const;

  // Byte length of each token (0 for special tokens), for bits-per-byte evaluation.
  std::vector<int32_t> token_bytes() const;

private:
  void byte_pair_encode(std::string_view piece, std::vector<token_t>& out) const;

  std::string pattern_;
  std::unique_ptr<Splitter> splitter_;
  std::vector<std::string> decoder_;                      // id -> bytes, specials included
  std::unordered_map<std::string_view, token_t> encoder_; // mergeable bytes -> rank; views into decoder_
  std::unordered_map<std::string, token_t> special_encoder_;
  token_t bos_;
};

} // namespace nanochat
