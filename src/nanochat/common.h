#pragma once

#include <filesystem>

#ifndef NANOCHAT_CACHE_DIR
#error "NANOCHAT_CACHE_DIR must be defined (set by CMakeLists.txt)"
#endif

namespace nanochat {

// Default location of intermediates (data shards, tokenizer, golden data): <project>/cache.
// Override with --base-dir; Python nanochat shares it via NANOCHAT_BASE_DIR.
inline std::filesystem::path default_base_dir() {
  return NANOCHAT_CACHE_DIR;
}

} // namespace nanochat
