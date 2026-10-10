#pragma once

#include <cstdlib>
#include <filesystem>

#ifndef NANOCHAT_PROJECT_DIR
#error "NANOCHAT_PROJECT_DIR must be defined (set by CMakeLists.txt)"
#endif

namespace nanochat {

// the source tree (scripts/)
inline std::filesystem::path project_dir() {
  return NANOCHAT_PROJECT_DIR;
}

// <project>/cache: this project's own files (metrics, golden data), wherever the base dir is
inline std::filesystem::path project_cache_dir() {
  return project_dir() / "cache";
}

// Default location of the data shared with Python nanochat (data shards, tokenizer, checkpoints): NANOCHAT_BASE_DIR
// when set and non-empty (as Python nanochat), else <project>/cache. Override with --base-dir.
inline std::filesystem::path default_base_dir() {
  const char* dir = std::getenv("NANOCHAT_BASE_DIR");
  return dir && *dir ? std::filesystem::path(dir) : project_cache_dir();
}

} // namespace nanochat
