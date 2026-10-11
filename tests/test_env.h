#pragma once

#include <filesystem>

struct TestEnv {
  std::filesystem::path base_dir;
  std::filesystem::path golden_dir;
  std::filesystem::path train_golden_dir; // per torch version: bf16 numerics differ between versions
};

TestEnv& test_env();
