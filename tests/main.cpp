// Test entry point: gtest flags, then --golden-dir / --train-golden-dir / --base-dir.
#include <gtest/gtest.h>
#if __has_include(<torch/version.h>) // train_tests
#include <torch/version.h>
#endif

#include "nanochat/common.h"
#include "nanochat/flags.h"
#include "test_env.h"

TestEnv& test_env() {
  static TestEnv env;
  return env;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv); // removes gtest's own flags
  nanochat::Flags flags(argc, argv, "nanochat-cpp tests");
  auto& env = test_env();
  env.base_dir = flags.str("base-dir", nanochat::default_base_dir().string(), "nanochat data directory");
  env.golden_dir = flags.str(
        "golden-dir", (nanochat::project_cache_dir() / "golden").string(),
        "golden data from scripts/export_golden.py (tests skip if missing)");
#ifdef TORCH_VERSION_MAJOR
  const auto train_golden = env.golden_dir / "train" /
                            ("torch-" + std::to_string(TORCH_VERSION_MAJOR) + "." +
                             std::to_string(TORCH_VERSION_MINOR));
  env.train_golden_dir = flags.str(
        "train-golden-dir", train_golden.string(), "golden data from scripts/export_train_golden.py, for this torch");
#endif
  flags.done();
  return RUN_ALL_TESTS();
}
