// Test entry point: gtest flags, then --golden-dir / --base-dir.
#include <gtest/gtest.h>

#include "nanochat/common.h"
#include "nanochat/flags.h"
#include "test_env.h"

TestEnv &test_env() {
    static TestEnv env;
    return env;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv); // removes gtest's own flags
    nanochat::Flags flags(argc, argv, "nanochat-cpp tests");
    auto &env = test_env();
    env.base_dir = flags.str("base-dir", nanochat::default_base_dir().string(), "nanochat data directory");
    env.golden_dir = flags.str("golden-dir", (env.base_dir / "golden").string(),
                               "golden data from tools/export_golden.py (tests skip if missing)");
    flags.done();
    return RUN_ALL_TESTS();
}
