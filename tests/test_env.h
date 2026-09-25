#pragma once

#include <filesystem>

struct TestEnv {
    std::filesystem::path base_dir;
    std::filesystem::path golden_dir;
};

TestEnv &test_env();
