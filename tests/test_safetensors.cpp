#include <gtest/gtest.h>

#include <unistd.h>

#include "nanochat/train/safetensors.h"
#include "test_env.h"

namespace fs = std::filesystem;
namespace st = nanochat::safetensors;

namespace {

    fs::path temp_file(const std::string &name) {
        return fs::temp_directory_path() / ("nanochat_" + std::to_string(::getpid()) + "_" + name);
    }

    // Values written by tools/export_train_golden.py (safetensors_check.safetensors).
    st::TensorMap check_tensors() {
        return {{"f32", torch::arange(12, torch::kFloat32).reshape({3, 4}) / 3},
                {"bf16", (torch::arange(6, torch::kFloat32) - 2.5).to(torch::kBFloat16)},
                {"i64", torch::arange(-3, 4, torch::kInt64)},
                {"i32", torch::arange(5, torch::kInt32).reshape({5, 1})},
                {"scalar", torch::tensor(1.25, torch::kFloat32)}};
    }

    void expect_same(const st::TensorMap &got, const st::TensorMap &want) {
        ASSERT_EQ(got.size(), want.size());
        for (const auto &[name, t]: want) {
            ASSERT_TRUE(got.contains(name)) << name;
            const auto &g = got.at(name);
            EXPECT_EQ(g.scalar_type(), t.scalar_type()) << name;
            EXPECT_EQ(g.sizes(), t.sizes()) << name;
            EXPECT_TRUE(torch::equal(g.cpu(), t.cpu())) << name;
        }
    }

} // namespace

TEST(Safetensors, RoundTrip) {
    const auto path = temp_file("roundtrip.safetensors");
    auto tensors = check_tensors();
    tensors["empty"] = torch::empty({0, 3});
    tensors["strided"] = torch::arange(20, torch::kFloat32).reshape({4, 5}).t(); // non-contiguous
    st::save(path, tensors, {{"step", "7"}});
    expect_same(st::load(path), tensors);
    EXPECT_EQ(st::load_metadata(path).at("step"), "7");
    fs::remove(path);
}

TEST(Safetensors, LoadToCuda) {
    const auto path = temp_file("cuda.safetensors");
    st::save(path, check_tensors());
    const auto tensors = st::load(path, torch::kCUDA);
    EXPECT_TRUE(tensors.at("bf16").is_cuda());
    expect_same(tensors, check_tensors());
    fs::remove(path);
}

TEST(Safetensors, ReadsPythonFile) {
    const auto path = test_env().golden_dir / "train" / "safetensors_check.safetensors";
    if (!fs::exists(path))
        GTEST_SKIP() << "missing " << path << " (run tools/export_train_golden.py)";
    expect_same(st::load(path), check_tensors());
    EXPECT_EQ(st::load_metadata(path).at("source"), "python");
}
