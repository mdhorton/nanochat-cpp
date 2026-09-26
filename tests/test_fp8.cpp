// FP8 quantization kernels vs fp8.py's torch ops (the model-level checks against Python are in test_gpt/test_train).
#include <gtest/gtest.h>

#include "nanochat/model/fp8.h"

using namespace nanochat;

TEST(Fp8, FusedQuantizeMatchesTorchOps) {
  torch::manual_seed(0);
  const auto opts = torch::TensorOptions().device(torch::kCUDA);
  // partial tiles in both dims, fp32 weights and bf16 activations
  for (const auto& [rows, cols] : {std::pair<int64_t, int64_t>{100, 24}, {1000, 136}, {64, 64}, {4097, 768}}) {
    for (const auto dtype : {torch::kBFloat16, torch::kFloat32}) {
      const auto x = (torch::randn({rows, cols}, opts) * 3).to(dtype);
      for (const auto format : {torch::kFloat8_e4m3fn, torch::kFloat8_e5m2}) {
        const auto got = quantize_fp8(x, format), want = quantize_fp8(x, format, false);
        const auto bits = [](const torch::Tensor& t) {
          return t.view(torch::kUInt8);
        };
        EXPECT_TRUE(torch::equal(bits(got.data), bits(want.data))) << rows << "x" << cols;
        EXPECT_TRUE(torch::equal(bits(got.data_t), bits(want.data.t().contiguous()))) << rows << "x" << cols;
        EXPECT_EQ(got.inv_scale.item<float>(), want.inv_scale.item<float>());
      }
    }
  }
}
