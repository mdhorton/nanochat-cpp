// libtorch smoke tests: CUDA, bf16 matmul, SDPA (causal and sliding-window mask).
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace {

    const auto kCuda = torch::TensorOptions().device(torch::kCUDA);

} // namespace

TEST(Torch, CudaAvailable) {
    ASSERT_TRUE(torch::cuda::is_available());
    EXPECT_GE(torch::cuda::device_count(), 1u);
}

TEST(Torch, Bf16Matmul) {
    auto a = torch::randn({128, 64}, kCuda);
    auto b = torch::randn({64, 32}, kCuda);
    auto ref = torch::matmul(a, b);
    auto got = torch::matmul(a.to(torch::kBFloat16), b.to(torch::kBFloat16)).to(torch::kFloat32);
    EXPECT_TRUE(torch::allclose(got, ref, 5e-2, 5e-1));
}

TEST(Torch, SdpaCausalMatchesMask) {
    const int64_t B = 2, H = 4, T = 64, D = 32;
    auto opts = kCuda.dtype(torch::kBFloat16);
    auto q = torch::randn({B, H, T, D}, opts), k = torch::randn({B, H, T, D}, opts), v = torch::randn({B, H, T, D}, opts);
    auto causal = at::scaled_dot_product_attention(q, k, v, {}, 0.0, /*is_causal=*/true);
    auto mask = torch::ones({T, T}, kCuda.dtype(torch::kBool)).tril();
    auto masked = at::scaled_dot_product_attention(q, k, v, mask);
    EXPECT_TRUE(torch::allclose(causal.to(torch::kFloat32), masked.to(torch::kFloat32), 2e-2, 2e-2));
}

TEST(Torch, SdpaSlidingWindow) {
    // window w: each query sees keys (row - w)..row, i.e. w + 1 keys
    const int64_t T = 16, w = 3, D = 8;
    auto q = torch::randn({1, 1, T, D}, kCuda), k = torch::randn({1, 1, T, D}, kCuda), v = torch::randn({1, 1, T, D}, kCuda);
    auto row = torch::arange(T, kCuda).unsqueeze(1), col = torch::arange(T, kCuda).unsqueeze(0);
    auto mask = (col <= row).logical_and(row - col <= w);
    auto y = at::scaled_dot_product_attention(q, k, v, mask);
    // reference for the last query: softmax over its w + 1 keys
    auto qi = q[0][0][T - 1], ks = k[0][0].slice(0, T - 1 - w, T), vs = v[0][0].slice(0, T - 1 - w, T);
    auto p = torch::softmax(torch::matmul(ks, qi) / std::sqrt(static_cast<double>(D)), 0);
    EXPECT_TRUE(torch::allclose(y[0][0][T - 1], torch::matmul(p, vs), 1e-4, 1e-4));
}
