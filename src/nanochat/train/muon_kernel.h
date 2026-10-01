// Fused Muon update (no torch headers): optim.cpp's muon_update in a few passes over k stacked (m, n) fp32 matrices,
// Polar Express left to the host's GEMMs. Not bit-identical to the op path: norms are reduced in fp32 in another
// order, and the row equilibration's and Muon+'s norms are derived from sums of squares rather than re-measured after
// each rescale (they differ only by bf16 rounding).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace nanochat::kernels {

// Nesterov momentum and MuonEq + the Polar Express pre-normalization: mbuf = lerp(mbuf, grad, 1 - momentum), x =
// bf16(lerp(grad, mbuf, momentum)) with each row scaled to the mean row norm, then divided by its Frobenius norm *
// 1.01 + 1e-6. grad, mbuf: (k, m, n) fp32; x: (k, m, n) bf16 out; row_sq: k * m fp32 scratch.
void muon_pre(
      const float* grad, float* mbuf, void* x, float* row_sq, int64_t k, int64_t m, int64_t n, float momentum,
      cudaStream_t stream);

// Workspace floats muon_post needs
int64_t muon_post_scratch(int64_t k, int64_t m, int64_t n);

// Muon+ (Frobenius norm to sqrt(min(m, n))), NorMuon (smb: (k, m) when m >= n, else (k, n): lerp to the mean of g²
// along the other dim, then the per-row / column rsqrt scale renormalized to the norm before) and the cautious
// update: p -= lr * g + lr * wd * p * (g * p >= 0). x: Polar Express's (k, m, n) bf16; p: (k, m, n) fp32.
void muon_post(
      const void* x, float* p, float* smb, float* scratch, int64_t k, int64_t m, int64_t n, float lr, float wd,
      float beta2, cudaStream_t stream);

} // namespace nanochat::kernels
