// The block-scale layout cuBLAS and CUTLASS read for MXFP8 (a scale per 32) and NVFP4 (per 16) operands.
#pragma once

#include <cstdint>

namespace nanochat {

// swizzled: 128x4 tiles of (row, block), 512 bytes each, row-major over tiles (tiles per 128 rows: blocks per row / 4)
__host__ __device__ __forceinline__ int64_t mx_scale_index(int64_t row, int64_t block, int64_t tiles) {
  return ((row >> 7) * tiles + (block >> 2)) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (block & 3);
}

} // namespace nanochat
