#include "nanochat/model/flash_attention_kernel.h"

#include <flash_fwd_launch_template.h>

namespace nanochat::kernels {

namespace fl = FLASH_NAMESPACE;
using T = cutlass::bfloat16_t;

// Causal: 128 x 64 tiles (64 KB of shared memory), as FlashAttention picks for sm80. Local (a left window): 64 x 64
// (48 KB, 2 blocks per SM), as it picks for sm86 / sm89's causal: ~20% faster on d12's windows of 512.
void flash_fwd(fl::Flash_fwd_params& p, cudaStream_t stream) {
  if (p.is_causal)
    fl::run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, T>, false, true>(p, stream);
  else
    fl::run_flash_fwd<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, T>, false, false>(p, stream);
}

} // namespace nanochat::kernels
