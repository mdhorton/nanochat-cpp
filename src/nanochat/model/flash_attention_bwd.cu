#include "nanochat/model/flash_attention_kernel.h"

#include <flash_bwd_launch_template.h>

namespace nanochat::kernels {

namespace fl = FLASH_NAMESPACE;
using T = cutlass::bfloat16_t;
// FlashAttention's pick for < 144 KB of shared memory (80 KB); the other configs that fit weren't faster
using Traits = Flash_bwd_kernel_traits<128, 64, 64, 8, 4, 2, 2, true, false, T>;

void flash_bwd(fl::Flash_bwd_params& p, cudaStream_t stream) {
  if (p.is_causal)
    fl::run_flash_bwd<Traits, false, true>(p, stream);
  else
    fl::run_flash_bwd<Traits, false, false>(p, stream);
}

} // namespace nanochat::kernels
