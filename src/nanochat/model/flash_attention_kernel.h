// FlashAttention-2 kernel launches (hdim 128, bf16) with tile configs chosen for sm_120 (~99 KB shared memory per
// block).
#pragma once

#include <cuda_runtime_api.h>

#include <flash.h>

namespace nanochat::kernels {

void flash_fwd(FLASH_NAMESPACE::Flash_fwd_params& params, cudaStream_t stream);
void flash_bwd(FLASH_NAMESPACE::Flash_bwd_params& params, cudaStream_t stream);

} // namespace nanochat::kernels
