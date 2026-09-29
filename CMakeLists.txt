cmake_minimum_required(VERSION 3.28)

project(nanochat_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)

# libtorch from the pixi env's PyPI torch wheel.
set(PIXI_ENV "${PROJECT_SOURCE_DIR}/.pixi/envs/default" CACHE PATH "pixi environment prefix")
file(GLOB TORCH_DIR "${PIXI_ENV}/lib/python3*/site-packages/torch")
list(GET TORCH_DIR 0 TORCH_DIR) # conda also adds a python3.1 -> python3.12 symlink
file(REAL_PATH "${TORCH_DIR}" TORCH_DIR)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})

set(CUDA_TOOLKIT_ROOT_DIR "${PIXI_ENV}/targets/x86_64-linux")
set(CMAKE_CUDA_COMPILER "${PIXI_ENV}/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CUDA_nvrtc_LIBRARY "${TORCH_DIR}/../nvidia/cu13/lib/libnvrtc.so.13") # not in the conda env
set(TORCH_CUDA_ARCH_LIST 12.0a) # torch sets the nvcc -gencode flags from this; a: block-scaled mma.sync

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_RUNTIME_LIBRARY None) # use torch's libcudart
enable_language(CUDA)

unset(CMAKE_CUDA_ARCHITECTURES CACHE) # torch warns if set
set(kineto_LIBRARY "${TORCH_DIR}/lib/libtorch_cpu.so") # kineto is built into libtorch_cpu
find_package(Torch REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES OFF) # torch adds the -gencode flags

# link and load torch's libcudart, not the copy nvcc brings in. The rpath goes first: conda's LDFLAGS put the env's
# lib dir (with a libcudart.so.13 symlink) at the front.
set_property(TARGET CUDA::cudart PROPERTY IMPORTED_LOCATION "${TORCH_DIR}/../nvidia/cu13/lib/libcudart.so.13")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,${TORCH_DIR}/../nvidia/cu13/lib ${CMAKE_EXE_LINKER_FLAGS}")

add_library(nanochat
        src/nanochat/dataset.cpp
        src/nanochat/flags.cpp
        src/nanochat/tokenizer/base64.cpp
        src/nanochat/tokenizer/bpe.cpp
        src/nanochat/tokenizer/splitter.cpp
        src/nanochat/tokenizer/tokenizer.cpp
        src/nanochat/tokenizer/utf8.cpp)
target_include_directories(nanochat PUBLIC src)
target_compile_definitions(nanochat PUBLIC NANOCHAT_CACHE_DIR="${PROJECT_SOURCE_DIR}/cache")
target_link_libraries(nanochat PUBLIC
        PkgConfig::PCRE2 Arrow::arrow_shared Parquet::parquet_shared nlohmann_json::nlohmann_json Threads::Threads)

# pre-training (libtorch)
add_library(nanochat_train
        src/nanochat/model/backout.cpp
        src/nanochat/model/backout_kernel.cu
        src/nanochat/model/embedding.cpp
        src/nanochat/model/embedding_kernel.cu
        src/nanochat/model/fp8.cpp
        src/nanochat/model/flash.cpp
        src/nanochat/model/flash_bwd_kernel.cu
        src/nanochat/model/flash_bwd_mx_kernel.cu
        src/nanochat/model/flash_kernel.cu
        src/nanochat/model/fp8_kernel.cu
        src/nanochat/model/gpt.cpp
        src/nanochat/model/mx_attention.cpp
        src/nanochat/model/mx_attention_kernel.cu
        src/nanochat/model/mx_flash.cpp
        src/nanochat/model/mx_flash_kernel.cu
        src/nanochat/model/mx_gemm.cpp
        src/nanochat/model/mx_gemm_kernel.cu
        src/nanochat/model/nvfp4.cpp
        src/nanochat/model/nvfp4_gemm_kernel.cu
        src/nanochat/model/nvfp4_kernel.cu
        src/nanochat/model/nvfp4_sim.cpp
        src/nanochat/model/nvfp4_sim_kernel.cu
        src/nanochat/model/qkv_kernel.cu
        src/nanochat/model/relu_square.cpp
        src/nanochat/model/relu_square_kernel.cu
        src/nanochat/model/residual_norm.cpp
        src/nanochat/model/residual_norm_kernel.cu
        src/nanochat/model/rotary_norm.cpp
        src/nanochat/model/rotary_norm_kernel.cu
        src/nanochat/model/smear.cpp
        src/nanochat/model/smear_kernel.cu
        src/nanochat/model/softcap_ce.cpp
        src/nanochat/model/softcap_ce_kernel.cu
        src/nanochat/train/adamw_kernel.cu
        src/nanochat/train/checkpoint.cpp
        src/nanochat/train/dataloader.cpp
        src/nanochat/train/dist.cpp
        src/nanochat/train/loss_eval.cpp
        src/nanochat/train/optim.cpp
        src/nanochat/train/safetensors.cpp
        src/nanochat/train/trainer.cpp)
target_link_libraries(nanochat_train PUBLIC nanochat ${TORCH_LIBRARIES} "${TORCH_DIR}/../nvidia/cu13/lib/libcublasLt.so.13")
# ATen/cuda headers need the CUDA library headers (cuBLAS, cuSPARSE, ...), which come with the NVIDIA wheels
target_include_directories(nanochat_train SYSTEM PUBLIC "${TORCH_DIR}/../nvidia/cu13/include")
target_include_directories(nanochat_train SYSTEM PRIVATE "${TORCH_DIR}/../nvidia/nccl/include") # ProcessGroupNCCL
# CUTLASS (headers only)
target_include_directories(nanochat_train SYSTEM PRIVATE
        "${PROJECT_SOURCE_DIR}/external/cutlass/include" "${PROJECT_SOURCE_DIR}/external/cutlass/tools/util/include")
# warn on register spills; -lineinfo: source lines in ncu (no codegen change)
target_compile_options(nanochat_train PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills -lineinfo>)

add_executable(tok_train apps/tok_train.cpp)
target_link_libraries(tok_train PRIVATE nanochat)

add_executable(tok_eval apps/tok_eval.cpp)
target_link_libraries(tok_eval PRIVATE nanochat)

add_executable(base_train apps/base_train.cpp)
target_link_libraries(base_train PRIVATE nanochat_train)

# cuBLASLt algorithm search for the MX GEMM shapes
add_executable(bench_gemm apps/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE nanochat_train)

# attention backends (FA2, ours, cuDNN, memory-efficient) at a micro-step's shapes
add_executable(bench_attention apps/bench_attention.cpp)
target_link_libraries(bench_attention PRIVATE nanochat_train)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)
add_executable(nanochat_tests
        tests/main.cpp
        tests/test_dataset.cpp
        tests/test_golden.cpp
        tests/test_splitter.cpp
        tests/test_tokenizer.cpp
        tests/test_utf8.cpp)
target_link_libraries(nanochat_tests PRIVATE nanochat GTest::gtest)
gtest_discover_tests(nanochat_tests DISCOVERY_TIMEOUT 30)

add_executable(train_tests
        tests/main.cpp
        tests/test_backout.cpp
        tests/test_checkpoint.cpp
        tests/test_dataloader.cpp
        tests/test_dist.cpp
        tests/test_embedding.cpp
        tests/test_flash.cpp
        tests/test_fp8.cpp
        tests/test_gpt.cpp
        tests/test_loss_eval.cpp
        tests/test_mx_flash.cpp
        tests/test_nvfp4.cpp
        tests/test_optim.cpp
        tests/test_qkv.cpp
        tests/test_relu_square.cpp
        tests/test_residual_norm.cpp
        tests/test_rotary_norm.cpp
        tests/test_safetensors.cpp
        tests/test_smear.cpp
        tests/test_softcap_ce.cpp
        tests/test_torch.cpp
        tests/test_train.cpp)
target_link_libraries(train_tests PRIVATE nanochat_train GTest::gtest)
target_compile_definitions(train_tests PRIVATE NANOCHAT_DIST_WORKER="$<TARGET_FILE:dist_worker>")

add_executable(dist_worker tests/dist_worker.cpp)
target_link_libraries(dist_worker PRIVATE nanochat_train)
add_dependencies(train_tests dist_worker)
gtest_discover_tests(train_tests DISCOVERY_TIMEOUT 60)
