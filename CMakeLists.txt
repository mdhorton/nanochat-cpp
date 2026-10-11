cmake_minimum_required(VERSION 3.28)

project(nanochat_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) 

# before any find_package: deps set this, so a fresh configure lacks -pthread and the next one rebuilds everything
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(PkgConfig REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(NvidiaCutlass REQUIRED) # headers only
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
# torch's NVIDIA wheels: cu13 puts every library under nvidia/cu13, cu12 has a directory per library
set(NV_DIR "${TORCH_DIR}/../nvidia")
if(EXISTS "${NV_DIR}/cu13")
    set(NV_CUDART "${NV_DIR}/cu13/lib/libcudart.so.13")
    set(NV_NVRTC "${NV_DIR}/cu13/lib/libnvrtc.so.13")
    set(NV_CUBLASLT "${NV_DIR}/cu13/lib/libcublasLt.so.13")
    set(NV_INCLUDE_DIRS "${NV_DIR}/cu13/include")
    set(NV_RPATH "${NV_DIR}/cu13/lib")
else()
    set(NV_CUDART "${NV_DIR}/cuda_runtime/lib/libcudart.so.12")
    set(NV_NVRTC "${NV_DIR}/cuda_nvrtc/lib/libnvrtc.so.12")
    set(NV_CUBLASLT "${NV_DIR}/cublas/lib/libcublasLt.so.12")
    file(GLOB NV_INCLUDE_DIRS "${NV_DIR}/*/include")
    set(NV_RPATH "${NV_DIR}/cuda_runtime/lib:${NV_DIR}/cuda_nvrtc/lib:${NV_DIR}/cublas/lib")
endif()
set(CUDA_nvrtc_LIBRARY "${NV_NVRTC}") # not in the conda env

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_RUNTIME_LIBRARY None) # use torch's libcudart
enable_language(CUDA)

# torch sets the nvcc -gencode flags from this. block-scaled mma.sync needs an arch-specific target:
# f (cuda 12.9+) runs on sm_120/sm_121, a (cuda 12.8) only on sm_120
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9)
    set(TORCH_CUDA_ARCH_LIST 12.0f)
else()
    set(TORCH_CUDA_ARCH_LIST 12.0a)
endif()

unset(CMAKE_CUDA_ARCHITECTURES CACHE) # torch warns if set
set(kineto_LIBRARY "${TORCH_DIR}/lib/libtorch_cpu.so") # kineto is built into libtorch_cpu
find_package(Torch REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES OFF) # torch adds the -gencode flags

# link and load torch's libcudart, not the copy nvcc brings in. The rpath goes first: conda's LDFLAGS put the env's
# lib dir (with a libcudart symlink) at the front.
set_property(TARGET CUDA::cudart PROPERTY IMPORTED_LOCATION "${NV_CUDART}")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,${NV_RPATH} ${CMAKE_EXE_LINKER_FLAGS}")

add_library(nanochat
        src/nanochat/flags.cpp
        src/nanochat/utf8.cpp
        src/nanochat/data/dataset.cpp
        src/nanochat/tokenizer/base64.cpp
        src/nanochat/tokenizer/bpe.cpp
        src/nanochat/tokenizer/splitter.cpp
        src/nanochat/tokenizer/tokenizer.cpp)
target_include_directories(nanochat PUBLIC src)
target_compile_definitions(nanochat PUBLIC NANOCHAT_PROJECT_DIR="${PROJECT_SOURCE_DIR}")
# stdc++ first: torch 2.11's libtorch.so exports a broken std::filesystem that would otherwise take precedence
target_link_libraries(nanochat PUBLIC stdc++
        PkgConfig::PCRE2 Arrow::arrow_shared Parquet::parquet_shared nlohmann_json::nlohmann_json Threads::Threads)

# pre-training (libtorch)
add_library(nanochat_train
        src/nanochat/safetensors.cpp
        src/nanochat/data/dataloader.cpp
        src/nanochat/model/gpt.cpp
        src/nanochat/model/attention/flash.cpp
        src/nanochat/model/attention/flash_bwd_kernel.cu
        src/nanochat/model/attention/flash_kernel.cu
        src/nanochat/model/attention/mx_attention.cpp
        src/nanochat/model/attention/mx_attention_kernel.cu
        src/nanochat/model/attention/mx_flash.cpp
        src/nanochat/model/attention/mx_flash_bwd_kernel.cu
        src/nanochat/model/attention/mx_flash_kernel.cu
        src/nanochat/model/attention/rotary_norm.cpp
        src/nanochat/model/attention/rotary_norm_kernel.cu
        src/nanochat/model/fp8/fp8.cpp
        src/nanochat/model/fp8/fp8_kernel.cu
        src/nanochat/model/fp8/mx_gemm.cpp
        src/nanochat/model/fp8/mx_gemm_kernel.cu
        src/nanochat/model/fp8/qkv_kernel.cu
        src/nanochat/model/ops/backout.cpp
        src/nanochat/model/ops/backout_kernel.cu
        src/nanochat/model/ops/embedding.cpp
        src/nanochat/model/ops/embedding_kernel.cu
        src/nanochat/model/ops/relu_square.cpp
        src/nanochat/model/ops/relu_square_kernel.cu
        src/nanochat/model/ops/residual_norm.cpp
        src/nanochat/model/ops/residual_norm_kernel.cu
        src/nanochat/model/ops/smear.cpp
        src/nanochat/model/ops/smear_kernel.cu
        src/nanochat/model/ops/softcap_ce.cpp
        src/nanochat/model/ops/softcap_ce_kernel.cu
        src/nanochat/train/adamw_kernel.cu
        src/nanochat/train/checkpoint.cpp
        src/nanochat/train/dist.cpp
        src/nanochat/train/loss_eval.cpp
        src/nanochat/train/muon_kernel.cu
        src/nanochat/train/optim.cpp
        src/nanochat/train/trainer.cpp)
target_link_libraries(nanochat_train PUBLIC nanochat ${TORCH_LIBRARIES} "${NV_CUBLASLT}")
# ATen/cuda headers need the CUDA library headers (cuBLAS, cuSPARSE, ...), which come with the NVIDIA wheels
target_include_directories(nanochat_train SYSTEM PUBLIC ${NV_INCLUDE_DIRS})
target_include_directories(nanochat_train SYSTEM PRIVATE "${TORCH_DIR}/../nvidia/nccl/include") # ProcessGroupNCCL
target_link_libraries(nanochat_train PRIVATE nvidia::cutlass::cutlass nvidia::cutlass::tools::util)
# -lineinfo: source lines in ncu (no codegen change)
target_compile_options(nanochat_train PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
# git branch for the metrics header, refreshed every build
set(GIT_INFO_H "${CMAKE_BINARY_DIR}/generated/nanochat/git_info.h")
add_custom_target(git_info
        COMMAND ${CMAKE_COMMAND} -DSRC=${PROJECT_SOURCE_DIR} -DOUT=${GIT_INFO_H} -P ${PROJECT_SOURCE_DIR}/cmake/git_info.cmake
        BYPRODUCTS ${GIT_INFO_H})
add_dependencies(nanochat_train git_info)
target_include_directories(nanochat_train PRIVATE "${CMAKE_BINARY_DIR}/generated")
option(NANOCHAT_WARN_SPILLS "warn on register spills (pixi run build --warn-spills)" OFF)
if(NANOCHAT_WARN_SPILLS)
    target_compile_options(nanochat_train PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>)
endif()

add_executable(tok_train apps/tok_train.cpp)
target_link_libraries(tok_train PRIVATE nanochat)

add_executable(tok_eval apps/tok_eval.cpp)
target_link_libraries(tok_eval PRIVATE nanochat)

add_executable(base_train apps/base_train.cpp)
target_link_libraries(base_train PRIVATE nanochat_train)

# cuBLASLt algorithm search for the MX GEMM shapes
add_executable(bench_gemm tools/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE nanochat_train)

# attention backends (FA2, ours, cuDNN, memory-efficient) at a micro-step's shapes
add_executable(bench_attention tools/bench_attention.cpp)
target_link_libraries(bench_attention PRIVATE nanochat_train)

# tests (tests/CMakeLists.txt): enable_testing here so ctest finds them from the build root
enable_testing()
add_subdirectory(tests)
