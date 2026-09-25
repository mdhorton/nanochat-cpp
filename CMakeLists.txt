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

# libtorch from the pixi env's PyPI torch wheel. Its FindCUDA-based config needs the conda CUDA layout spelled out.
set(PIXI_ENV "${PROJECT_SOURCE_DIR}/.pixi/envs/default" CACHE PATH "pixi environment prefix")
file(GLOB TORCH_DIR "${PIXI_ENV}/lib/python3*/site-packages/torch")
list(GET TORCH_DIR 0 TORCH_DIR) # conda also adds a python3.1 -> python3.12 symlink
file(REAL_PATH "${TORCH_DIR}" TORCH_DIR)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
set(CUDA_TOOLKIT_ROOT_DIR "${PIXI_ENV}/targets/x86_64-linux")
set(CMAKE_CUDA_COMPILER "${PIXI_ENV}/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CUDA_nvrtc_LIBRARY "${TORCH_DIR}/../nvidia/cu13/lib/libnvrtc.so.13") # not in the conda env
find_package(Torch REQUIRED)

add_library(nanochat
        src/nanochat/base64.cpp
        src/nanochat/bpe.cpp
        src/nanochat/dataset.cpp
        src/nanochat/flags.cpp
        src/nanochat/splitter.cpp
        src/nanochat/tokenizer.cpp
        src/nanochat/utf8.cpp)
target_include_directories(nanochat PUBLIC src)
target_compile_definitions(nanochat PUBLIC NANOCHAT_CACHE_DIR="${PROJECT_SOURCE_DIR}/cache")
target_link_libraries(nanochat PUBLIC
        PkgConfig::PCRE2 Arrow::arrow_shared Parquet::parquet_shared nlohmann_json::nlohmann_json Threads::Threads)

# pre-training (libtorch)
add_library(nanochat_train
        src/nanochat/checkpoint.cpp
        src/nanochat/dataloader.cpp
        src/nanochat/gpt.cpp
        src/nanochat/loss_eval.cpp
        src/nanochat/optim.cpp
        src/nanochat/safetensors.cpp
        src/nanochat/softcap_ce.cpp
        src/nanochat/trainer.cpp)
target_link_libraries(nanochat_train PUBLIC nanochat ${TORCH_LIBRARIES})
# ATen/cuda headers need the CUDA library headers (cuBLAS, cuSPARSE, ...), which come with the NVIDIA wheels
target_include_directories(nanochat_train SYSTEM PUBLIC "${TORCH_DIR}/../nvidia/cu13/include")

add_executable(nanochat_cpp main.cpp)

add_executable(tok_train apps/tok_train.cpp)
target_link_libraries(tok_train PRIVATE nanochat)

add_executable(tok_eval apps/tok_eval.cpp)
target_link_libraries(tok_eval PRIVATE nanochat)

add_executable(base_train apps/base_train.cpp)
target_link_libraries(base_train PRIVATE nanochat_train)

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
        tests/test_checkpoint.cpp
        tests/test_dataloader.cpp
        tests/test_gpt.cpp
        tests/test_loss_eval.cpp
        tests/test_optim.cpp
        tests/test_safetensors.cpp
        tests/test_softcap_ce.cpp
        tests/test_torch.cpp
        tests/test_train.cpp)
target_link_libraries(train_tests PRIVATE nanochat_train GTest::gtest)
gtest_discover_tests(train_tests DISCOVERY_TIMEOUT 60)
