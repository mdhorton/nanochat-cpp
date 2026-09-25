cmake_minimum_required(VERSION 3.28)

project(nanochat_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(nlohmann_json REQUIRED)

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
        PkgConfig::PCRE2 Arrow::arrow_shared Parquet::parquet_shared nlohmann_json::nlohmann_json)

add_executable(nanochat_cpp main.cpp)

add_executable(tok_train apps/tok_train.cpp)
target_link_libraries(tok_train PRIVATE nanochat)

add_executable(tok_eval apps/tok_eval.cpp)
target_link_libraries(tok_eval PRIVATE nanochat)

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
