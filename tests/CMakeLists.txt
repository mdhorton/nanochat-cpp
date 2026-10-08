# not in the default build (pixi run test builds them)
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(nanochat_tests EXCLUDE_FROM_ALL
        main.cpp
        test_dataset.cpp
        test_golden.cpp
        test_splitter.cpp
        test_tokenizer.cpp
        test_utf8.cpp)
target_link_libraries(nanochat_tests PRIVATE nanochat GTest::gtest)
gtest_discover_tests(nanochat_tests DISCOVERY_TIMEOUT 30)

add_executable(train_tests EXCLUDE_FROM_ALL
        main.cpp
        test_backout.cpp
        test_checkpoint.cpp
        test_dataloader.cpp
        test_dist.cpp
        test_embedding.cpp
        test_flash.cpp
        test_fp8.cpp
        test_gpt.cpp
        test_loss_eval.cpp
        test_mx_flash.cpp
        test_optim.cpp
        test_qkv.cpp
        test_relu_square.cpp
        test_residual_norm.cpp
        test_rotary_norm.cpp
        test_safetensors.cpp
        test_smear.cpp
        test_softcap_ce.cpp
        test_torch.cpp
        test_train.cpp)
target_link_libraries(train_tests PRIVATE nanochat_train GTest::gtest)
target_compile_definitions(train_tests PRIVATE NANOCHAT_DIST_WORKER="$<TARGET_FILE:dist_worker>")

add_executable(dist_worker EXCLUDE_FROM_ALL dist_worker.cpp)
target_link_libraries(dist_worker PRIVATE nanochat_train)
add_dependencies(train_tests dist_worker)
gtest_discover_tests(train_tests DISCOVERY_TIMEOUT 60)
