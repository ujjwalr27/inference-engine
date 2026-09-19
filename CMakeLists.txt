cmake_minimum_required(VERSION 3.24)
project(gpt2_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(GPT2_BUILD_TESTS "Build GoogleTest suite" ON)
option(GPT2_ENABLE_ASAN "Build with AddressSanitizer + UBSan" OFF)
option(GPT2_ENABLE_TSAN "Build with ThreadSanitizer" OFF)

if(GPT2_ENABLE_ASAN AND GPT2_ENABLE_TSAN)
  message(FATAL_ERROR "ASan and TSan cannot be enabled together")
endif()

# LibTorch: pass -DCMAKE_PREFIX_PATH=<libtorch dir> (local) or torch.utils.cmake_prefix_path (Kaggle)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
message(STATUS "Torch version: ${Torch_VERSION}")

# Project-wide warnings/sanitizers, applied only to our targets (not third-party)
add_library(gpt2_options INTERFACE)
target_compile_options(gpt2_options INTERFACE -Wall -Wextra -Wpedantic)
if(GPT2_ENABLE_ASAN)
  target_compile_options(gpt2_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(gpt2_options INTERFACE -fsanitize=address,undefined)
endif()
if(GPT2_ENABLE_TSAN)
  target_compile_options(gpt2_options INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(gpt2_options INTERFACE -fsanitize=thread)
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

# cpp-httplib picks up OpenSSL/zlib/brotli/zstd from the host when it finds them, which makes the
# build depend on what happens to be installed - and on GitHub's runners zstd is detected without
# defining the imported target it then links, breaking the configure step. The engine serves plain
# local HTTP for benchmarking, so every optional backend stays off and all machines build alike.
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(cpp_httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.27.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(cpp_httplib)

# Hugging Face tokenizers via mlc-ai/tokenizers-cpp (builds the Rust crate with cargo).
# SentencePiece is not needed for GPT-2's byte-level BPE, so it stays off.
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)

# Core library (grows each phase)
add_library(gpt2_core
  src/common/device.cpp
  src/io/safetensors.cpp
  src/cache/kv_cache.cpp
  src/model/config.cpp
  src/model/layers.cpp
  src/model/gpt2.cpp
  src/model/generate.cpp
  src/scheduler/padding.cpp
  src/scheduler/static_batch.cpp
  src/scheduler/request.cpp
  src/scheduler/queue.cpp
  src/scheduler/scheduler.cpp
  src/tokenizer/tokenizer.cpp
  src/server/http_server.cpp
)
target_include_directories(gpt2_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gpt2_core
  PUBLIC ${TORCH_LIBRARIES}
  PRIVATE nlohmann_json::nlohmann_json tokenizers_cpp httplib::httplib gpt2_options)

# Tools
add_executable(hello_tensor tools/hello_tensor.cpp)
target_link_libraries(hello_tensor PRIVATE gpt2_core gpt2_options)

add_executable(gpt2_forward tools/forward.cpp)
target_link_libraries(gpt2_forward PRIVATE gpt2_core gpt2_options)

add_executable(gpt2_generate tools/generate.cpp)
target_link_libraries(gpt2_generate PRIVATE gpt2_core gpt2_options)

add_executable(gpt2_bench tools/bench.cpp)
target_link_libraries(gpt2_bench PRIVATE gpt2_core gpt2_options)

add_executable(gpt2_serve tools/serve.cpp)
target_link_libraries(gpt2_serve PRIVATE gpt2_core gpt2_options)

# The load generator only speaks HTTP; it deliberately does not link the engine.
add_executable(gpt2_loadgen tools/loadgen.cpp)
target_link_libraries(gpt2_loadgen PRIVATE httplib::httplib nlohmann_json::nlohmann_json gpt2_options
                      Threads::Threads)

# Tests
if(GPT2_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)

  add_executable(gpt2_tests
    tests/test_smoke.cpp
    tests/test_safetensors.cpp
    tests/test_layers.cpp
    tests/test_gpt2_reference.cpp
    tests/test_kv_cache.cpp
    tests/test_generate.cpp
    tests/test_batch.cpp
    tests/test_concurrency.cpp
    tests/test_scheduler.cpp
    tests/test_server.cpp
    tests/test_gpu.cpp
    tests/test_tokenizer.cpp
  )
  target_link_libraries(gpt2_tests PRIVATE gpt2_core gpt2_options nlohmann_json::nlohmann_json httplib::httplib
                        GTest::gtest_main)
  target_compile_definitions(gpt2_tests PRIVATE GPT2_DEFAULT_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/weights")
  gtest_discover_tests(gpt2_tests DISCOVERY_TIMEOUT 60)
endif()
