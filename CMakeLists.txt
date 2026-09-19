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

# Core library (grows each phase)
add_library(gpt2_core
  src/common/device.cpp
)
target_include_directories(gpt2_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gpt2_core
  PUBLIC ${TORCH_LIBRARIES}
  PRIVATE gpt2_options)

# Tools
add_executable(hello_tensor tools/hello_tensor.cpp)
target_link_libraries(hello_tensor PRIVATE gpt2_core gpt2_options)

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
  )
  target_link_libraries(gpt2_tests PRIVATE gpt2_core gpt2_options GTest::gtest_main)
  target_compile_definitions(gpt2_tests PRIVATE GPT2_DEFAULT_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/weights")
  gtest_discover_tests(gpt2_tests DISCOVERY_TIMEOUT 60)
endif()
