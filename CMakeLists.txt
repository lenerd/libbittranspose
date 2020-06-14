cmake_minimum_required(VERSION 3.13)  # oldest version used for testing
project(bittranspose VERSION 0.1 LANGUAGES C CXX)

option(BITTRANSPOSE_USE_AVX2 "Use AVX2 intrinsics instead of plain C")
option(BITTRANSPOSE_BUILD_TESTS "Build unit tests")
option(BITTRANSPOSE_BUILD_BENCHMARKS "Build benchmarks")

add_library(bittranspose)

# Combine multiple sources into a single one to allow inlining.
set_target_properties(bittranspose PROPERTIES UNITY_BUILD_MODE GROUP)

if(${BITTRANSPOSE_USE_AVX2})
  target_compile_options(bittranspose PRIVATE "-mavx" "-mavx2")
  set(SOURCE_FILES
    src/transpose_square_common.c
    src/transpose_square_avx2.c
    src/transpose_rectangular_avx2.c
  )
else()
  set(SOURCE_FILES
    src/transpose_square_common.c
    src/transpose_square_plain.c
    src/transpose_rectangular_plain.c
  )
endif()
target_sources(bittranspose PRIVATE ${SOURCE_FILES})
# Put all sources into the same UNITY_GROUP
set_source_files_properties(${SOURCE_FILES} PROPERTIES UNITY_GROUP "unity_group")

target_include_directories(bittranspose PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")


if(BITTRANSPOSE_BUILD_TESTS)
  add_subdirectory(extern/Catch2)

  add_executable(test
    test/main.cpp
    test/test_data_square.cpp
    test/test_data_rectangular.cpp
    test/test_square_transpose.cpp
    test/test_rectangular_transpose.cpp
  )
  target_link_libraries(test bittranspose Catch2::Catch2)
endif()


if(BITTRANSPOSE_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Don't build tests for the benchmark library")
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Don't install the benchmark library")
  add_subdirectory(extern/benchmark)

  add_executable(transpose_benchmark
    benchmark/bench_square_transpose.cpp
    benchmark/bench_rectangular_transpose.cpp
  )
  target_link_libraries(transpose_benchmark bittranspose benchmark::benchmark benchmark::benchmark_main)
endif()
