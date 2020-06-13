cmake_minimum_required(VERSION 3.13)  # oldest version used for testing
project(bittranspose VERSION 0.1 LANGUAGES C CXX)

option(BITTRANSPOSE_USE_AVX2 "Use AVX2 intrinsics instead of plain C")

add_library(bittranspose)

# Combine multiple sources into a single one to allow inlining.
set_target_properties(bittranspose PROPERTIES UNITY_BUILD_MODE GROUP)

if(${BITTRANSPOSE_USE_AVX2})
  target_compile_options(bittranspose PRIVATE "-mavx" "-mavx2")
  set(SOURCE_FILES
    src/transpose_square_common.c
    src/transpose_square_avx2.c
  )
else()
  set(SOURCE_FILES
    src/transpose_square_common.c
    src/transpose_square_plain.c
  )
endif()
target_sources(bittranspose PRIVATE ${SOURCE_FILES})
# Put all sources into the same UNITY_GROUP
set_source_files_properties(${SOURCE_FILES} PROPERTIES UNITY_GROUP "unity_group")

target_include_directories(bittranspose PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
