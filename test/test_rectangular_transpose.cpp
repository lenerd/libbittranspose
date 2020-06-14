// MIT License
//
// Copyright (c) 2020 Lennart Braun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "test_data.hpp"
#include <bit_transpose.h>
#include <catch2/catch.hpp>
#include <cstdint>
#include <cstring>

TEST_CASE("rectangular 8xN bit transpositions", "[8] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<std::uint8_t, 8 * N_max> computed = {0};
    std::array<const std::uint8_t*, 8> input_ptrs;
    for (std::size_t i = 0; i < 8; ++i) {
        input_ptrs[i] = input_8x72.data() + i * N_max;
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), 0x00);
        transpose_bit_8xN(computed.data(), input_ptrs.data(), N);
        REQUIRE(std::memcmp(computed.data(), output_8x72.data(),
                            8 * N * sizeof(decltype(computed)::value_type))
                == 0);
    }
}

TEST_CASE("rectangular Nx8 bit transpositions", "[8] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<std::uint8_t, 8 * N_max> computed alignas(32) = {0};
    std::array<std::uint8_t*, 8> output_ptrs;
    std::array<const std::uint8_t*, 8> input_ptrs;
    for (std::size_t i = 0; i < 8; ++i) {
        output_ptrs[i] = computed.data() + i * N_max;
        input_ptrs[i] = input_8x72.data() + i * N_max;
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), 0x00);
        transpose_bit_Nx8(output_ptrs.data(), output_8x72.data(), N);
        for (std::size_t i = 0; i < 8; ++i) {
            REQUIRE(std::memcmp(computed.data() + i * N_max, input_8x72.data() + i * N_max,
                                N * sizeof(decltype(computed)::value_type))
                    == 0);
        }
    }
}

TEST_CASE("rectangular 16xN bit transpositions", "[16] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<std::uint16_t, 16 * N_max> computed alignas(32) = {0};
    std::array<const std::uint8_t*, 16> input_ptrs;
    for (std::size_t i = 0; i < 16; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(input_16x144.data() + i * N_max);
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), 0x00);
        transpose_bit_16xN(computed.data(), input_ptrs.data(), N);
        REQUIRE(std::memcmp(computed.data(), output_16x144.data(),
                            16 * N * sizeof(decltype(computed)::value_type))
                == 0);
    }
}

TEST_CASE("rectangular Nx16 bit transpositions", "[16] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<std::uint16_t, 16 * N_max> computed alignas(32) = {0};
    std::array<std::uint8_t*, 16> output_ptrs;
    std::array<const std::uint8_t*, 16> input_ptrs;
    for (std::size_t i = 0; i < 16; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(computed.data() + i * N_max);
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(input_16x144.data() + i * N_max);
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), 0x00);
        transpose_bit_Nx16(output_ptrs.data(), output_16x144.data(), N);
        for (std::size_t i = 0; i < 16; ++i) {
            REQUIRE(std::memcmp(computed.data() + i * N_max, input_16x144.data() + i * N_max,
                                N * sizeof(decltype(computed)::value_type))
                    == 0);
        }
    }
}

TEST_CASE("rectangular 32xN bit transpositions", "[32] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<std::uint32_t, 32 * N_max> computed alignas(32) = {0};
    std::array<const std::uint8_t*, 32> input_ptrs;
    for (std::size_t i = 0; i < 32; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(input_32x288.data() + i * N_max);
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), 0x00);
        transpose_bit_32xN(computed.data(), input_ptrs.data(), N);
        REQUIRE(std::memcmp(computed.data(), output_32x288.data(),
                            32 * N * sizeof(decltype(computed)::value_type))
                == 0);
    }
}

TEST_CASE("rectangular Nx32 bit transpositions", "[32] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<std::uint32_t, 32 * N_max> computed alignas(32) = {0};
    std::array<std::uint8_t*, 32> output_ptrs;
    std::array<const std::uint8_t*, 32> input_ptrs;
    for (std::size_t i = 0; i < 32; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(computed.data() + i * N_max);
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(input_32x288.data() + i * N_max);
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), 0x00);
        transpose_bit_Nx32(output_ptrs.data(), output_32x288.data(), N);
        for (std::size_t i = 0; i < 32; ++i) {
            REQUIRE(std::memcmp(computed.data() + i * N_max, input_32x288.data() + i * N_max,
                                N * sizeof(decltype(computed)::value_type))
                    == 0);
        }
    }
}

TEST_CASE("rectangular 64xN bit transpositions", "[64] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<std::uint64_t, 64 * N_max> computed alignas(32) = {0};
    std::array<const std::uint8_t*, 64> input_ptrs;
    for (std::size_t i = 0; i < 64; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(input_64x576.data() + i * N_max);
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), 0x00);
        transpose_bit_64xN(computed.data(), input_ptrs.data(), N);
        REQUIRE(std::memcmp(computed.data(), output_64x576.data(),
                            64 * N * sizeof(decltype(computed)::value_type))
                == 0);
    }
}

TEST_CASE("rectangular Nx64 bit transpositions", "[64] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<std::uint64_t, 64 * N_max> computed alignas(32) = {0};
    std::array<std::uint8_t*, 64> output_ptrs;
    std::array<const std::uint8_t*, 64> input_ptrs;
    for (std::size_t i = 0; i < 64; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(computed.data() + i * N_max);
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(input_64x576.data() + i * N_max);
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), 0x00);
        transpose_bit_Nx64(output_ptrs.data(), output_64x576.data(), N);
        for (std::size_t i = 0; i < 64; ++i) {
            REQUIRE(std::memcmp(computed.data() + i * N_max, input_64x576.data() + i * N_max,
                                N * sizeof(decltype(computed)::value_type))
                    == 0);
        }
    }
}

TEST_CASE("rectangular 128xN bit transpositions", "[128] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<__m128i, 128 * N_max> computed alignas(32) = {0};
    std::array<const std::uint8_t*, 128> input_ptrs;
    for (std::size_t i = 0; i < 128; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(input_128x1152.data() + i * N_max);
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), _mm_set1_epi64x(0x00));
        transpose_bit_128xN(reinterpret_cast<std::uint8_t*>(computed.data()), input_ptrs.data(), N);
        REQUIRE(std::memcmp(computed.data(), output_128x1152.data(),
                            128 * N * sizeof(decltype(computed)::value_type))
                == 0);
    }
}

TEST_CASE("rectangular Nx128 bit transpositions", "[128] [rectangular]") {
    constexpr std::size_t N_max = 9;
    std::array<__m128i, 128 * N_max> computed alignas(32) = {0};
    std::array<std::uint8_t*, 128> output_ptrs;
    std::array<const std::uint8_t*, 128> input_ptrs;
    for (std::size_t i = 0; i < 128; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(computed.data() + i * N_max);
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(input_128x1152.data() + i * N_max);
    }
    for (std::size_t N = 1; N <= N_max; ++N) {
        std::fill(std::begin(computed), std::end(computed), _mm_set1_epi64x(0x00));
        transpose_bit_Nx128(output_ptrs.data(),
                            reinterpret_cast<const std::uint8_t*>(output_128x1152.data()), N);
        for (std::size_t i = 0; i < 128; ++i) {
            REQUIRE(std::memcmp(computed.data() + i * N_max, input_128x1152.data() + i * N_max,
                                N * sizeof(decltype(computed)::value_type))
                    == 0);
        }
    }
}
