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

TEST_CASE("square 8x8 bit transposition", "[8] [square] [direct]") {
    std::uint64_t input;
    std::uint64_t output;
    std::memcpy(&input, input_8x8.data(), sizeof(input));
    std::memcpy(&output, output_8x8.data(), sizeof(output));
    std::array<std::uint8_t, 8> computed_array;
    std::uint64_t computed = transpose_bit_8x8_direct(input);
    REQUIRE(computed == output);
}

TEST_CASE("square 8x8 bit transpositions", "[8] [square] [inplace]") {
    std::array<std::uint8_t, 8> computed;
    std::copy(std::begin(input_8x8), std::end(input_8x8), std::begin(computed));
    transpose_bit_8x8_inplace(computed.data());
    REQUIRE(computed == output_8x8);
}

TEST_CASE("packed square 8x8 bit transpositions", "[8] [square] [packed] [inplace]") {
    std::array<std::uint8_t, 4 * 8> computed;
    std::copy(std::begin(input_8x8_packed), std::end(input_8x8_packed), std::begin(computed));
    transpose_bit_8x8_packed_x4_inplace(computed.data());
    REQUIRE(computed == output_8x8_packed);
}

TEST_CASE("packed square 8x8 bit transpositions aligned", "[8] [square] [packed] [inplace] [aligned]") {
    std::array<std::uint8_t, 4 * 8> computed alignas(32);
    std::copy(std::begin(input_8x8_packed), std::end(input_8x8_packed), std::begin(computed));
    transpose_bit_8x8_packed_x4_inplace_aligned(computed.data());
    REQUIRE(computed == output_8x8_packed);
}

TEST_CASE("square 16x16 bit transpositions", "[16] [square]") {
    std::array<std::uint16_t, 16> computed;
    std::copy(std::begin(input_16x16), std::end(input_16x16), std::begin(computed));
    transpose_bit_16x16_inplace(computed.data());
    REQUIRE(computed == output_16x16);
}

TEST_CASE("square 16x16 bit transpositions aligned", "[16] [square] [aligned]") {
    std::array<std::uint16_t, 16> computed alignas(32);
    std::copy(std::begin(input_16x16), std::end(input_16x16), std::begin(computed));
    transpose_bit_16x16_inplace_aligned(computed.data());
    REQUIRE(computed == output_16x16);
}

TEST_CASE("square 32x32 bit transpositions", "[32] [square]") {
    std::array<std::uint32_t, 32> computed;
    std::copy(std::begin(input_32x32), std::end(input_32x32), std::begin(computed));
    transpose_bit_32x32_inplace(computed.data());
    REQUIRE(computed == output_32x32);
}

TEST_CASE("square 32x32 bit transpositions aligned", "[32] [square] [aligned]") {
    std::array<std::uint32_t, 32> computed alignas(32);
    std::copy(std::begin(input_32x32), std::end(input_32x32), std::begin(computed));
    transpose_bit_32x32_inplace_aligned(computed.data());
    REQUIRE(computed == output_32x32);
}

TEST_CASE("square 64x64 bit transpositions", "[64] [square]") {
    std::array<std::uint64_t, 64> computed;
    std::copy(std::begin(input_64x64), std::end(input_64x64), std::begin(computed));
    transpose_bit_64x64_inplace(computed.data());
    REQUIRE(computed == output_64x64);
}

TEST_CASE("square 64x64 bit transpositions aligned", "[64] [square] [aligned]") {
    std::array<std::uint64_t, 64> computed alignas(32);
    std::copy(std::begin(input_64x64), std::end(input_64x64), std::begin(computed));
    transpose_bit_64x64_inplace_aligned(computed.data());
    REQUIRE(computed == output_64x64);
}

TEST_CASE("square 128x128 bit transpositions", "[128] [square]") {
    std::array<__m128i, 128> computed;
    std::copy(std::begin(input_128x128), std::end(input_128x128), std::begin(computed));
    transpose_bit_128x128_inplace(computed.data());
    REQUIRE(std::memcmp(computed.data(), output_128x128.data(), computed.size() * sizeof(__m128i))
            == 0);
}

TEST_CASE("square 128x128 bit transpositions aligned", "[128] [square] [aligned]") {
    std::array<__m128i, 128> computed alignas(32);
    std::copy(std::begin(input_128x128), std::end(input_128x128), std::begin(computed));
    transpose_bit_128x128_inplace_aligned(computed.data());
    REQUIRE(std::memcmp(computed.data(), output_128x128.data(), computed.size() * sizeof(__m128i))
            == 0);
}
