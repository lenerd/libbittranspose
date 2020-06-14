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

#ifndef BITTRANSPOSE_TEST_DATA_HPP
#define BITTRANSPOSE_TEST_DATA_HPP

#include <array>
#include <cstdint>
#include <immintrin.h>

// input/output pairs for square matrix transposition
extern const std::array<std::uint8_t, 8> input_8x8 alignas(32);
extern const std::array<std::uint8_t, 8> output_8x8 alignas(32);
extern const std::array<std::uint8_t, 4 * 8> input_8x8_packed alignas(32);
extern const std::array<std::uint8_t, 4 * 8> output_8x8_packed alignas(32);
extern const std::array<std::uint16_t, 16> input_16x16 alignas(32);
extern const std::array<std::uint16_t, 16> output_16x16 alignas(32);
extern const std::array<std::uint32_t, 32> input_32x32 alignas(32);
extern const std::array<std::uint32_t, 32> output_32x32 alignas(32);
extern const std::array<std::uint64_t, 64> input_64x64 alignas(32);
extern const std::array<std::uint64_t, 64> output_64x64 alignas(32);
extern const std::array<__m128i, 128> input_128x128 alignas(32);
extern const std::array<__m128i, 128> output_128x128 alignas(32);

// input/output pairs for rectangular matrix transposition
extern const std::array<std::uint8_t, 8 * 9> input_8x72 alignas(32);
extern const std::array<std::uint8_t, 8 * 9> output_8x72 alignas(32);
extern const std::array<std::uint16_t, 16 * 9> input_16x144 alignas(32);
extern const std::array<std::uint16_t, 16 * 9> output_16x144 alignas(32);
extern const std::array<std::uint32_t, 32 * 9> input_32x288 alignas(32);
extern const std::array<std::uint32_t, 32 * 9> output_32x288 alignas(32);
extern const std::array<std::uint64_t, 64 * 9> input_64x576 alignas(32);
extern const std::array<std::uint64_t, 64 * 9> output_64x576 alignas(32);
extern const std::array<__m128i, 128 * 9> input_128x1152 alignas(32);
extern const std::array<__m128i, 128 * 9> output_128x1152 alignas(32);

#endif // BITTRANSPOSE_TEST_DATA_HPP
