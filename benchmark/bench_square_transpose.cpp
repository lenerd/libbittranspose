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

#include <benchmark/benchmark.h>
#include <bit_transpose.h>
#include <immintrin.h>
#include <random>

static void BM_transpose_bit_8x8_direct(benchmark::State& state) {
    for (auto _ : state) {
        transpose_bit_8x8_direct(0x13374247deadbeef);
    }
}
BENCHMARK(BM_transpose_bit_8x8_direct);

static void BM_transpose_bit_8x8_inplace(benchmark::State& state) {
    std::uint64_t matrix = 0x13374247deadbeef;

    for (auto _ : state) {
        transpose_bit_8x8_inplace(&matrix);
    }
}
BENCHMARK(BM_transpose_bit_8x8_inplace);

static void BM_transpose_bit_8x8_packed_x4_inplace(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint64_t> dist(0);
    std::array<std::uint64_t, 4> matrix alignas(32);

    for (auto _ : state) {
        transpose_bit_8x8_packed_x4_inplace(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_8x8_packed_x4_inplace);

static void BM_transpose_bit_16x16_inplace(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<std::uint16_t> dist(0);
    std::array<std::uint16_t, 16> matrix alignas(32);

    for (auto _ : state) {
        transpose_bit_16x16_inplace(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_16x16_inplace);

static void BM_transpose_bit_16x16_inplace_aligned(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<std::uint16_t> dist(0);
    std::array<std::uint16_t, 16> matrix alignas(32);

    for (auto _ : state) {
        transpose_bit_16x16_inplace_aligned(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_16x16_inplace_aligned);

static void BM_transpose_bit_32x32_inplace(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<std::uint32_t> dist(0);
    std::array<std::uint32_t, 32> matrix alignas(32);
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_32x32_inplace(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_32x32_inplace);

static void BM_transpose_bit_32x32_inplace_aligned(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<std::uint32_t> dist(0);
    std::array<std::uint32_t, 32> matrix alignas(32);
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_32x32_inplace_aligned(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_32x32_inplace_aligned);

static void BM_transpose_bit_64x64_inplace(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint64_t> dist(0);
    std::array<std::uint64_t, 64> matrix alignas(32);
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_64x64_inplace(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_64x64_inplace);

static void BM_transpose_bit_64x64_inplace_aligned(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint64_t> dist(0);
    std::array<std::uint64_t, 64> matrix alignas(32);
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_64x64_inplace_aligned(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_64x64_inplace_aligned);

static void BM_transpose_bit_128x128_inplace(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint64_t> dist(0);
    std::array<__m128i, 128> matrix alignas(32);
    std::generate(matrix.begin(), matrix.end(), [&] { return _mm_set_epi64x(dist(mt), dist(mt)); });

    for (auto _ : state) {
        transpose_bit_128x128_inplace(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_128x128_inplace);

static void BM_transpose_bit_128x128_inplace_aligned(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint64_t> dist(0);
    std::array<__m128i, 128> matrix alignas(32);
    std::generate(matrix.begin(), matrix.end(), [&] { return _mm_set_epi64x(dist(mt), dist(mt)); });

    for (auto _ : state) {
        transpose_bit_128x128_inplace_aligned(matrix.data());
    }
}
BENCHMARK(BM_transpose_bit_128x128_inplace_aligned);
