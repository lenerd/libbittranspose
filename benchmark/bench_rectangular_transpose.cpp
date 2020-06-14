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

static void BM_transpose_bit_8xN(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint16_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<std::uint8_t, 8 * N> matrix alignas(32);
    std::array<std::uint8_t, 8 * N> output alignas(32);
    std::array<const std::uint8_t*, 8> input_ptrs;
    for (std::size_t i = 0; i < 8; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_8xN(output.data(), input_ptrs.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_8xN);

static void BM_transpose_bit_Nx8(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint16_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<std::uint8_t, 8 * N> matrix alignas(32);
    std::array<std::uint8_t, 8 * N> output alignas(32);
    std::array<std::uint8_t*, 8> output_ptrs;
    for (std::size_t i = 0; i < 8; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(output.data() + i * N);
    }
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_Nx8(output_ptrs.data(), matrix.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_Nx8);

static void BM_transpose_bit_16xN(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint16_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<std::uint16_t, 16 * N> matrix alignas(32);
    std::array<std::uint16_t, 16 * N> output alignas(32);
    std::array<const std::uint8_t*, 16> input_ptrs;
    for (std::size_t i = 0; i < 16; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_16xN(output.data(), input_ptrs.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_16xN);

static void BM_transpose_bit_Nx16(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint16_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<std::uint16_t, 16 * N> matrix alignas(32);
    std::array<std::uint16_t, 16 * N> output alignas(32);
    std::array<std::uint8_t*, 16> output_ptrs;
    for (std::size_t i = 0; i < 16; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_Nx16(output_ptrs.data(), matrix.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_Nx16);

static void BM_transpose_bit_32xN(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint32_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<std::uint32_t, 32 * N> matrix alignas(32);
    std::array<std::uint32_t, 32 * N> output alignas(32);
    std::array<const std::uint8_t*, 32> input_ptrs;
    for (std::size_t i = 0; i < 32; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_32xN(output.data(), input_ptrs.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_32xN);

static void BM_transpose_bit_Nx32(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint32_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<std::uint32_t, 32 * N> matrix alignas(32);
    std::array<std::uint32_t, 32 * N> output alignas(32);
    std::array<std::uint8_t*, 32> output_ptrs;
    for (std::size_t i = 0; i < 32; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_Nx32(output_ptrs.data(), matrix.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_Nx32);

static void BM_transpose_bit_64xN(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint64_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<std::uint64_t, 64 * N> matrix alignas(32);
    std::array<std::uint64_t, 64 * N> output alignas(32);
    std::array<const std::uint8_t*, 64> input_ptrs;
    for (std::size_t i = 0; i < 64; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_64xN(output.data(), input_ptrs.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_64xN);

static void BM_transpose_bit_Nx64(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint64_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<std::uint64_t, 64 * N> matrix alignas(32);
    std::array<std::uint64_t, 64 * N> output alignas(32);
    std::array<std::uint8_t*, 64> output_ptrs;
    for (std::size_t i = 0; i < 64; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate(matrix.begin(), matrix.end(), [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_Nx64(output_ptrs.data(), matrix.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_Nx64);

static void BM_transpose_bit_128xN(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint8_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<__m128i, 128 * N> matrix alignas(32);
    std::array<__m128i, 128 * N> output alignas(32);
    std::array<const std::uint8_t*, 128> input_ptrs;
    for (std::size_t i = 0; i < 128; ++i) {
        input_ptrs[i] = reinterpret_cast<const std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate_n(reinterpret_cast<std::uint8_t*>(matrix.data()), 16 * 128 * 1024,
                    [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_128xN(reinterpret_cast<std::uint8_t*>(output.data()), input_ptrs.data(), N);
    }
}
BENCHMARK(BM_transpose_bit_128xN);

static void BM_transpose_bit_Nx128(benchmark::State& state) {
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<std::uint8_t> dist(0);
    constexpr std::size_t N = 1024;
    std::array<__m128i, 128 * N> matrix alignas(32);
    std::array<__m128i, 128 * N> output alignas(32);
    std::array<std::uint8_t*, 128> output_ptrs;
    for (std::size_t i = 0; i < 128; ++i) {
        output_ptrs[i] = reinterpret_cast<std::uint8_t*>(matrix.data() + i * N);
    }
    std::generate_n(reinterpret_cast<std::uint8_t*>(matrix.data()), 16 * 128 * 1024,
                    [&] { return dist(mt); });

    for (auto _ : state) {
        transpose_bit_Nx128(output_ptrs.data(),
                            reinterpret_cast<const std::uint8_t*>(matrix.data()), N);
    }
}
BENCHMARK(BM_transpose_bit_Nx128);
