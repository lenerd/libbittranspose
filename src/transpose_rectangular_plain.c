/* MIT License
 *
 * Copyright (c) 2020 Lennart Braun
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "bit_transpose.h"

#include <stddef.h>
#include <string.h>

void transpose_bit_8xN(uint8_t* dst, const uint8_t* const* src, size_t N) {
    for (size_t block_i = 0; block_i < N; ++block_i) {
        for (size_t row_j = 0; row_j < 8; ++row_j) {
            dst[8 * block_i + row_j] = src[row_j][block_i];
        }
        transpose_bit_8x8_inplace(dst + 8 * block_i);
    }
}

void transpose_bit_Nx8(uint8_t** dst, const uint8_t* src, size_t N) {
    uint8_t matrix[8];
    for (size_t block_i = 0; block_i < N; ++block_i) {
        memcpy(matrix, src + 8 * block_i, 8);
        transpose_bit_8x8_inplace(matrix);
        for (size_t row_j = 0; row_j < 8; ++row_j) {
            dst[row_j][block_i] = matrix[row_j];
        }
    }
}

void transpose_bit_16xN(uint16_t* dst, const uint8_t* const* src, size_t N) {
    for (size_t block_i = 0; block_i < N; ++block_i) {
        for (size_t row_j = 0; row_j < 16; ++row_j) {
            memcpy(&dst[16 * block_i + row_j], &src[row_j][2 * block_i], 2);
        }
        transpose_bit_16x16_inplace(dst + 16 * block_i);
    }
}

void transpose_bit_Nx16(uint8_t** dst, const uint16_t* src, size_t N) {
    uint16_t matrix[16];
    for (size_t block_i = 0; block_i < N; ++block_i) {
        memcpy(matrix, src + 16 * block_i, 32);
        transpose_bit_16x16_inplace(matrix);
        for (size_t row_j = 0; row_j < 16; ++row_j) {
            memcpy(&dst[row_j][2 * block_i], &matrix[row_j], 2);
        }
    }
}

void transpose_bit_32xN(uint32_t* dst, const uint8_t* const* src, size_t N) {
    for (size_t block_i = 0; block_i < N; ++block_i) {
        for (size_t row_j = 0; row_j < 32; ++row_j) {
            memcpy(&dst[32 * block_i + row_j], &src[row_j][4 * block_i], 4);
        }
        transpose_bit_32x32_inplace(dst + 32 * block_i);
    }
}

void transpose_bit_Nx32(uint8_t** dst, const uint32_t* src, size_t N) {
    uint32_t matrix[32];
    for (size_t block_i = 0; block_i < N; ++block_i) {
        memcpy(matrix, src + 32 * block_i, 128);
        transpose_bit_32x32_inplace(matrix);
        for (size_t row_j = 0; row_j < 32; ++row_j) {
            memcpy(&dst[row_j][4 * block_i], &matrix[row_j], 4);
        }
    }
}

void transpose_bit_64xN(uint64_t* dst, const uint8_t* const* src, size_t N) {
    for (size_t block_i = 0; block_i < N; ++block_i) {
        for (size_t row_j = 0; row_j < 64; ++row_j) {
            memcpy(&dst[64 * block_i + row_j], &src[row_j][8 * block_i], 8);
        }
        transpose_bit_64x64_inplace(dst + 64 * block_i);
    }
}

void transpose_bit_Nx64(uint8_t** dst, const uint64_t* src, size_t N) {
    uint64_t matrix[64];
    for (size_t block_i = 0; block_i < N; ++block_i) {
        memcpy(matrix, src + 64 * block_i, 512);
        transpose_bit_64x64_inplace(matrix);
        for (size_t row_j = 0; row_j < 64; ++row_j) {
            memcpy(&dst[row_j][8 * block_i], &matrix[row_j], 8);
        }
    }
}

void transpose_bit_128xN(uint8_t* dst, const uint8_t* const* src, size_t N) {
    for (size_t block_i = 0; block_i < N; ++block_i) {
        for (size_t row_j = 0; row_j < 128; ++row_j) {
            memcpy(&dst[16 * (128 * block_i + row_j)], &src[row_j][16 * block_i], 16);
        }
        transpose_bit_128x128_inplace(dst + 16 * 128 * block_i);
    }
}

void transpose_bit_Nx128(uint8_t** dst, const uint8_t* src, size_t N) {
    uint8_t matrix[2048];
    for (size_t block_i = 0; block_i < N; ++block_i) {
        memcpy(matrix, src + 16 * 128 * block_i, 2048);
        transpose_bit_128x128_inplace(matrix);
        for (size_t row_j = 0; row_j < 128; ++row_j) {
            memcpy(&dst[row_j][16 * block_i], &matrix[16 * row_j], 16);
        }
    }
}
