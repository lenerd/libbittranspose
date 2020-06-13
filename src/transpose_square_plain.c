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

void transpose_bit_8x8_packed_x4_inplace(void* input) {
    uint64_t matrices[4];
    memcpy(matrices, input, 4 * sizeof(uint64_t));
    for (size_t i = 0; i < 4; ++i) {
        matrices[i] = transpose_bit_8x8_direct(matrices[i]);
    }
    memcpy(input, matrices, 4 * sizeof(uint64_t));
}

void transpose_bit_8x8_packed_x4_inplace_aligned(void* input) {
    transpose_bit_8x8_packed_x4_inplace(input);
}

void transpose_bit_16x16_inplace(void* input) {
    uint16_t tmp[16];

    /* aliasing with byte-sized pointers is allowed */
    uint8_t* input_bytes = (uint8_t*)input;
    uint8_t* tmp_bytes = (uint8_t*)tmp;

    /* load the four submatrices to the correct positions and swap the
     * antidiagonal ones */
    for (size_t i = 0; i < 8; ++i) {
        tmp_bytes[i] = input_bytes[2 * i];
        tmp_bytes[8 + i] = input_bytes[16 + 2 * i];
        tmp_bytes[16 + i] = input_bytes[2 * i + 1];
        tmp_bytes[24 + i] = input_bytes[16 + 2 * i + 1];
    }

    /* transpose the four submatrices */
    transpose_bit_8x8_packed_x4_inplace(tmp);

    /* write them back */
    for (size_t i = 0; i < 8; ++i) {
        input_bytes[2 * i] = tmp_bytes[i];
        input_bytes[2 * i + 1] = tmp_bytes[8 + i];
        input_bytes[16 + 2 * i] = tmp_bytes[16 + i];
        input_bytes[16 + 2 * i + 1] = tmp_bytes[24 + i];
    }
}

void transpose_bit_16x16_inplace_aligned(void* input) {
    transpose_bit_16x16_inplace(input);
}

void transpose_bit_32x32_inplace(void* input) {
    uint32_t tmp[32];

    /* aliasing with byte-sized pointers is allowed */
    uint8_t* input_bytes = (uint8_t*)input;
    uint8_t* tmp_bytes = (uint8_t*)tmp;

    /* load the four submatrices to the correct positions and swap the
     * antidiagonal ones */
    for (size_t i = 0; i < 16; ++i) {
        uint16_t buffer[4];
        memcpy(buffer, input_bytes + 4 * i, 2 * sizeof(uint16_t));
        memcpy(buffer + 2, input_bytes + 64 + 4 * i, 2 * sizeof(uint16_t));
        memcpy(tmp_bytes + 2 * i, buffer, sizeof(uint16_t));
        memcpy(tmp_bytes + 32 + 2 * i, buffer + 2, sizeof(uint16_t));
        memcpy(tmp_bytes + 64 + 2 * i, buffer + 1, sizeof(uint16_t));
        memcpy(tmp_bytes + 96 + 2 * i, buffer + 3, sizeof(uint16_t));
    }

    /* transpose the four submatrices */
    for (size_t i = 0; i < 4; ++i) {
        transpose_bit_16x16_inplace(tmp + i * 8);
    }

    /* write them back */
    for (size_t i = 0; i < 16; ++i) {
        uint16_t buffer[4];
        memcpy(buffer, tmp_bytes + 2 * i, sizeof(uint16_t));
        memcpy(buffer + 1, tmp_bytes + 32 + 2 * i, sizeof(uint16_t));
        memcpy(buffer + 2, tmp_bytes + 64 + 2 * i, sizeof(uint16_t));
        memcpy(buffer + 3, tmp_bytes + 96 + 2 * i, sizeof(uint16_t));
        memcpy(input_bytes + 4 * i, buffer, 2 * sizeof(uint16_t));
        memcpy(input_bytes + 64 + 4 * i, buffer + 2, 2 * sizeof(uint16_t));
    }
}

void transpose_bit_32x32_inplace_aligned(void* input) {
    return transpose_bit_32x32_inplace(input);
}

void transpose_bit_64x64_inplace(void* input) {
    uint64_t tmp[64];

    /* aliasing with byte-sized pointers is allowed */
    uint8_t* input_bytes = (uint8_t*)input;
    uint8_t* tmp_bytes = (uint8_t*)tmp;

    /* load the four submatrices to the correct positions and swap the
     * antidiagonal ones */
    for (size_t i = 0; i < 32; ++i) {
        uint32_t buffer[4];
        memcpy(buffer, input_bytes + 8 * i, 2 * sizeof(uint32_t));
        memcpy(buffer + 2, input_bytes + 256 + 8 * i, 2 * sizeof(uint32_t));
        memcpy(tmp_bytes + 4 * i, buffer, sizeof(uint32_t));
        memcpy(tmp_bytes + 128 + 4 * i, buffer + 2, sizeof(uint32_t));
        memcpy(tmp_bytes + 256 + 4 * i, buffer + 1, sizeof(uint32_t));
        memcpy(tmp_bytes + 384 + 4 * i, buffer + 3, sizeof(uint32_t));
    }

    /* transpose the four submatrices */
    for (size_t i = 0; i < 4; ++i) {
        transpose_bit_32x32_inplace(tmp + i * 16);
    }

    /* write them back */
    for (size_t i = 0; i < 32; ++i) {
        uint32_t buffer[4];
        memcpy(buffer, tmp_bytes + 4 * i, sizeof(uint32_t));
        memcpy(buffer + 1, tmp_bytes + 128 + 4 * i, sizeof(uint32_t));
        memcpy(buffer + 2, tmp_bytes + 256 + 4 * i, sizeof(uint32_t));
        memcpy(buffer + 3, tmp_bytes + 384 + 4 * i, sizeof(uint32_t));
        memcpy(input_bytes + 8 * i, buffer, 2 * sizeof(uint32_t));
        memcpy(input_bytes + 256 + 8 * i, buffer + 2, 2 * sizeof(uint32_t));
    }
}

void transpose_bit_64x64_inplace_aligned(void* input) {
    transpose_bit_64x64_inplace(input);
}

void transpose_bit_128x128_inplace(void* input) {
    uint64_t tmp[256];

    /* aliasing with byte-sized pointers is allowed */
    uint8_t* input_bytes = (uint8_t*)input;
    uint8_t* tmp_bytes = (uint8_t*)tmp;

    /* load the four submatrices to the correct positions and swap the
     * antidiagonal ones */
    for (size_t i = 0; i < 64; ++i) {
        uint64_t buffer[4];
        memcpy(buffer, input_bytes + 16 * i, 2 * sizeof(uint64_t));
        memcpy(buffer + 2, input_bytes + 1024 + 16 * i, 2 * sizeof(uint64_t));
        memcpy(tmp_bytes + 8 * i, buffer, sizeof(uint64_t));
        memcpy(tmp_bytes + 512 + 8 * i, buffer + 2, sizeof(uint64_t));
        memcpy(tmp_bytes + 1024 + 8 * i, buffer + 1, sizeof(uint64_t));
        memcpy(tmp_bytes + 1536 + 8 * i, buffer + 3, sizeof(uint64_t));
    }

    /* transpose the four submatrices */
    for (size_t i = 0; i < 4; ++i) {
        transpose_bit_64x64_inplace(tmp + i * 64);
    }

    /* write them back */
    for (size_t i = 0; i < 64; ++i) {
        uint64_t buffer[4];
        memcpy(buffer, tmp_bytes + 8 * i, sizeof(uint64_t));
        memcpy(buffer + 1, tmp_bytes + 512 + 8 * i, sizeof(uint64_t));
        memcpy(buffer + 2, tmp_bytes + 1024 + 8 * i, sizeof(uint64_t));
        memcpy(buffer + 3, tmp_bytes + 1536 + 8 * i, sizeof(uint64_t));
        memcpy(input_bytes + 16 * i, buffer, 2 * sizeof(uint64_t));
        memcpy(input_bytes + 1024 + 16 * i, buffer + 2, 2 * sizeof(uint64_t));
    }
}

void transpose_bit_128x128_inplace_aligned(void* input) {
    transpose_bit_128x128_inplace(input);
}
