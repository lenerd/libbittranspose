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

#include <immintrin.h>
#include <stddef.h>
#include <string.h>

__m256i transpose_bit_8x8_packed_x4_direct(__m256i matrix) {
    const __m256i mask_2x2 = _mm256_set1_epi64x(0x5500550055005500);
    const __m256i mask_4x4 = _mm256_set1_epi64x(0x3333000033330000);
    const __m256i mask_8x8 = _mm256_set1_epi64x(0x0f0f0f0f00000000);
    const size_t shift_2x2 = 7;
    const size_t shift_4x4 = 14;
    const size_t shift_8x8 = 28;
    __m256i tmp;

    tmp = (matrix ^ _mm256_slli_epi64(matrix, shift_2x2)) & mask_2x2;
    matrix ^= tmp;
    tmp = _mm256_srli_epi64(tmp, shift_2x2);
    matrix ^= tmp;

    tmp = (matrix ^ _mm256_slli_epi64(matrix, shift_4x4)) & mask_4x4;
    matrix ^= tmp;
    tmp = _mm256_srli_epi64(tmp, shift_4x4);
    matrix ^= tmp;

    tmp = (matrix ^ _mm256_slli_epi64(matrix, shift_8x8)) & mask_8x8;
    matrix ^= tmp;
    tmp = _mm256_srli_epi64(tmp, shift_8x8);
    matrix ^= tmp;
    return matrix;
}

void transpose_bit_8x8_packed_x4_inplace_aligned(void* input) {
    __m256i* matrix = input;
    matrix[0] = transpose_bit_8x8_packed_x4_direct(matrix[0]);
}

void transpose_bit_8x8_packed_x4_inplace(void* input) {
    __m256i matrix = _mm256_loadu_si256(input);
    matrix = transpose_bit_8x8_packed_x4_direct(matrix);
    _mm256_storeu_si256(input, matrix);
}

__m256i transpose_bit_16x16_direct(__m256i matrix) {
    /* in-line byte shuffle */
    const __m256i shuffle_mask_1 = _mm256_set_epi64x(0x0f0d0b0907050301, 0x0e0c0a0806040200,
                                                     0x0f0d0b0907050301, 0x0e0c0a0806040200);
    matrix = _mm256_shuffle_epi8(matrix, shuffle_mask_1);

    /* transpose packed 8x8 */
    matrix = transpose_bit_8x8_packed_x4_direct(matrix);

    /* cross-lane byte shuffle */
    matrix = _mm256_permute4x64_epi64(matrix, 0b11011000);

    /* in-line byte shuffle */
    const __m256i shuffle_mask_2 = _mm256_set_epi64x(0x0f070e060d050c04, 0x0b030a0209010800,
                                                     0x0f070e060d050c04, 0x0b030a0209010800);
    matrix = _mm256_shuffle_epi8(matrix, shuffle_mask_2);

    return matrix;
}

void transpose_bit_16x16_inplace(void* input) {
    __m256i matrix = _mm256_loadu_si256(input);
    matrix = transpose_bit_16x16_direct(matrix);
    _mm256_storeu_si256(input, matrix);
}

void transpose_bit_16x16_inplace_aligned(void* input) {
    __m256i* matrix = input;
    matrix[0] = transpose_bit_16x16_direct(matrix[0]);
}

void transpose_bit_32x32_inplace_aligned(void* input) {
    const __m256i shuffle_mask = _mm256_set_epi64x(0x0f0e0b0a07060302, 0x0d0c090805040100,
                                                   0x0f0e0b0a07060302, 0x0d0c090805040100);
    const __m256i inv_shuffle_mask = _mm256_set_epi64x(0x0f0e07060d0c0504, 0x0b0a030209080100,
                                                       0x0f0e07060d0c0504, 0x0b0a030209080100);

    __m256i* src = input;

    /* get 16x16 submatrices */

    __m256i rows_0 = _mm256_shuffle_epi8(src[0], shuffle_mask);
    __m256i rows_1 = _mm256_shuffle_epi8(src[1], shuffle_mask);
    __m256i rows_2 = _mm256_shuffle_epi8(src[2], shuffle_mask);
    __m256i rows_3 = _mm256_shuffle_epi8(src[3], shuffle_mask);

    __m256i submatrix_a = _mm256_unpacklo_epi64(rows_0, rows_1);
    __m256i submatrix_b = _mm256_unpackhi_epi64(rows_0, rows_1);
    __m256i submatrix_c = _mm256_unpacklo_epi64(rows_2, rows_3);
    __m256i submatrix_d = _mm256_unpackhi_epi64(rows_2, rows_3);

    submatrix_a = _mm256_permute4x64_epi64(submatrix_a, 0b11011000);
    submatrix_b = _mm256_permute4x64_epi64(submatrix_b, 0b11011000);
    submatrix_c = _mm256_permute4x64_epi64(submatrix_c, 0b11011000);
    submatrix_d = _mm256_permute4x64_epi64(submatrix_d, 0b11011000);

    /* swap antidiagonal ones */
    __m256i tmp = submatrix_b;
    submatrix_b = submatrix_c;
    submatrix_c = tmp;

    /* transpose 16x16s */
    submatrix_a = transpose_bit_16x16_direct(submatrix_a);
    submatrix_b = transpose_bit_16x16_direct(submatrix_b);
    submatrix_c = transpose_bit_16x16_direct(submatrix_c);
    submatrix_d = transpose_bit_16x16_direct(submatrix_d);

    /* write submatrices back */
    submatrix_a = _mm256_permute4x64_epi64(submatrix_a, 0b11011000);
    submatrix_b = _mm256_permute4x64_epi64(submatrix_b, 0b11011000);
    submatrix_c = _mm256_permute4x64_epi64(submatrix_c, 0b11011000);
    submatrix_d = _mm256_permute4x64_epi64(submatrix_d, 0b11011000);
    rows_0 = _mm256_unpacklo_epi64(submatrix_a, submatrix_b);
    rows_1 = _mm256_unpackhi_epi64(submatrix_a, submatrix_b);
    rows_2 = _mm256_unpacklo_epi64(submatrix_c, submatrix_d);
    rows_3 = _mm256_unpackhi_epi64(submatrix_c, submatrix_d);

    src[0] = _mm256_shuffle_epi8(rows_0, inv_shuffle_mask);
    src[1] = _mm256_shuffle_epi8(rows_1, inv_shuffle_mask);
    src[2] = _mm256_shuffle_epi8(rows_2, inv_shuffle_mask);
    src[3] = _mm256_shuffle_epi8(rows_3, inv_shuffle_mask);
}

void transpose_bit_32x32_inplace(void* input) {
    __m256i matrix[4];
    memcpy(matrix, input, 32 * 4);
    transpose_bit_32x32_inplace_aligned(matrix);
    memcpy(input, matrix, 32 * 4);
}

void transpose_bit_64x64_inplace_aligned(void* input) {
    const __m256i permute_mask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    const __m256i inv_permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    __m256i* src = input;
    __m256i tmp[16];
    __m128i* src_as_128i = (__m128i*)(src);
    __m128i* tmp_as_128i = (__m128i*)(tmp);

    /* get 32x32 submatrices */
    /* and swap antidiagonal ones */
    src[0] = _mm256_permutevar8x32_epi32(src[0], permute_mask);
    src[1] = _mm256_permutevar8x32_epi32(src[1], permute_mask);
    tmp[0] = _mm256_set_m128i(src_as_128i[2], src_as_128i[0]);
    tmp[8] = _mm256_set_m128i(src_as_128i[3], src_as_128i[1]);

    src[2] = _mm256_permutevar8x32_epi32(src[2], permute_mask);
    src[3] = _mm256_permutevar8x32_epi32(src[3], permute_mask);
    tmp[1] = _mm256_set_m128i(src_as_128i[6], src_as_128i[4]);
    tmp[9] = _mm256_set_m128i(src_as_128i[7], src_as_128i[5]);

    src[4] = _mm256_permutevar8x32_epi32(src[4], permute_mask);
    src[5] = _mm256_permutevar8x32_epi32(src[5], permute_mask);
    tmp[2] = _mm256_set_m128i(src_as_128i[10], src_as_128i[8]);
    tmp[10] = _mm256_set_m128i(src_as_128i[11], src_as_128i[9]);

    src[6] = _mm256_permutevar8x32_epi32(src[6], permute_mask);
    src[7] = _mm256_permutevar8x32_epi32(src[7], permute_mask);
    tmp[3] = _mm256_set_m128i(src_as_128i[14], src_as_128i[12]);
    tmp[11] = _mm256_set_m128i(src_as_128i[15], src_as_128i[13]);

    src[8] = _mm256_permutevar8x32_epi32(src[8], permute_mask);
    src[9] = _mm256_permutevar8x32_epi32(src[9], permute_mask);
    tmp[4] = _mm256_set_m128i(src_as_128i[18], src_as_128i[16]);
    tmp[12] = _mm256_set_m128i(src_as_128i[19], src_as_128i[17]);

    src[10] = _mm256_permutevar8x32_epi32(src[10], permute_mask);
    src[11] = _mm256_permutevar8x32_epi32(src[11], permute_mask);
    tmp[5] = _mm256_set_m128i(src_as_128i[22], src_as_128i[20]);
    tmp[13] = _mm256_set_m128i(src_as_128i[23], src_as_128i[21]);

    src[12] = _mm256_permutevar8x32_epi32(src[12], permute_mask);
    src[13] = _mm256_permutevar8x32_epi32(src[13], permute_mask);
    tmp[6] = _mm256_set_m128i(src_as_128i[26], src_as_128i[24]);
    tmp[14] = _mm256_set_m128i(src_as_128i[27], src_as_128i[25]);

    src[14] = _mm256_permutevar8x32_epi32(src[14], permute_mask);
    src[15] = _mm256_permutevar8x32_epi32(src[15], permute_mask);
    tmp[7] = _mm256_set_m128i(src_as_128i[30], src_as_128i[28]);
    tmp[15] = _mm256_set_m128i(src_as_128i[31], src_as_128i[29]);

    /* transpose 32x32s */
    transpose_bit_32x32_inplace_aligned(tmp);
    transpose_bit_32x32_inplace_aligned(tmp + 4);
    transpose_bit_32x32_inplace_aligned(tmp + 8);
    transpose_bit_32x32_inplace_aligned(tmp + 12);

    /* write submatrices back */
    src[0] = _mm256_set_m128i(tmp_as_128i[8], tmp_as_128i[0]);
    src[0] = _mm256_permutevar8x32_epi32(src[0], inv_permute_mask);
    src[1] = _mm256_set_m128i(tmp_as_128i[9], tmp_as_128i[1]);
    src[1] = _mm256_permutevar8x32_epi32(src[1], inv_permute_mask);
    src[2] = _mm256_set_m128i(tmp_as_128i[10], tmp_as_128i[2]);
    src[2] = _mm256_permutevar8x32_epi32(src[2], inv_permute_mask);
    src[3] = _mm256_set_m128i(tmp_as_128i[11], tmp_as_128i[3]);
    src[3] = _mm256_permutevar8x32_epi32(src[3], inv_permute_mask);

    src[4] = _mm256_set_m128i(tmp_as_128i[12], tmp_as_128i[4]);
    src[4] = _mm256_permutevar8x32_epi32(src[4], inv_permute_mask);
    src[5] = _mm256_set_m128i(tmp_as_128i[13], tmp_as_128i[5]);
    src[5] = _mm256_permutevar8x32_epi32(src[5], inv_permute_mask);
    src[6] = _mm256_set_m128i(tmp_as_128i[14], tmp_as_128i[6]);
    src[6] = _mm256_permutevar8x32_epi32(src[6], inv_permute_mask);
    src[7] = _mm256_set_m128i(tmp_as_128i[15], tmp_as_128i[7]);
    src[7] = _mm256_permutevar8x32_epi32(src[7], inv_permute_mask);

    src[8] = _mm256_set_m128i(tmp_as_128i[24], tmp_as_128i[16]);
    src[8] = _mm256_permutevar8x32_epi32(src[8], inv_permute_mask);
    src[9] = _mm256_set_m128i(tmp_as_128i[25], tmp_as_128i[17]);
    src[9] = _mm256_permutevar8x32_epi32(src[9], inv_permute_mask);
    src[10] = _mm256_set_m128i(tmp_as_128i[26], tmp_as_128i[18]);
    src[10] = _mm256_permutevar8x32_epi32(src[10], inv_permute_mask);
    src[11] = _mm256_set_m128i(tmp_as_128i[27], tmp_as_128i[19]);
    src[11] = _mm256_permutevar8x32_epi32(src[11], inv_permute_mask);

    src[12] = _mm256_set_m128i(tmp_as_128i[28], tmp_as_128i[20]);
    src[12] = _mm256_permutevar8x32_epi32(src[12], inv_permute_mask);
    src[13] = _mm256_set_m128i(tmp_as_128i[29], tmp_as_128i[21]);
    src[13] = _mm256_permutevar8x32_epi32(src[13], inv_permute_mask);
    src[14] = _mm256_set_m128i(tmp_as_128i[30], tmp_as_128i[22]);
    src[14] = _mm256_permutevar8x32_epi32(src[14], inv_permute_mask);
    src[15] = _mm256_set_m128i(tmp_as_128i[31], tmp_as_128i[23]);
    src[15] = _mm256_permutevar8x32_epi32(src[15], inv_permute_mask);
}

void transpose_bit_64x64_inplace(void* input) {
    __m256i src[16];
    memcpy(src, input, 16 * 32);
    transpose_bit_64x64_inplace_aligned(src);
    memcpy(input, src, 16 * 32);
}

void transpose_bit_128x128_inplace_aligned(void* input) {
    const uint8_t permute_mask_64 = 0b11011000;    /* 0b11'01'10'00 */
    const uint8_t permute_mask_128_0 = 0b00100000; /* 0b0010'0000 */
    const uint8_t permute_mask_128_1 = 0b00110001; /* 0b0011'0001 */
    __m256i* src = input;
    __m256i tmp[5];

    /* separate the four submatrices and swap antidiagonal ones */

    for (size_t row_block_i = 0; row_block_i < 16; ++row_block_i) {
        /* swap the middle 64 bit words of each 256 bit word */
        /* [0 1 2 3] -> [0 2 1 3] */
        tmp[0] = _mm256_permute4x64_epi64(src[2 * row_block_i + 0], 0b11011000);
        tmp[1] = _mm256_permute4x64_epi64(src[2 * row_block_i + 1], 0b11011000);
        tmp[2] = _mm256_permute4x64_epi64(src[2 * row_block_i + 32], 0b11011000);
        tmp[3] = _mm256_permute4x64_epi64(src[2 * row_block_i + 33], 0b11011000);
        /* exchange the middle 128 bit lanes of two 256 bit words */
        /* [0 2 1 3], [4 6 5 7] -> [0 2 4 6], [1 3 5 7] */
        /* and swap submatrices */
        src[2 * row_block_i + 0] = _mm256_permute2x128_si256(tmp[0], tmp[1], 0b00100000);
        src[2 * row_block_i + 32] = _mm256_permute2x128_si256(tmp[0], tmp[1], 0b00110001);
        src[2 * row_block_i + 1] = _mm256_permute2x128_si256(tmp[2], tmp[3], 0b00100000);
        src[2 * row_block_i + 33] = _mm256_permute2x128_si256(tmp[2], tmp[3], 0b00110001);
    }

    /* the neighbouring submatrices are now interleaved as 256 bit words each */

    /* clang-format off */

    /* move the elements to the right position in 5er cycles */
    /* upper half: */
    tmp[0] = src[1]; tmp[1] = src[16]; tmp[2] = src[8]; tmp[3] = src[4]; tmp[4] = src[2];
    src[16] = tmp[0]; src[8] = tmp[1]; src[4] = tmp[2]; src[2] = tmp[3]; src[1] = tmp[4];
    tmp[0] = src[3]; tmp[1] = src[17]; tmp[2] = src[24]; tmp[3] = src[12]; tmp[4] = src[6];
    src[17] = tmp[0]; src[24] = tmp[1]; src[12] = tmp[2]; src[6] = tmp[3]; src[3] = tmp[4];
    tmp[0] = src[5]; tmp[1] = src[18]; tmp[2] = src[9]; tmp[3] = src[20]; tmp[4] = src[10];
    src[18] = tmp[0]; src[9] = tmp[1]; src[20] = tmp[2]; src[10] = tmp[3]; src[5] = tmp[4];
    tmp[0] = src[7]; tmp[1] = src[19]; tmp[2] = src[25]; tmp[3] = src[28]; tmp[4] = src[14];
    src[19] = tmp[0]; src[25] = tmp[1]; src[28] = tmp[2]; src[14] = tmp[3]; src[7] = tmp[4];
    tmp[0] = src[11]; tmp[1] = src[21]; tmp[2] = src[26]; tmp[3] = src[13]; tmp[4] = src[22];
    src[21] = tmp[0]; src[26] = tmp[1]; src[13] = tmp[2]; src[22] = tmp[3]; src[11] = tmp[4];
    tmp[0] = src[15]; tmp[1] = src[23]; tmp[2] = src[27]; tmp[3] = src[29]; tmp[4] = src[30];
    src[23] = tmp[0]; src[27] = tmp[1]; src[29] = tmp[2]; src[30] = tmp[3]; src[15] = tmp[4];
    /* lower half */
    tmp[0] = src[33]; tmp[1] = src[48]; tmp[2] = src[40]; tmp[3] = src[36]; tmp[4] = src[34];
    src[48] = tmp[0]; src[40] = tmp[1]; src[36] = tmp[2]; src[34] = tmp[3]; src[33] = tmp[4];
    tmp[0] = src[35]; tmp[1] = src[49]; tmp[2] = src[56]; tmp[3] = src[44]; tmp[4] = src[38];
    src[49] = tmp[0]; src[56] = tmp[1]; src[44] = tmp[2]; src[38] = tmp[3]; src[35] = tmp[4];
    tmp[0] = src[37]; tmp[1] = src[50]; tmp[2] = src[41]; tmp[3] = src[52]; tmp[4] = src[42];
    src[50] = tmp[0]; src[41] = tmp[1]; src[52] = tmp[2]; src[42] = tmp[3]; src[37] = tmp[4];
    tmp[0] = src[39]; tmp[1] = src[51]; tmp[2] = src[57]; tmp[3] = src[60]; tmp[4] = src[46];
    src[51] = tmp[0]; src[57] = tmp[1]; src[60] = tmp[2]; src[46] = tmp[3]; src[39] = tmp[4];
    tmp[0] = src[43]; tmp[1] = src[53]; tmp[2] = src[58]; tmp[3] = src[45]; tmp[4] = src[54];
    src[53] = tmp[0]; src[58] = tmp[1]; src[45] = tmp[2]; src[54] = tmp[3]; src[43] = tmp[4];
    tmp[0] = src[47]; tmp[1] = src[55]; tmp[2] = src[59]; tmp[3] = src[61]; tmp[4] = src[62];
    src[55] = tmp[0]; src[59] = tmp[1]; src[61] = tmp[2]; src[62] = tmp[3]; src[47] = tmp[4];

    /* clang-format on */

    /* transpose each submatrix */
    transpose_bit_64x64_inplace_aligned(src);
    transpose_bit_64x64_inplace_aligned(src + 16);
    transpose_bit_64x64_inplace_aligned(src + 32);
    transpose_bit_64x64_inplace_aligned(src + 48);

    /* interleave the neighbouring submatrices again */

    /* clang-format off */

    /* interleave them as 256 bit words each in 5er cycles */
    /* upper half: */
    tmp[0] = src[1]; tmp[1] = src[16]; tmp[2] = src[8]; tmp[3] = src[4]; tmp[4] = src[2];
    src[2] = tmp[0]; src[1] = tmp[1]; src[16] = tmp[2]; src[8] = tmp[3]; src[4] = tmp[4];
    tmp[0] = src[3]; tmp[1] = src[17]; tmp[2] = src[24]; tmp[3] = src[12]; tmp[4] = src[6];
    src[6] = tmp[0]; src[3] = tmp[1]; src[17] = tmp[2]; src[24] = tmp[3]; src[12] = tmp[4];
    tmp[0] = src[5]; tmp[1] = src[18]; tmp[2] = src[9]; tmp[3] = src[20]; tmp[4] = src[10];
    src[10] = tmp[0]; src[5] = tmp[1]; src[18] = tmp[2]; src[9] = tmp[3]; src[20] = tmp[4];
    tmp[0] = src[7]; tmp[1] = src[19]; tmp[2] = src[25]; tmp[3] = src[28]; tmp[4] = src[14];
    src[14] = tmp[0]; src[7] = tmp[1]; src[19] = tmp[2]; src[25] = tmp[3]; src[28] = tmp[4];
    tmp[0] = src[11]; tmp[1] = src[21]; tmp[2] = src[26]; tmp[3] = src[13]; tmp[4] = src[22];
    src[22] = tmp[0]; src[11] = tmp[1]; src[21] = tmp[2]; src[26] = tmp[3]; src[13] = tmp[4];
    tmp[0] = src[15]; tmp[1] = src[23]; tmp[2] = src[27]; tmp[3] = src[29]; tmp[4] = src[30];
    src[30] = tmp[0]; src[15] = tmp[1]; src[23] = tmp[2]; src[27] = tmp[3]; src[29] = tmp[4];
    /* lower half: */
    tmp[0] = src[33]; tmp[1] = src[48]; tmp[2] = src[40]; tmp[3] = src[36]; tmp[4] = src[34];
    src[34] = tmp[0]; src[33] = tmp[1]; src[48] = tmp[2]; src[40] = tmp[3]; src[36] = tmp[4];
    tmp[0] = src[35]; tmp[1] = src[49]; tmp[2] = src[56]; tmp[3] = src[44]; tmp[4] = src[38];
    src[38] = tmp[0]; src[35] = tmp[1]; src[49] = tmp[2]; src[56] = tmp[3]; src[44] = tmp[4];
    tmp[0] = src[37]; tmp[1] = src[50]; tmp[2] = src[41]; tmp[3] = src[52]; tmp[4] = src[42];
    src[42] = tmp[0]; src[37] = tmp[1]; src[50] = tmp[2]; src[41] = tmp[3]; src[52] = tmp[4];
    tmp[0] = src[39]; tmp[1] = src[51]; tmp[2] = src[57]; tmp[3] = src[60]; tmp[4] = src[46];
    src[46] = tmp[0]; src[39] = tmp[1]; src[51] = tmp[2]; src[57] = tmp[3]; src[60] = tmp[4];
    tmp[0] = src[43]; tmp[1] = src[53]; tmp[2] = src[58]; tmp[3] = src[45]; tmp[4] = src[54];
    src[54] = tmp[0]; src[43] = tmp[1]; src[53] = tmp[2]; src[58] = tmp[3]; src[45] = tmp[4];
    tmp[0] = src[47]; tmp[1] = src[55]; tmp[2] = src[59]; tmp[3] = src[61]; tmp[4] = src[62];
    src[62] = tmp[0]; src[47] = tmp[1]; src[55] = tmp[2]; src[59] = tmp[3]; src[61] = tmp[4];

    /* clang-format on */

    /* the neighbouring submatrices are now interleaved as 256 bit words each */

    for (size_t row_block_i = 0; row_block_i < 16; ++row_block_i) {
        /* exchange the middle 128 bit lanes of two 256 bit words */
        /* [0 2 4 6], [1 3 5 7] -> [0 2 1 3], [4 6 5 7] */
        tmp[0] = _mm256_permute2x128_si256(src[4 * row_block_i + 0], src[4 * row_block_i + 1],
                                           0b00100000);
        tmp[1] = _mm256_permute2x128_si256(src[4 * row_block_i + 0], src[4 * row_block_i + 1],
                                           0b00110001);
        tmp[2] = _mm256_permute2x128_si256(src[4 * row_block_i + 2], src[4 * row_block_i + 3],
                                           0b00100000);
        tmp[3] = _mm256_permute2x128_si256(src[4 * row_block_i + 2], src[4 * row_block_i + 3],
                                           0b00110001);
        /* swap the middle 64 bit words of each 256 bit word */
        /* [0 2 1 3] -> [0 1 2 3] */
        src[4 * row_block_i + 0] = _mm256_permute4x64_epi64(tmp[0], 0b11011000);
        src[4 * row_block_i + 1] = _mm256_permute4x64_epi64(tmp[1], 0b11011000);
        src[4 * row_block_i + 2] = _mm256_permute4x64_epi64(tmp[2], 0b11011000);
        src[4 * row_block_i + 3] = _mm256_permute4x64_epi64(tmp[3], 0b11011000);
    }
}

void transpose_bit_128x128_inplace(void* input) {
    __m256i src[64];
    memcpy(src, input, 64 * 32);
    transpose_bit_128x128_inplace_aligned(src);
    memcpy(input, src, 64 * 32);
}
