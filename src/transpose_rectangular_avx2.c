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
#include "bit_transpose_extra_avx2.h"

#include <immintrin.h>
#include <string.h>

void transpose_bit_8xN(uint8_t* dst, const uint8_t* const* src, size_t N) {
    /* shuffle / permutation masks used below */
    const __m256i shuffle_mask = _mm256_set_epi64x(0x0f0b07030e0a0602, 0x0d0905010c080400,
                                                   0x0f0b07030e0a0602, 0x0d0905010c080400);
    const __m256i permute_mask = _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

    /* transposition of a 8xN matrix: */
    /* - partition the matrix into blocks of size 8x8 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        /* transpose 4 8x8 blocks */
        __m256i vec;
        /* load 4 words from each row */
        memcpy((uint8_t*)(&vec), src[0] + 4 * superblock_i, 4);
        memcpy((uint8_t*)(&vec) + 4, src[1] + 4 * superblock_i, 4);
        memcpy((uint8_t*)(&vec) + 8, src[2] + 4 * superblock_i, 4);
        memcpy((uint8_t*)(&vec) + 12, src[3] + 4 * superblock_i, 4);
        memcpy((uint8_t*)(&vec) + 16, src[4] + 4 * superblock_i, 4);
        memcpy((uint8_t*)(&vec) + 20, src[5] + 4 * superblock_i, 4);
        memcpy((uint8_t*)(&vec) + 24, src[6] + 4 * superblock_i, 4);
        memcpy((uint8_t*)(&vec) + 28, src[7] + 4 * superblock_i, 4);
        /* shuffle the bytes on both 128 bit lanes */
        /* [FEDC BA98 7654 3210] -> [FB73 EA62 D951 C840] */
        vec = _mm256_shuffle_epi8(vec, shuffle_mask);
        /* permute 32 bit words across lanes */
        /* [7777 6666 5555 4444 | 3333 2222 1111 0000] -> [7777 3333 6666 2222 | 5555 1111 4444
         * 0000] */
        vec = _mm256_permutevar8x32_epi32(vec, permute_mask);
        /* now each 64 bit word contains a 8x8 submatrix which we transpose separately */
        vec = transpose_bit_8x8_packed_x4_direct(vec);
        memcpy(&dst[32 * superblock_i], &vec, 32);
    }
    /* process the remaining 8x8 blocks */
    for (size_t block_i = 4 * num_super_blocks; block_i < N; ++block_i) {
        uint64_t block = 0;
        for (size_t i = 0; i < 8; ++i) {
            block |= (uint64_t)(src[i][block_i]) << (i * 8);
        }
        block = transpose_bit_8x8_direct(block);
        memcpy(&dst[8 * block_i], &block, 8);
    }
}

void transpose_bit_Nx8(uint8_t** dst, const uint8_t* src, size_t N) {
    /* shuffle / permutation masks used below */
    const __m256i shuffle_mask = _mm256_set_epi64x(0x0f0b07030e0a0602, 0x0d0905010c080400,
                                                   0x0f0b07030e0a0602, 0x0d0905010c080400);
    const __m256i permute_mask = _mm256_set_epi32(0x07, 0x05, 0x03, 0x01, 0x06, 0x04, 0x02, 0x00);

    /* transposition of a 8xN matrix: */
    /* - partition the matrix into blocks of size 8x8 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        /* transpose 4 8x8 blocks */
        /* load 4 blocks */
        __m256i vec;
        memcpy(&vec, &src[32 * superblock_i], 32);
        /* each 64 bit word contains a 8x8 submatrix which we transpose separately */
        vec = transpose_bit_8x8_packed_x4_direct(vec);
        /* permute 32 bit words across lanes */
        /* [7777 6666 5555 4444 | 3333 2222 1111 0000] -> [7777 5555 3333 1111 | 6666 4444 2222
         * 0000] */
        vec = _mm256_permutevar8x32_epi32(vec, permute_mask);
        /* shuffle the bytes on both 128 bit lanes */
        /* [FEDC BA98 7654 3210] -> [FB73 EA62 D951 C840] */
        vec = _mm256_shuffle_epi8(vec, shuffle_mask);
        /* store 4 words into each row */
        memcpy(dst[0] + 4 * superblock_i, (uint8_t*)(&vec), 4);
        memcpy(dst[1] + 4 * superblock_i, (uint8_t*)(&vec) + 4, 4);
        memcpy(dst[2] + 4 * superblock_i, (uint8_t*)(&vec) + 8, 4);
        memcpy(dst[3] + 4 * superblock_i, (uint8_t*)(&vec) + 12, 4);
        memcpy(dst[4] + 4 * superblock_i, (uint8_t*)(&vec) + 16, 4);
        memcpy(dst[5] + 4 * superblock_i, (uint8_t*)(&vec) + 20, 4);
        memcpy(dst[6] + 4 * superblock_i, (uint8_t*)(&vec) + 24, 4);
        memcpy(dst[7] + 4 * superblock_i, (uint8_t*)(&vec) + 28, 4);
    }
    /* process the remaining 8x8 blocks */
    for (size_t block_i = 4 * num_super_blocks; block_i < N; ++block_i) {
        uint64_t block;
        memcpy(&block, &src[8 * block_i], 8);
        block = transpose_bit_8x8_direct(block);
        for (size_t i = 0; i < 8; ++i) {
            dst[i][block_i] = (block >> (i * 8)) & 0xff;
        }
    }
}

void transpose_bit_16xN_onebyone(uint16_t* dst, const uint8_t* const* src, size_t N) {
    /* one-by-one implementation */

    /* transposition of a 16xN matrix: */
    /* - partition the matrix into blocks of size 16x16 */
    for (size_t block_i = 0; block_i < N; ++block_i) {
        // transpose 16x16 blocks
        __m256i vec;
        /* load a word from each row */
        for (size_t i = 0; i < 16; ++i) {
            memcpy((uint8_t*)(&vec) + 2 * i, src[i] + 2 * block_i, 2);
        }
        vec = transpose_bit_16x16_direct(vec);
        memcpy(&dst[block_i * 16], &vec, 32);
    }
}

void transpose_bit_16xN(uint16_t* dst, const uint8_t* const* src, size_t N) {
    /* batched implementation */

    /* shuffle / permutation masks used below */
    const __m256i shuffle_mask = _mm256_set_epi64x(0x0f0e07060d0c0504, 0x0b0a030209080100,
                                                   0x0f0e07060d0c0504, 0x0b0a030209080100);
    const __m256i permute_mask = _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

    /* transposition of a 16xN matrix: */
    /* - partition the matrix into blocks of size 16x16 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    size_t num_rest = N % 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        /* transpose 4 16x16 blocks */
        __m256i vec[4];
        __m256i tmp[4];
        /* load 4 words from each row */
        for (size_t j = 0; j < 4; ++j) {
            memcpy((uint8_t*)(&vec[j]), src[4 * j + 0] + 8 * superblock_i, 8);
            memcpy((uint8_t*)(&vec[j]) + 8, src[4 * j + 1] + 8 * superblock_i, 8);
            memcpy((uint8_t*)(&vec[j]) + 16, src[4 * j + 2] + 8 * superblock_i, 8);
            memcpy((uint8_t*)(&vec[j]) + 24, src[4 * j + 3] + 8 * superblock_i, 8);
        }
        // interleave the 16 bit words of the two 64 bit block on each 128 bit lane
        // [FEDC BA98 7654 3210] -> [FE76 DC54 BA32 9810]
        for (size_t j = 0; j < 4; ++j) {
            vec[j] = _mm256_shuffle_epi8(vec[j], shuffle_mask);
        }
        /* permute 32 bit words across lanes */
        /* [7777 6666 5555 4444 | 3333 2222 1111 0000] -> [7777 3333 6666 2222 | 5555 1111 4444
         * 0000] */
        for (size_t j = 0; j < 4; ++j) {
            vec[j] = _mm256_permutevar8x32_epi32(vec[j], permute_mask);
        }

        /* now each 64 bit word the quarter of a 16x16 submatrix */
        /* - vec[0] contains the first 4 rows, vec[1] the second 4, and so on */
        tmp[0] = _mm256_permute2x128_si256(vec[0], vec[1], 0b00100000);
        tmp[1] = _mm256_permute2x128_si256(vec[0], vec[1], 0b00110001);
        tmp[2] = _mm256_permute2x128_si256(vec[2], vec[3], 0b00100000);
        tmp[3] = _mm256_permute2x128_si256(vec[2], vec[3], 0b00110001);
        /* swap the middle 64 bit words across lanes */
        for (size_t j = 0; j < 4; ++j) {
            vec[j] = _mm256_permute4x64_epi64(tmp[j], 0b11011000);
        }
        tmp[0] = _mm256_permute2x128_si256(vec[0], vec[2], 0b00100000);
        tmp[1] = _mm256_permute2x128_si256(vec[0], vec[2], 0b00110001);
        tmp[2] = _mm256_permute2x128_si256(vec[1], vec[3], 0b00100000);
        tmp[3] = _mm256_permute2x128_si256(vec[1], vec[3], 0b00110001);

        for (size_t j = 0; j < 4; ++j) {
            tmp[j] = transpose_bit_16x16_direct(tmp[j]);
        }
        memcpy(&dst[4 * superblock_i * 16], &tmp, 4 * 32);
    }
    /* process the remaining 16x16 blocks */
    uint16_t* rest_dst = dst + 4 * num_super_blocks * 16;
    const uint8_t* rest_src[16];
    for (size_t i = 0; i < 16; ++i) {
        rest_src[i] = src[i] + 2 * 4 * num_super_blocks;
    }
    transpose_bit_16xN_onebyone(rest_dst, rest_src, num_rest);
}

void transpose_bit_Nx16_onebyone(uint8_t** dst, const uint16_t* src, size_t N) {
    /* one-by-one implementation */

    /* transposition of a 16xN matrix: */
    /* - partition the matrix into blocks of size 16x16 */
    for (size_t block_i = 0; block_i < N; ++block_i) {
        // transpose 16x16 blocks
        __m256i vec;
        memcpy(&vec, &src[block_i * 16], 32);
        vec = transpose_bit_16x16_direct(vec);
        /* store a word into each row */
        for (size_t i = 0; i < 16; ++i) {
            memcpy(dst[i] + 2 * block_i, (uint8_t*)(&vec) + 2 * i, 2);
        }
    }
}

void transpose_bit_Nx16(uint8_t** dst, const uint16_t* src, size_t N) {
    /* batched implementation */

    /* shuffle / permutation masks used below */
    const __m256i shuffle_mask = _mm256_set_epi64x(0x0f0e0b0a07060302, 0x0d0c090805040100,
                                                   0x0f0e0b0a07060302, 0x0d0c090805040100);
    const __m256i permute_mask = _mm256_set_epi32(0x07, 0x05, 0x03, 0x01, 0x06, 0x04, 0x02, 0x00);

    /* transposition of a 16xN matrix: */
    /* - partition the matrix into blocks of size 16x16 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    size_t num_rest = N % 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        /* transpose 4 16x16 blocks */
        __m256i vec[4];
        __m256i tmp[4];
        /* load 4 16x16 matrices and transpose them */
        memcpy(tmp, &src[4 * superblock_i * 16], 4 * 32);
        for (size_t j = 0; j < 4; ++j) {
            tmp[j] = transpose_bit_16x16_direct(tmp[j]);
        }
        vec[0] = _mm256_permute2x128_si256(tmp[0], tmp[1], 0b00100000);
        vec[2] = _mm256_permute2x128_si256(tmp[0], tmp[1], 0b00110001);
        vec[1] = _mm256_permute2x128_si256(tmp[2], tmp[3], 0b00100000);
        vec[3] = _mm256_permute2x128_si256(tmp[2], tmp[3], 0b00110001);

        /* swap the middle 64 bit words across lanes */
        for (size_t j = 0; j < 4; ++j) {
            tmp[j] = _mm256_permute4x64_epi64(vec[j], 0b11011000);
        }

        vec[0] = _mm256_permute2x128_si256(tmp[0], tmp[1], 0b00100000);
        vec[1] = _mm256_permute2x128_si256(tmp[0], tmp[1], 0b00110001);
        vec[2] = _mm256_permute2x128_si256(tmp[2], tmp[3], 0b00100000);
        vec[3] = _mm256_permute2x128_si256(tmp[2], tmp[3], 0b00110001);

        /* permute 32 bit words across lanes */
        /* [7777 6666 5555 4444 | 3333 2222 1111 0000] -> [7777 5555 3333 1111 | 6666 4444 2222
         * 0000] */
        for (size_t j = 0; j < 4; ++j) {
            vec[j] = _mm256_permutevar8x32_epi32(vec[j], permute_mask);
        }
        /* deinterleave the 16 bit words of the two 64 bit block on each 128 bit lane */
        /* [FEDC BA98 7654 3210] -> [FEBA 7632 DC98 5410] */
        for (size_t j = 0; j < 4; ++j) {
            vec[j] = _mm256_shuffle_epi8(vec[j], shuffle_mask);
        }
        /* store 4 words into each row */
        for (size_t j = 0; j < 4; ++j) {
            memcpy(dst[4 * j + 0] + 8 * superblock_i, (uint8_t*)(&vec[j]), 8);
            memcpy(dst[4 * j + 1] + 8 * superblock_i, (uint8_t*)(&vec[j]) + 8, 8);
            memcpy(dst[4 * j + 2] + 8 * superblock_i, (uint8_t*)(&vec[j]) + 16, 8);
            memcpy(dst[4 * j + 3] + 8 * superblock_i, (uint8_t*)(&vec[j]) + 24, 8);
        }
    }
    /* process the remaining 16x16 blocks */
    const uint16_t* rest_src = src + 4 * num_super_blocks * 16;
    uint8_t* rest_dst[16];
    for (size_t i = 0; i < 16; ++i) {
        rest_dst[i] = dst[i] + 2 * 4 * num_super_blocks;
    }
    transpose_bit_Nx16_onebyone(rest_dst, rest_src, num_rest);
}

void transpose_bit_32xN_onebyone(uint32_t* dst, const uint8_t* const* src, size_t N) {
    /* one-by-one implementation */

    /* transposition of a 32xN matrix: */
    /* - partition the matrix into blocks of size 32x32 */
    for (size_t block_i = 0; block_i < N; ++block_i) {
        /* transpose 32x32 blocks */
        /* copy a word from each row */
        for (size_t j = 0; j < 32; ++j) {
            memcpy((uint8_t*)(dst) + 128 * block_i + 4 * j, src[j] + 4 * block_i, 4);
        }
        transpose_bit_32x32_inplace(&dst[32 * block_i]);
    }
}

void transpose_bit_32xN(uint32_t* dst, const uint8_t* const* src, size_t N) {
    /* batched implementation */

    /* permutation mask used below */
    const __m256i permute_mask = _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

    /* transposition of a 32xN matrix: */
    /* - partition the matrix into blocks of size 32x32 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    size_t num_rest = N % 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        /* transpose 4 16x16 blocks */
        __m256i vec[16];
        __m256i tmp[16];
        /* load 4 words from each row */
        for (size_t j = 0; j < 16; ++j) {
            memcpy((uint8_t*)(&vec[j]), src[2 * j] + 16 * superblock_i, 16);
            memcpy((uint8_t*)(&vec[j]) + 16, src[2 * j + 1] + 16 * superblock_i, 16);
        }
        /* interleave the 32 bit words across the two 128 bit lanes */
        /* [7777 6666 5555 4444 | 3333 2222 1111 0000] -> [7777 3333 6666 2222 | 5555 1111 4444
         * 0000] */
        for (size_t j = 0; j < 16; ++j) {
            vec[j] = _mm256_permutevar8x32_epi32(vec[j], permute_mask);
        }
        /* now each 64 bits belong together, so a 256 bit register looks like this */
        /* [BBBB AAAA BBBB AAAA | BBBB AAAA BBBB AAAA] where, AAAA/BBBB belong to the same row resp.
         */

        for (size_t j = 0; j < 8; ++j) {
            tmp[2 * j + 0] = _mm256_permute2x128_si256(vec[2 * j], vec[2 * j + 1], 0b00100000);
            tmp[2 * j + 1] = _mm256_permute2x128_si256(vec[2 * j], vec[2 * j + 1], 0b00110001);
        }
        /* now [DDDD CCCC DDDD CCCC | BBBB AAAA BBBB AAAA] where, A/B/C/D belong to the same row
         * resp. */
        /* interleave the 32 bit words across the two 128 bit lanes */
        /* [7777-6666 5555-4444 | 3333-2222 1111-0000] -> [7777-6666 3333-2222 | 5555-4444
         * 1111-0000] */
        for (size_t j = 0; j < 16; ++j) {
            tmp[j] = _mm256_permute4x64_epi64(tmp[j], 0b11011000);
        }
        /* now [DDDD CCCC BBBB AAAA | DDDD CCCC BBBB AAAA] where, A/B/C/D belong to the same row
         * resp. */
        /* and the whole state looks like this: */
        /* [DCBA DCBA] [DCBA DCBA] [HGFE HGFE] [HGFE HGFE] [LKJI LKJI] [LKJI LKJI] [PONM PONM] [PONM
         * PONM] */
        /* [TSRQ TSRQ] [TSRQ TSRQ] [XWVU XWVU] [XWVU XWVU] [baZY baZY] [baZY baZY] [fedc fedc] [fedc
         * fedc] */
        for (size_t j = 0; j < 4; ++j) {
            vec[j] = _mm256_permute2x128_si256(tmp[4 * j], tmp[4 * j + 2], 0b00100000);
            vec[4 + j] = _mm256_permute2x128_si256(tmp[4 * j], tmp[4 * j + 2], 0b00110001);
            vec[8 + j] = _mm256_permute2x128_si256(tmp[4 * j + 1], tmp[4 * j + 3], 0b00100000);
            vec[12 + j] = _mm256_permute2x128_si256(tmp[4 * j + 1], tmp[4 * j + 3], 0b00110001);
        }
        /* now each four 256 bit words contain one 32x32 matrix which we can transpose: */
        /* [HGFE DCBA] [PONM LKJI] [XWVU TSRQ] [fedc baZY] [HGFE DCBA] [PONM LKJI] [XWVU TSRQ] [fedc
         * baZY] */
        /* [HGFE DCBA] [PONM LKJI] [XWVU TSRQ] [fedc baZY] [HGFE DCBA] [PONM LKJI] [XWVU TSRQ] [fedc
         * baZY] */
        for (size_t j = 0; j < 4; ++j) {
            transpose_bit_32x32_inplace_aligned(&vec[4 * j]);
        }
        memcpy(&dst[128 * superblock_i], &vec, 16 * 32);
    }
    /* process the remaining 32x32 blocks */
    uint32_t* rest_dst = dst + 4 * num_super_blocks * 32;
    const uint8_t* rest_src[32];
    for (size_t i = 0; i < 32; ++i) {
        rest_src[i] = src[i] + 4 * 4 * num_super_blocks;
    }
    transpose_bit_32xN_onebyone(rest_dst, rest_src, num_rest);
}

void transpose_bit_Nx32_onebyone(uint8_t** dst, const uint32_t* src, size_t N) {
    /* one-by-one implementation */

    /* transposition of a 32xN matrix: */
    /* - partition the matrix into blocks of size 32x32 */
    for (size_t block_i = 0; block_i < N; ++block_i) {
        /* transpose 32x32 blocks */
        __m256i vec[4];
        memcpy(vec, src + block_i * 32, 128);
        transpose_bit_32x32_inplace_aligned(vec);
        /* store a word into each row */
        for (size_t j = 0; j < 32; ++j) {
            memcpy(dst[j] + 4 * block_i, (uint8_t*)(&vec) + 4 * j, 4);
        }
    }
}

void transpose_bit_Nx32(uint8_t** dst, const uint32_t* src, size_t N) {
    /* batched implementation */

    /* transposition of a 16xN matrix: */
    /* - partition the matrix into blocks of size 16x16 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    size_t num_rest = N % 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        /* transpose 4 32x32 blocks */
        __m256i vec[16];
        __m256i tmp[16];
        /* load four 32x32 matrices */
        memcpy(vec, &src[128 * superblock_i], 16 * 32);
        /* and transpose them in place */
        for (size_t j = 0; j < 4; ++j) {
            transpose_bit_32x32_inplace_aligned(&vec[4 * j]);
        }
        /* now do the reverse permutations of those in the 32xN case: */

        /* first, transform the state from this */
        /* [HGFE DCBA] [PONM LKJI] [XWVU TSRQ] [fedc baZY] [HGFE DCBA] [PONM LKJI] [XWVU TSRQ] [fedc
         * baZY] */
        /* [HGFE DCBA] [PONM LKJI] [XWVU TSRQ] [fedc baZY] [HGFE DCBA] [PONM LKJI] [XWVU TSRQ] [fedc
         * baZY] */
        /* to this */
        /* [DCBA DCBA] [DCBA DCBA] [HGFE HGFE] [HGFE HGFE] [LKJI LKJI] [LKJI LKJI] [PONM PONM] [PONM
         * PONM] */
        /* [TSRQ TSRQ] [TSRQ TSRQ] [XWVU XWVU] [XWVU XWVU] [baZY baZY] [baZY baZY] [fedc fedc] [fedc
         * fedc] */
        for (size_t j = 0; j < 4; ++j) {
            tmp[4 * j + 0] = _mm256_permute2x128_si256(vec[j], vec[4 + j], 0b00100000);
            tmp[4 * j + 1] = _mm256_permute2x128_si256(vec[8 + j], vec[12 + j], 0b00100000);
            tmp[4 * j + 2] = _mm256_permute2x128_si256(vec[j], vec[4 + j], 0b00110001);
            tmp[4 * j + 3] = _mm256_permute2x128_si256(vec[8 + j], vec[12 + j], 0b00110001);
        }
        /* interleave the 32 bit words across the two 128 bit lanes */
        /* [7777-6666 5555-4444 | 3333-2222 1111-0000] -> [7777-6666 3333-2222 | 5555-4444
         * 1111-0000] */
        for (size_t j = 0; j < 16; ++j) {
            tmp[j] = _mm256_permute4x64_epi64(tmp[j], 0b11011000);
        }
        /* now we have [DCDC BABA] [DCDC BABA] [HGHG FEFE] [HGHG FEFE] ... */
        for (size_t j = 0; j < 8; ++j) {
            vec[2 * j + 0] = _mm256_permute2x128_si256(tmp[2 * j], tmp[2 * j + 1], 0b00100000);
            vec[2 * j + 1] = _mm256_permute2x128_si256(tmp[2 * j], tmp[2 * j + 1], 0b00110001);
        }
        /* now we have [BABA BABA] [DCDC DCDC] [FEFE FEFE] [HGHG HGHG] ... */
        /* deinterleave the 32 bit words across the two 128 bit lanes */
        /* [7777 6666 5555 4444 | 3333 2222 1111 0000] -> [7777 5555 3333 1111 | 6666 4444 2222
         * 0000] */
        const __m256i permute_mask =
            _mm256_set_epi32(0x07, 0x05, 0x03, 0x01, 0x06, 0x04, 0x02, 0x00);
        for (size_t j = 0; j < 16; ++j) {
            vec[j] = _mm256_permutevar8x32_epi32(vec[j], permute_mask);
        }

        /* store 4 words into each row */
        for (size_t j = 0; j < 16; ++j) {
            memcpy(dst[2 * j] + 16 * superblock_i, (uint8_t*)(&vec[j]), 16);
            memcpy(dst[2 * j + 1] + 16 * superblock_i, (uint8_t*)(&vec[j]) + 16, 16);
        }
    }
    /* process the remaining 16x16 blocks */
    const uint32_t* rest_src = src + 4 * num_super_blocks * 32;
    uint8_t* rest_dst[32];
    for (size_t i = 0; i < 32; ++i) {
        rest_dst[i] = dst[i] + 4 * 4 * num_super_blocks;
    }
    transpose_bit_Nx32_onebyone(rest_dst, rest_src, num_rest);
}

void transpose_bit_64xN_onebyone(uint64_t* dst, const uint8_t* const* src, size_t N) {
    /* one-by-one implementation */

    /* transposition of a 64xN matrix: */
    /* - partition the matrix into blocks of size 64x64 */
    for (size_t block_i = 0; block_i < N; ++block_i) {
        /* transpose 64x64 blocks */
        /* copy a word from each row */
        for (size_t j = 0; j < 64; ++j) {
            memcpy((uint8_t*)(dst) + 512 * block_i + 8 * j, src[j] + 8 * block_i, 8);
        }
        transpose_bit_64x64_inplace(&dst[64 * block_i]);
    }
}

void transpose_bit_Nx64_onebyone(uint8_t** dst, const uint64_t* src, size_t N) {
    /* one-by-one implementation */

    /* transposition of a 64xN matrix: */
    /* - partition the matrix into blocks of size 64x64 */
    for (size_t block_i = 0; block_i < N; ++block_i) {
        /* transpose 64x64 blocks */
        __m256i vec[16];
        memcpy(vec, src + block_i * 64, 512);
        transpose_bit_64x64_inplace_aligned(vec);
        /* store a word into each row */
        for (size_t i = 0; i < 64; ++i) {
            memcpy(dst[i] + 8 * block_i, (uint8_t*)(&vec) + 8 * i, 8);
        }
    }
}

void transpose_bit_64xN(uint64_t* dst, const uint8_t* const* src, size_t N) {
    /* batched implementation */

    /* transposition of a 64xN matrix: */
    /* - partition the matrix into blocks of size 64x64 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    size_t num_rest = N % 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        /* transpose 4 64x64 blocks */
        __m256i vec[64];
        __m256i tmp[64];
        /* load 4 words from each row */
        /* -> [AAAA] [BBBB] [CCCC] [DDDD] ... */
        for (size_t j = 0; j < 64; ++j) {
            memcpy((uint8_t*)(&vec[j]), src[j] + 32 * superblock_i, 32);
        }
        /* interleave 128 bit words */
        /* -> [BBAA] [BBAA] [DDCC] [DDCC] ... */
        for (size_t j = 0; j < 32; ++j) {
            tmp[2 * j + 0] = _mm256_permute2x128_si256(vec[2 * j], vec[2 * j + 1], 0b00100000);
            tmp[2 * j + 1] = _mm256_permute2x128_si256(vec[2 * j], vec[2 * j + 1], 0b00110001);
        }
        /* swap the middle 64 bit words */
        /* -> [BABA] [BABA] [DCDC] [DCDC] ... */
        for (size_t j = 0; j < 64; ++j) {
            tmp[j] = _mm256_permute4x64_epi64(tmp[j], 0b11011000);
        }
        /* interleave 128 bit words again and put the results to the correct position */
        /* -> [DCBA] [HGFE] ... [DCBA] [HGFE] ... [DCBA] [HGFE] ... [DCBA] [HGFE] ... */
        for (size_t j = 0; j < 16; ++j) {
            vec[0 + j] = _mm256_permute2x128_si256(tmp[4 * j + 0], tmp[4 * j + 2], 0b00100000);
            vec[16 + j] = _mm256_permute2x128_si256(tmp[4 * j + 0], tmp[4 * j + 2], 0b00110001);
            vec[32 + j] = _mm256_permute2x128_si256(tmp[4 * j + 1], tmp[4 * j + 3], 0b00100000);
            vec[48 + j] = _mm256_permute2x128_si256(tmp[4 * j + 1], tmp[4 * j + 3], 0b00110001);
        }
        /* transpose each 64x64 block */
        for (size_t j = 0; j < 4; ++j) {
            transpose_bit_64x64_inplace_aligned(&vec[16 * j]);
        }
        memcpy(&dst[256 * superblock_i], vec, 64 * 32);
    }
    /* process the remaining 64x64 blocks */
    uint64_t* rest_dst = dst + 4 * num_super_blocks * 64;
    const uint8_t* rest_src[64];
    for (size_t i = 0; i < 64; ++i) {
        rest_src[i] = src[i] + 4 * 8 * num_super_blocks;
    }
    transpose_bit_64xN_onebyone(rest_dst, rest_src, num_rest);
}

void transpose_bit_Nx64(uint8_t** dst, const uint64_t* src, size_t N) {
    /* batched implementation */

    /* transposition of a 64xN matrix: */
    /* - partition the matrix into blocks of size 64x64 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    size_t num_rest = N % 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        /* transpose 4 64x64 blocks */
        __m256i vec[64];
        __m256i tmp[64];
        /* load four 64x64 matrices */
        memcpy(vec, &src[256 * superblock_i], 64 * 32);
        // transpose each 64x64 block
        for (size_t j = 0; j < 4; ++j) {
            transpose_bit_64x64_inplace_aligned(&vec[16 * j]);
        }
        // deinterleave 128 bit words again and put the results to the correct position
        // -> [BABA] [BABA] [DCDC] [DCDC] ...
        for (size_t j = 0; j < 16; ++j) {
            tmp[4 * j + 0] = _mm256_permute2x128_si256(vec[j], vec[16 + j], 0b00100000);
            tmp[4 * j + 2] = _mm256_permute2x128_si256(vec[j], vec[16 + j], 0b00110001);
            tmp[4 * j + 1] = _mm256_permute2x128_si256(vec[32 + j], vec[48 + j], 0b00100000);
            tmp[4 * j + 3] = _mm256_permute2x128_si256(vec[32 + j], vec[48 + j], 0b00110001);
        }
        // swap the middle 64 bit words
        // -> [BBAA] [BBAA] [DDCC] [DDCC] ...
        for (size_t j = 0; j < 64; ++j) {
            tmp[j] = _mm256_permute4x64_epi64(tmp[j], 0b11011000);
        }
        // deinterleave 128 bit words
        // -> [AAAA] [BBBB] [CCCC] [DDDD] ...
        for (size_t j = 0; j < 32; ++j) {
            vec[2 * j + 0] = _mm256_permute2x128_si256(tmp[2 * j], tmp[2 * j + 1], 0b00100000);
            vec[2 * j + 1] = _mm256_permute2x128_si256(tmp[2 * j], tmp[2 * j + 1], 0b00110001);
        }
        // store 4 words into each row
        for (size_t j = 0; j < 64; ++j) {
            memcpy(dst[j] + 32 * superblock_i, (uint8_t*)(&vec[j]), 32);
        }
    }
    // process the remaining 64x64 blocks
    const uint64_t* rest_src = src + 4 * num_super_blocks * 64;
    uint8_t* rest_dst[64];
    for (size_t i = 0; i < 64; ++i) {
        rest_dst[i] = dst[i] + 4 * 8 * num_super_blocks;
    }
    transpose_bit_Nx64_onebyone(rest_dst, rest_src, num_rest);
}

void transpose_bit_128xN_onebyone(uint8_t* dst, const uint8_t* const* src, size_t N) {
    /* one-by-one implementation */

    /* transposition of a 128xN matrix: */
    /* - partition the matrix into blocks of size 128x128 */
    for (size_t block_i = 0; block_i < N; ++block_i) {
        /* transpose 128x128 blocks */
        /* copy a word from each row */
        for (size_t j = 0; j < 128; ++j) {
            memcpy(dst + 2048 * block_i + 16 * j, src[j] + 16 * block_i, 16);
        }
        transpose_bit_128x128_inplace(&dst[2048 * block_i]);
    }
}

void transpose_bit_128xN(uint8_t* dst, const uint8_t* const* src, size_t N) {
    /* batched implementation */

    /* transposition of a 128xN matrix: */
    /* - partition the matrix into blocks of size 128x128 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    size_t num_rest = N % 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        __m256i vec[256];
        /* transpose 128x128 blocks */
        /* copy four words from each row */
        for (size_t j = 0; j < 128; ++j) {
            memcpy((uint8_t*)(&vec[2 * j]), src[j] + 64 * superblock_i, 64);
        }
        const __m128i* vec_as_128i_p = (const __m128i*)(vec);
        for (size_t j = 0; j < 128; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                memcpy(&dst[8192 * superblock_i + 2048 * k + 16 * j], &vec_as_128i_p[4 * j + k],
                       16);
            }
        }
        for (size_t j = 0; j < 4; ++j) {
            transpose_bit_128x128_inplace(&dst[8192 * superblock_i + 2048 * j]);
        }
    }
    /* process the remaining 128x128 blocks */
    uint8_t* rest_dst = dst + 4 * num_super_blocks * 128 * 16;
    const uint8_t* rest_src[128];
    for (size_t i = 0; i < 128; ++i) {
        rest_src[i] = src[i] + 4 * 16 * num_super_blocks;
    }
    transpose_bit_128xN_onebyone(rest_dst, rest_src, num_rest);
}

void transpose_bit_Nx128_onebyone(uint8_t** dst, const uint8_t* src, size_t N) {
    /* one-by-one implementation */

    /* transposition of a 128xN matrix: */
    /* - partition the matrix into blocks of size 128x128 */
    for (size_t block_i = 0; block_i < N; ++block_i) {
        /* transpose 128x128 blocks */
        __m256i vec[64];
        memcpy(vec, src + block_i * 2048, 2048);
        transpose_bit_128x128_inplace_aligned(vec);
        /* store a word into each row */
        for (size_t i = 0; i < 128; ++i) {
            memcpy(dst[i] + 16 * block_i, (uint8_t*)(&vec) + 16 * i, 16);
        }
    }
}

void transpose_bit_Nx128(uint8_t** dst, const uint8_t* src, size_t N) {
    /* batched implementation */

    /* transposition of a Nx128 matrix: */
    /* - partition the matrix into blocks of size 128x128 */
    /* - process as many as possible in super blocks of 4 */
    size_t num_super_blocks = N / 4;
    size_t num_rest = N % 4;
    for (size_t superblock_i = 0; superblock_i < num_super_blocks; ++superblock_i) {
        __m256i vec[256];
        __m256i tmp[256];
        /* load four matrices */
        memcpy(vec, &src[32 * superblock_i * 256], 256 * 32);
        /* transpose 128x128 blocks */
        for (size_t j = 0; j < 4; ++j) {
            transpose_bit_128x128_inplace_aligned(&vec[64 * j]);
        }
        const __m128i* vec_as_128i_p = (const __m128i*)(vec);
        __m128i* tmp_as_128i_p = (__m128i*)(tmp);
        for (size_t j = 0; j < 128; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                tmp_as_128i_p[4 * j + k] = vec_as_128i_p[128 * k + j];
            }
        }
        /* copy four words to each row */
        for (size_t j = 0; j < 128; ++j) {
            memcpy(dst[j] + 64 * superblock_i, (uint8_t*)(&tmp[2 * j]), 64);
        }
    }
    /* process the remaining 128x128 blocks */
    const uint8_t* rest_src = src + 4 * num_super_blocks * 128 * 16;
    uint8_t* rest_dst[128];
    for (size_t i = 0; i < 128; ++i) {
        rest_dst[i] = dst[i] + 4 * 16 * num_super_blocks;
    }
    transpose_bit_Nx128_onebyone(rest_dst, rest_src, num_rest);
}
