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

#ifndef BITTRANSPOSE_BIT_TRANSPOSE_H
#define BITTRANSPOSE_BIT_TRANSPOSE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Functions to transpose square bit matrices.
 *
 * - Functions with the suffix `_direct` take and return a matrix by value.
 * - Functions with the suffix `_inplace` write the transposed matrix back into
 *   the same buffer.  They might use additional memory on the stack.
 * - Functions with the suffix `_aligned` require the pointer to be aligned on
 *   32-byte boundaries.
 */

uint64_t transpose_bit_8x8_direct(uint64_t input);
void transpose_bit_8x8_inplace(void* input);
void transpose_bit_8x8_packed_x4_inplace(void* input);
void transpose_bit_8x8_packed_x4_inplace_aligned(void* input);
void transpose_bit_16x16_inplace(void* input);
void transpose_bit_16x16_inplace_aligned(void* input);
void transpose_bit_32x32_inplace(void* input);
void transpose_bit_32x32_inplace_aligned(void* input);
void transpose_bit_64x64_inplace(void* input);
void transpose_bit_64x64_inplace_aligned(void* input);
void transpose_bit_128x128_inplace(void* input);
void transpose_bit_128x128_inplace_aligned(void* input);

#ifdef __cplusplus
}
#endif

#endif /* BITTRANSPOSE_BIT_TRANSPOSE_H */
