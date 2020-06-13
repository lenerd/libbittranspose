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
#include <stdint.h>
#include <string.h>

uint64_t transpose_bit_8x8_direct(uint64_t x) {
  const uint64_t mask_2x2 = 0x5500550055005500;
  const size_t shift_2x2 = 7;
  const uint64_t mask_4x4 = 0x3333000033330000;
  const size_t shift_4x4 = 14;
  const uint64_t mask_8x8 = 0x0f0f0f0f00000000;
  const size_t shift_8x8 = 28;
  uint64_t tmp;

  tmp = (x ^ (x << shift_2x2)) & mask_2x2;
  x ^= tmp;
  tmp >>= shift_2x2;
  x ^= tmp;

  tmp = (x ^ (x << shift_4x4)) & mask_4x4;
  x ^= tmp;
  tmp >>= shift_4x4;
  x ^= tmp;

  tmp = (x ^ (x << shift_8x8)) & mask_8x8;
  x ^= tmp;
  tmp >>= shift_8x8;
  x ^= tmp;
  return x;
}

void transpose_bit_8x8_inplace(void* x) {
  uint64_t matrix;
  memcpy(&matrix, x, sizeof(matrix));
  matrix = transpose_bit_8x8_direct(matrix);
  memcpy(x, &matrix, sizeof(matrix));
}
