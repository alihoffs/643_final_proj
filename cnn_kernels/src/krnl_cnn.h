/****************************************************************
 * Copyright (c) 2020~2022, 18-643 Course Staff, CMU
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.

 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of the FreeBSD Project.
 ****************************************************************/

#pragma once

/*
 * CMU 18643 Fall 2022 Lab Exercise
 *
 * The parameters in this file sets the CNN kernel on FPGA
 *
 * You can edit this file
 */
#include "util643.h"
#include "instance643.h"

typedef uint32_t index_t;

/*
 * In the below, you can set the tile size and the
 * DRAM buffer access macros for the input, weight, and
 * output buffers to change the data layout from the
 * standard natural layout
 */

//////////////////////////////////////////////////////
// Layer X - General Implementation if not using DFX
//////////////////////////////////////////////////////
#define TR_X (4) // output row
#define TC_X (4) // output column
#define TM_X (4) // output depth
#define TN_X (4) // input depth

#define ARRAYi_X(ptr, iR, iC, dR, dC)               \
((ptr)[(iR)*(dC) + iC])

// Must use same macro as input so that the same kernel
// can be used for layer 0 and layer 1
#define ARRAYo_X(ptr, iR, iC, dR, dC)               \
((ptr)[(iR)*(dC) + iC])

#define ARRAYw_X(ptr, iR, iC, dR, dC)               \
((ptr)[(iR)*(dC) + iC])

/////////////////////////////////////////////////////////////
// The below are used by the host side code.
// You should not need to edit below this point for Lab 3
/////////////////////////////////////////////////////////////

//#define NON_RECURSIVE_2
//#define NON_RECURSIVE_4
//#define NON_RECURSIVE_8
//#define NON_RECURSIVE_16
//#define NON_RECURSIVE_32
#define NON_RECURSIVE_64

//#define NON_RECURSIVE_128
//#define NON_RECURSIVE_512


#ifdef __VITIS_CL__
extern "C"
#endif
void krnl_cnn_layerX(const cnndata_t* input, const cnndata_t* weights,
        cnndata_t* output);

// create function headers for strassen recursion
void strassen_64x64(cnndata_t InA[64][64],
                    cnndata_t InB[64][64],
                    cnndata_t OutC[64][64]);
void strassen_32x32(cnndata_t InA[32][32],
                    cnndata_t InB[32][32],
                    cnndata_t OutC[32][32]);
void strassen_16x16(cnndata_t InA[16][16],
                    cnndata_t InB[16][16],
                    cnndata_t OutC[16][16]);
void strassen_8x8(cnndata_t InA[8][8],
                  cnndata_t InB[8][8],
                  cnndata_t OutC[8][8]);
void strassen_4x4(cnndata_t InA[4][4],
                  cnndata_t InB[4][4],
                  cnndata_t OutC[4][4]);
void strassen_2x2(cnndata_t InA[2][2],
                  cnndata_t InB[2][2],
                  cnndata_t OutC[2][2]);

// create function headers for non-recursive mmm's

void mmm_512x512(cnndata_t InA[512][512],
                cnndata_t InB[512][512],
                cnndata_t OutC[512][512]);
void mmm_128x128(cnndata_t InA[128][128],
                cnndata_t InB[128][128],
                cnndata_t OutC[128][128]);
void mmm_64x64(cnndata_t InA[64][64],
                cnndata_t InB[64][64],
                cnndata_t OutC[64][64]);
void mmm_32x32(cnndata_t InA[32][32],
                cnndata_t InB[32][32],
                cnndata_t OutC[32][32]);
void mmm_16x16(cnndata_t InA[16][16],
                cnndata_t InB[16][16],
                cnndata_t OutC[16][16]);
void mmm_8x8(cnndata_t InA[8][8],
                cnndata_t InB[8][8],
                cnndata_t OutC[8][8]);
void mmm_4x4(cnndata_t InA[4][4],
                cnndata_t InB[4][4],
                cnndata_t OutC[4][4]);
void mmm_2x2(cnndata_t InA[2][2],
                cnndata_t InB[2][2],
                cnndata_t OutC[2][2]);

