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

/*
 * CMU 18643 Fall 2022 Lab Exercise
 *
 * You can edit this file
 */

/****************************************************************
 * Blocked convolution layer implementation
 * based on Figure 5:
 *    C. Zhang, et al., "Optimizing FPGA-based Accelerator
 *    Design for Deep Convolutional Neural Networks," FPGA, 2015.
 ****************************************************************/

#include "krnl_cnn.h"
#include <iostream>
#include <string>

// Prevent aliasing
#undef BATCH_SIZE
#undef R_OFM
#undef C_OFM
#undef R_IFM
#undef C_IFM
#undef M_OFM
#undef N_IFM

#include "util643.h"

// create function headers for strassen recursion
void strassen_128x128(cnndata_t InA[128][128],
					  cnndata_t InB[128][128],
					  cnndata_t OutC[128][128]);
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
void mmm_256x256(cnndata_t InA[256][256],
                    cnndata_t InB[256][256],
                    cnndata_t OutC[128][128]);
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

#ifdef __VITIS_CL__
extern "C" {
#endif
void krnl_cnn_layerX(const cnndata_t* inA, const cnndata_t* inB,
        cnndata_t* OutC) {
#pragma HLS TOP name=krnl_cnn_layerX

  index_t i, j, k, j_offset, k_offset;

  cnndata_t inputs[2][128][128];
#pragma HLS ARRAY_RESHAPE dim=1 type=complete variable=inputs
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=bram
  cnndata_t mults[4][128][128];
#pragma HLS ARRAY_RESHAPE dim=1 type=complete variable=mults
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=bram


  // compute C11: M1+M4-M5+M7
  //M1=(A11+A22)(B11+B22)
  //M4=A22(B21-B11)
  //M5=(A11+A12)B22
  //M7=(A12-A22)(B21+B22)
/*
  //X21
  for (j = 0, j_offset = 128; j < 128; j++, j_offset++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
    	elem_b = ARRAYi_X(inB, j_offset, k, 256, 256);
        inputs[3][j][k] = elem_b; // B21-B11
        inputs[7][j][k] = elem_b; // B21+B22
    }
  }

  //X12
  for (j = 0; j < 128; j++) {
    for (k = 0, k_offset = 128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
    	elem_a = ARRAYi_X(inA, j, k_offset, 256, 256);
        inputs[4][j][k] = elem_a; // A11+A12
        inputs[6][j][k] = elem_a; // A12-A22
    }
  }

  //X11
  for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
       	elem_a = ARRAYi_X(inA, j, k, 256, 256);
       	elem_b = ARRAYi_X(inB, j, k, 256, 256);
        inputs[0][j][k] = elem_a; // A11+A22
        inputs[4][j][k] +=  elem_a;// A11+A12

        inputs[1][j][k] = elem_b; // B11+B22
        inputs[3][j][k] -= elem_b; // B21-B11
    }
  }

  //X22
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
       	elem_a = ARRAYi_X(inA, j_offset, k_offset, 256, 256);
		    elem_b = ARRAYi_X(inB, j_offset, k_offset, 256, 256);
        inputs[0][j][k] += elem_a; // A11+A22
        inputs[2][j][k] = elem_a; // A22
        inputs[6][j][k] -= elem_a; // A12-A22

        inputs[1][j][k] += elem_b; // B11+B22
        inputs[5][j][k] = elem_b; // B22
        inputs[7][j][k] += elem_b; // B21+B22
    }
  }
*/


// **** NEW CODE ****

/* COMPUTE M1
 * M1=(A11+A22)(B11+B22) */
  //X11
  for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[0][j][k] = ARRAYi_X(inA, j, k, 256, 256);; // A11+A22
    }
  }
  for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[1][j][k] = ARRAYi_X(inB, j, k, 256, 256);; // B11+B22
    }
  }
    //X22
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[0][j][k] += ARRAYi_X(inA, j_offset, k_offset, 256, 256); // A11+A22
    }
  }
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[1][j][k] += ARRAYi_X(inB, j_offset, k_offset, 256, 256); // B11+B22
    }
  }

 strassen_128x128(inputs[0], inputs[1], mults[0]);

/* COMPUTE M4
 * M4=A22(B21-B11) */
//X21
  for (j = 0, j_offset = 128; j < 128; j++, j_offset++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[1][j][k] = ARRAYi_X(inB, j_offset, k, 256, 256); // B21-B11
    }
  }

  //X11
  for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[1][j][k] -= ARRAYi_X(inB, j, k, 256, 256); // B21-B11
    }
  }

  //X22
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[0][j][k] = ARRAYi_X(inA, j_offset, k_offset, 256, 256); // A22
    }
  }
 strassen_128x128(inputs[0], inputs[1], mults[1]);

/* COMPUTE M5
* M5=(A11+A12)B22 */

  //X12
  for (j = 0; j < 128; j++) {
    for (k = 0, k_offset = 128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[0][j][k] = ARRAYi_X(inA, j, k_offset, 256, 256); // A11+A12
    }
  }

  //X11
  for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[0][j][k] +=  ARRAYi_X(inA, j, k, 256, 256); // A11+A12
    }
  }

  //X22
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[1][j][k] = ARRAYi_X(inB, j_offset, k_offset, 256, 256); // B22
    }
  }
 strassen_128x128(inputs[0], inputs[1], mults[2]);
/* COMPUTE M7
* M7=(A12-A22)(B21+B22) */
  //X21
  // change to +=
  for (j = 0, j_offset = 128; j < 128; j++, j_offset++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[1][j][k] = ARRAYi_X(inB, j_offset, k, 256, 256); // B21+B22
    }
  }

  //X12
  for (j = 0; j < 128; j++) {
    for (k = 0, k_offset = 128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[0][j][k] = ARRAYi_X(inA, j, k_offset, 256, 256); // A12-A22
    }
  }

  //X22
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[0][j][k] -= ARRAYi_X(inA, j_offset, k_offset, 256, 256); // A12-A22  
    }
  }
  // comment this out
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[1][j][k] += ARRAYi_X(inB, j_offset, k_offset, 256, 256); // B21+B22
    }
  }
  

 strassen_128x128(inputs[0], inputs[1], mults[3]);

  // for (i = 0; i < 4; i++) {
  // 	  strassen_128x128(inputs[2*i], inputs[2*i+1], mults[i]);
  // }
  top_C11_out_0:for (j = 0; j < 128; j++) {
    top_C11_out_1:for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
//#pragma HLS PIPELINE
      // M1 + M4 - M5 + M7
      ARRAYi_X(OutC, j, k, 256, 256) = mults[0][j][k] + mults[1][j][k] - mults[2][j][k] + mults[3][j][k];  // C11
    }
  }


  // Calc C12
  //M3=A11(B12-B22)
  //X11
  for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[0][j][k] = ARRAYi_X(inA, j, k, 256, 256); // A11
    }
  }

  //X12
  for (j = 0; j < 128; j++) {
    for (k = 0, k_offset = 128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[1][j][k] = ARRAYi_X(inB, j, k_offset, 256, 256); // B12-B22
    }
  }

  //X22
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[1][j][k] -= ARRAYi_X(inB, j_offset, k_offset, 256, 256); // B12-B22
    }
  }
  // Calc M5
  strassen_128x128(inputs[0], inputs[1], mults[3]);

  top_C12_out_0:for (j = 0; j < 128; j++) {
    top_C12_out_1:for (k = 0, k_offset=128; k < 128; k++,k_offset++) {
#pragma HLS PIPELINE
//#pragma HLS PIPELINE
      // M3 + M5
      ARRAYi_X(OutC, j, k_offset, 256, 256) = mults[2][j][k] + mults[3][j][k];  // C12
    }
  }

  // Calc C21
  // M2=(A21+A22)B11
  // Calc M2

  //X11
 for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[1][j][k] = ARRAYi_X(inB, j, k, 256, 256); // B11
    }
  }
  //X21
  for (j = 0, j_offset = 128; j < 128; j++, j_offset++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
        inputs[0][j][k] = ARRAYi_X(inA, j_offset, k, 256, 256); // A21+A22
    }
  }

  //X22
  for (j = 0, j_offset=128; j < 128; j++, j_offset++) {
    for (k = 0, k_offset=128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[0][j][k] += ARRAYi_X(inA, j_offset, k_offset, 256, 256); // A21+A22
    }
  }

  strassen_128x128(inputs[0], inputs[1], mults[2]);

  top_C21_out_0:for (j = 0, j_offset=128; j < 128; j++,j_offset++) {
    top_C21_out_1:for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
//#pragma HLS PIPELINE
      // M2 + M4
      ARRAYi_X(OutC, j_offset, k, 256, 256) = mults[1][j][k] + mults[2][j][k];  // C21
    }
  }


  // Compute C22
  // M6=(A21-A11)(B11+B12) -- evict M1
  //X21
  for (j = 0, j_offset = 128; j < 128; j++, j_offset++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
      inputs[0][j][k] = ARRAYi_X(inA, j_offset, k, 256, 256); // A21-A11
    }
  }
  //X11
  for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
//      inputs[0][j][k] -= ARRAYi_X(inA, j_offset, k, 256, 256); // A21-A11 THIS WAS AN ISSUE
    	inputs[0][j][k] -= ARRAYi_X(inA, j, k, 256, 256);
    }
  }
  
  // could be potentially removed, since inputs[1] is previously B11
  for (j = 0; j < 128; j++) {
    for (k = 0; k < 128; k++) {
#pragma HLS PIPELINE
      inputs[1][j][k] = ARRAYi_X(inB, j, k, 256, 256); // B11+B12
    }
  }
  
  //X12
  for (j = 0; j < 128; j++) {
    for (k = 0, k_offset = 128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
        inputs[1][j][k] += ARRAYi_X(inB, j, k_offset, 256, 256); // B11+B12
    }
  }


  // Calc M1 and M7
    strassen_128x128(inputs[0], inputs[1], mults[1]);

  top_C22_out_0:for (j = 0, j_offset = 128; j < 128; j++, j_offset++) {
    top_C22_out_1:for (k = 0, k_offset = 128; k < 128; k++, k_offset++) {
#pragma HLS PIPELINE
//#pragma HLS PIPELINE
      // M1 - M2 + M3 + M6
      ARRAYi_X(OutC, j_offset, k_offset, 256, 256) = mults[0][j][k] - mults[2][j][k] + mults[3][j][k] + mults[1][j][k];  // C11
    }
  }

}
#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif

// define strassen/recursive mmm's
void strassen_128x128(cnndata_t InA[128][128],
					  cnndata_t InB[128][128],
					  cnndata_t OutC[128][128]) {
#pragma HLS INLINE recursive


  index_t i, j, k;

  cnndata_t inputs[14][64][64];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=bram
  cnndata_t mults[7][64][64];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=bram


  strassen_128x128_in_0:for (j = 0; j < 64; j++) {
    strassen_128x128_in_1:for (k = 0; k < 64; k++) {
      inputs[0][j][k] = InA[j][k] + InA[j+64][k+64]; // A11+A22
      inputs[1][j][k] = InB[j][k] + InB[j+64][k+64]; // B11+B22
      inputs[2][j][k] = InA[j+64][k] + InA[j+64][k+64]; // A21+A22
      inputs[3][j][k] = InB[j][k]; // B11
      inputs[4][j][k] = InA[j][k]; // A11
      inputs[5][j][k] = InB[j][k+64] - InB[j+64][k+64]; // B12-B22
      inputs[6][j][k] = InA[j+64][k+64]; // A22
      inputs[7][j][k] = InB[j+64][k] - InB[j][k]; // B21-B11
      inputs[8][j][k] = InA[j][k] + InA[j][k+64]; // A11+A12
      inputs[9][j][k] = InB[j+64][k+64]; // B22
      inputs[10][j][k] = InA[j+64][k] - InA[j][k]; // A21-A11
      inputs[11][j][k] = InB[j][k] + InB[j][k+64]; // B11+B12
      inputs[12][j][k] = InA[j][k+64] - InA[j+64][k+64]; // A12-A22
      inputs[13][j][k] = InB[j+64][k] + InB[j+64][k+64]; // B21+B22
  }
}

strassen_128x128_solve:for (i = 0; i < 7; i++) {
    #ifdef NON_RECURSIVE_64
      mmm_64x64(inputs[2*i], inputs[2*i+1], mults[i]);
    #else
      strassen_64x64(inputs[2*i], inputs[2*i+1], mults[i]);
    #endif
}

// create outputs
strassen_128x128_out_0:for (j = 0; j < 64; j++) {
  strassen_128x128_out_1:for (k = 0; k < 64; k++) {
        OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        OutC[j][k+64] = mults[2][j][k] + mults[4][j][k];  // C12
        OutC[j+64][k] = mults[1][j][k] + mults[3][j][k];  // C21
        OutC[j+64][k+64] = mults[0][j][k] - mults[1][j][k] + mults[2][j][k] + mults[5][j][k];  // C22
    }
  }


/*
  index_t i, j, k, j_offset, k_offset;

  cnndata_t inputs[8][64][64];
#pragma HLS ARRAY_RESHAPE dim=1 factor=2 type=block variable=inputs
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=bram
  cnndata_t mults[4][64][64];
#pragma HLS ARRAY_RESHAPE dim=1 factor=2 type=block variable=mults
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=bram
  cnndata_t elem_a, elem_b;


  // compute C11: M1+M4-M5+M7
  //M1=(A11+A22)(B11+B22)
  //M4=A22(B21-B11)
  //M5=(A11+A12)B22
  //M7=(A12-A22)(B21+B22)

  //X21
  for (j = 0, j_offset = 64; j < 64; j++, j_offset++) {
    for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
    	// elem_b = ARRAYi_X(inB, j_offset, k, 128, 128);
    	elem_b = InB[j_offset][k];
        inputs[3][j][k] = elem_b; // B21-B11
        inputs[7][j][k] = elem_b; // B21+B22
    }
  }

  //X12
  for (j = 0; j < 64; j++) {
    for (k = 0, k_offset = 64; k < 64; k++, k_offset++) {
#pragma HLS PIPELINE
    	// elem_a = ARRAYi_X(inA, j, k_offset, 128, 128);
    	elem_a = InA[j][k_offset];
        inputs[4][j][k] = elem_a; // A11+A12
        inputs[6][j][k] = elem_a; // A12-A22
    }
  }

  //X11
  for (j = 0; j < 64; j++) {
    for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
       	// elem_a = ARRAYi_X(inA, j, k, 128, 128);
       	elem_a = InA[j][k];
       	// elem_b = ARRAYi_X(inB, j, k, 128, 128);
       	elem_b = InB[j][k];
        inputs[0][j][k] = elem_a; // A11+A22
        inputs[4][j][k] +=  elem_a;// A11+A12

        inputs[1][j][k] = elem_b; // B11+B22
        inputs[3][j][k] -= elem_b; // B21-B11
    }
  }

  //X22
  for (j = 0, j_offset=64; j < 64; j++, j_offset++) {
    for (k = 0, k_offset=64; k < 64; k++, k_offset++) {
#pragma HLS PIPELINE
       	// elem_a = ARRAYi_X(inA, j_offset, k_offset, 128, 128);
       	elem_a = InA[j_offset][k_offset];
		// elem_b = ARRAYi_X(inB, j_offset, k_offset, 128, 128);
		elem_b = InB[j_offset][k_offset];
        inputs[0][j][k] += elem_a; // A11+A22
        inputs[2][j][k] = elem_a; // A22
        inputs[6][j][k] -= elem_a; // A12-A22

        inputs[1][j][k] += elem_b; // B11+B22
        inputs[5][j][k] = elem_b; // B22
        inputs[7][j][k] += elem_b; // B21+B22
    }
  }


  for (i = 0; i < 4; i++) {
  	  strassen_64x64(inputs[2*i], inputs[2*i+1], mults[i]);
  }
  top_C11_out_0:for (j = 0; j < 64; j++) {
    top_C11_out_1:for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
      // M1 + M4 - M5 + M7
      // ARRAYi_X(OutC, j, k, 128, 128) = mults[0][j][k] + mults[1][j][k] - mults[2][j][k] + mults[3][j][k];  // C11
      OutC[j][k] = mults[0][j][k] + mults[1][j][k] - mults[2][j][k] + mults[3][j][k];  // C11
    }
  }


  // Calc C12
  //M3=A11(B12-B22)
  //X11
  for (j = 0; j < 64; j++) {
    for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
       	// elem_a = ARRAYi_X(inA, j, k, 128, 128);
       	elem_a = InA[j][k];
        inputs[6][j][k] = elem_a; // A11
    }
  }

  //X12
  for (j = 0; j < 64; j++) {
    for (k = 0, k_offset = 64; k < 64; k++, k_offset++) {
#pragma HLS PIPELINE
    	// elem_b = ARRAYi_X(inB, j, k_offset, 128, 128);
    	elem_b = InB[j][k_offset];
        inputs[7][j][k] = elem_b; // B12-B22
    }
  }

  //X22
  for (j = 0, j_offset=64; j < 64; j++, j_offset++) {
    for (k = 0, k_offset=64; k < 64; k++, k_offset++) {
#pragma HLS PIPELINE
		// elem_b = ARRAYi_X(inB, j_offset, k_offset, 128, 128);
		elem_b = InB[j_offset][k_offset];
        inputs[7][j][k] -= elem_b; // B12-B22
    }
  }
  // Calc M5
  strassen_64x64(inputs[6], inputs[7], mults[3]);

  top_C12_out_0:for (j = 0; j < 64; j++) {
    top_C12_out_1:for (k = 0, k_offset=64; k < 64; k++,k_offset++) {
#pragma HLS PIPELINE
      // M3 + M5
      // ARRAYi_X(OutC, j, k_offset, 128, 128) = mults[2][j][k] + mults[3][j][k];  // C12
      OutC[j][k_offset] = mults[2][j][k] + mults[3][j][k];  // C12
    }
  }

  // Calc C21
  // M2=(A21+A22)B11
  // Calc M2

  //X11
 for (j = 0; j < 64; j++) {
    for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
       	// elem_b = ARRAYi_X(inB, j, k, 128, 128);
       	elem_b = InB[j][k];
        inputs[5][j][k] = elem_b; // B11
    }
  }
  //X21
  for (j = 0, j_offset = 64; j < 64; j++, j_offset++) {
    for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
    	// elem_a = ARRAYi_X(inA, j_offset, k, 128, 128);
    	elem_a = InA[j_offset][k];
        inputs[4][j][k] = elem_a; // A21+A22
    }
  }

  //X22
  for (j = 0, j_offset=64; j < 64; j++, j_offset++) {
    for (k = 0, k_offset=64; k < 64; k++, k_offset++) {
#pragma HLS PIPELINE
       	// elem_a = ARRAYi_X(inA, j_offset, k_offset, 128, 128);
       	elem_a = InA[j_offset][k_offset];
        inputs[4][j][k] += elem_a; // A21+A22
    }
  }

  strassen_64x64(inputs[4], inputs[5], mults[2]);

  top_C21_out_0:for (j = 0, j_offset=64; j < 64; j++,j_offset++) {
    top_C21_out_1:for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
      // M2 + M4
      // ARRAYi_X(OutC, j_offset, k, 128, 128) = mults[1][j][k] + mults[2][j][k];  // C21
      OutC[j_offset][k] = mults[1][j][k] + mults[2][j][k];  // C21
    }
  }


  // Compute C22
  // M6=(A21-A11)(B11+B12) -- evict M1
  //X21
  for (j = 0, j_offset = 64; j < 64; j++, j_offset++) {
    for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
      // elem_a = ARRAYi_X(inA, j_offset, k, 128, 128);
      elem_a = InA[j_offset][k];
      inputs[2][j][k] = elem_a; // A21-A11
    }
  }
  //X11
  for (j = 0; j < 64; j++) {
    for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
      // elem_a = ARRAYi_X(inA, j_offset, k, 128, 128);
      elem_a = InA[j_offset][k];
      // elem_b = ARRAYi_X(inB, j_offset, k, 128, 128);
      elem_b = InB[j_offset][k];
      inputs[2][j][k] -= elem_a; // A21-A11
      inputs[3][j][k] = elem_b; // B11+B12
    }
  }
  //X12
  for (j = 0; j < 64; j++) {
    for (k = 0, k_offset = 64; k < 64; k++, k_offset++) {
#pragma HLS PIPELINE
    	// elem_b = ARRAYi_X(inB, j, k_offset, 128, 128);
    	elem_b = InB[j][k_offset];
        inputs[3][j][k] += elem_b; // B11+B12
    }
  }


  // Calc M1 and M7
    strassen_64x64(inputs[2], inputs[3], mults[1]);

  top_C22_out_0:for (j = 0; j < 64; j++) {
    top_C22_out_1:for (k = 0; k < 64; k++) {
#pragma HLS PIPELINE
      // M1 - M2 + M3 + M6
      // ARRAYi_X(OutC, j+64, k+64, 128, 128) = mults[0][j][k] - mults[2][j][k] + mults[3][j][k] + mults[1][j][k];  // C11
      OutC[j+64][k+64] = mults[0][j][k] - mults[2][j][k] + mults[3][j][k] + mults[1][j][k];  // C11
    }
  }
*/

}

void strassen_64x64(cnndata_t InA[64][64],
                    cnndata_t InB[64][64],
                    cnndata_t OutC[64][64]) {

  index_t i, j, k;

  cnndata_t inputs[14][32][32];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=bram
  cnndata_t mults[7][32][32];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=bram


  strassen_64x64_in_0:for (j = 0; j < 32; j++) {
    strassen_64x64_in_1:for (k = 0; k < 32; k++) {
      inputs[0][j][k] = InA[j][k] + InA[j+32][k+32]; // A11+A22
      inputs[1][j][k] = InB[j][k] + InB[j+32][k+32]; // B11+B22
      inputs[2][j][k] = InA[j+32][k] + InA[j+32][k+32]; // A21+A22
      inputs[3][j][k] = InB[j][k]; // B11
      inputs[4][j][k] = InA[j][k]; // A11
      inputs[5][j][k] = InB[j][k+32] - InB[j+32][k+32]; // B12-B22
      inputs[6][j][k] = InA[j+32][k+32]; // A22
      inputs[7][j][k] = InB[j+32][k] - InB[j][k]; // B21-B11
      inputs[8][j][k] = InA[j][k] + InA[j][k+32]; // A11+A12
      inputs[9][j][k] = InB[j+32][k+32]; // B22
      inputs[10][j][k] = InA[j+32][k] - InA[j][k]; // A21-A11
      inputs[11][j][k] = InB[j][k] + InB[j][k+32]; // B11+B12
      inputs[12][j][k] = InA[j][k+32] - InA[j+32][k+32]; // A12-A22
      inputs[13][j][k] = InB[j+32][k] + InB[j+32][k+32]; // B21+B22
  }
}

strassen_64x64_solve:for (i = 0; i < 7; i++) {
    #ifdef NON_RECURSIVE_32
      mmm_32x32(inputs[2*i], inputs[2*i+1], mults[i]);
    #else
      strassen_32x32(inputs[2*i], inputs[2*i+1], mults[i]);
    #endif
}

// create outputs
strassen_64x64_out_0:for (j = 0; j < 32; j++) {
  strassen_64x64_out_1:for (k = 0; k < 32; k++) {
        OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        OutC[j][k+32] = mults[2][j][k] + mults[4][j][k];  // C12
        OutC[j+32][k] = mults[1][j][k] + mults[3][j][k];  // C21
        OutC[j+32][k+32] = mults[0][j][k] - mults[1][j][k] + mults[2][j][k] + mults[5][j][k];  // C22
    }
  }
}


void strassen_32x32(cnndata_t InA[32][32],
                    cnndata_t InB[32][32],
                    cnndata_t OutC[32][32]) {
//#pragma HLS INLINE recursive

  index_t i, j, k;

  cnndata_t inputs[14][16][16];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=lutram
  cnndata_t mults[7][16][16];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=lutram

  strassen_32x32_in_0:for (j = 0; j < 16; j++) {
    strassen_32x32_in_1:for (k = 0; k < 16; k++) {
      inputs[0][j][k] = InA[j][k] + InA[j+16][k+16]; // A11+A22
      inputs[1][j][k] = InB[j][k] + InB[j+16][k+16]; // B11+B22
      inputs[2][j][k] = InA[j+16][k] + InA[j+16][k+16]; // A21+A22
      inputs[3][j][k] = InB[j][k]; // B11
      inputs[4][j][k] = InA[j][k]; // A11
      inputs[5][j][k] = InB[j][k+16] - InB[j+16][k+16]; // B12-B22
      inputs[6][j][k] = InA[j+16][k+16]; // A22
      inputs[7][j][k] = InB[j+16][k] - InB[j][k]; // B21-B11
      inputs[8][j][k] = InA[j][k] + InA[j][k+16]; // A11+A12
      inputs[9][j][k] = InB[j+16][k+16]; // B22
      inputs[10][j][k] = InA[j+16][k] - InA[j][k]; // A21-A11
      inputs[11][j][k] = InB[j][k] + InB[j][k+16]; // B11+B12
      inputs[12][j][k] = InA[j][k+16] - InA[j+16][k+16]; // A12-A22
      inputs[13][j][k] = InB[j+16][k] + InB[j+16][k+16]; // B21+B22
  }
}

strassen_32x32_solve:for (i = 0; i < 7; i++) {
    #ifdef NON_RECURSIVE_16
      mmm_16x16(inputs[2*i], inputs[2*i+1], mults[i]);
    #else
	    strassen_16x16(inputs[2*i], inputs[2*i+1], mults[i]);
    #endif
}

// create outputs
strassen_32x32_out_0:for (j = 0; j < 16; j++) {
  strassen_32x32_out_1:for (k = 0; k < 16; k++) {
        OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        OutC[j][k+16] = mults[2][j][k] + mults[4][j][k];  // C12
        OutC[j+16][k] = mults[1][j][k] + mults[3][j][k];  // C21
        OutC[j+16][k+16] = mults[0][j][k] - mults[1][j][k] + mults[2][j][k] + mults[5][j][k];  // C22

    }
  }
}


void strassen_16x16(cnndata_t InA[16][16],
                    cnndata_t InB[16][16],
                    cnndata_t OutC[16][16]) {

  index_t i, j, k;

  cnndata_t inputs[14][8][8];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=lutram
  cnndata_t mults[7][8][8];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=lutram

strassen_16x16_in_0:for (j = 0; j < 8; j++) {
  strassen_16x16_in_1:for (k = 0; k < 8; k++) {
#pragma HLS PIPELINE
      inputs[0][j][k] = InA[j][k] + InA[j+8][k+8]; // A11+A22
      inputs[1][j][k] = InB[j][k] + InB[j+8][k+8]; // B11+B22
      inputs[2][j][k] = InA[j+8][k] + InA[j+8][k+8]; // A21+A22
      inputs[3][j][k] = InB[j][k]; // B11
      inputs[4][j][k] = InA[j][k]; // A11
      inputs[5][j][k] = InB[j][k+8] - InB[j+8][k+8]; // B12-B22
      inputs[6][j][k] = InA[j+8][k+8]; // A22
      inputs[7][j][k] = InB[j+8][k] - InB[j][k]; // B21-B11
      inputs[8][j][k] = InA[j][k] + InA[j][k+8]; // A11+A12
      inputs[9][j][k] = InB[j+8][k+8]; // B22
      inputs[10][j][k] = InA[j+8][k] - InA[j][k]; // A21-A11
      inputs[11][j][k] = InB[j][k] + InB[j][k+8]; // B11+B12
      inputs[12][j][k] = InA[j][k+8] - InA[j+8][k+8]; // A12-A22
      inputs[13][j][k] = InB[j+8][k] + InB[j+8][k+8]; // B21+B22
  }
}

strassen_16x16_solve:for (i = 0; i < 7; i++) {
    #ifdef NON_RECURSIVE_8
      mmm_8x8(inputs[2*i], inputs[2*i+1], mults[i]);
    #else
	    strassen_8x8(inputs[2*i], inputs[2*i+1], mults[i]);
    #endif

}

// create outputs
  strassen_16x16_out_0:for (j = 0; j < 8; j++) {
    strassen_16x16_out_1:for (k = 0; k < 8; k++) {
#pragma HLS PIPELINE
        OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        OutC[j][k+8] = mults[2][j][k] + mults[4][j][k];  // C12
        OutC[j+8][k] = mults[1][j][k] + mults[3][j][k];  // C21
        OutC[j+8][k+8] = mults[0][j][k] - mults[1][j][k] + mults[2][j][k] + mults[5][j][k];  // C22

    }
  }
}

void strassen_8x8(cnndata_t InA[8][8],
                    cnndata_t InB[8][8],
                    cnndata_t OutC[8][8]) {

  index_t i, j, k;

  cnndata_t inputs[14][4][4];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=lutram

  cnndata_t mults[7][4][4];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=lutram

	strassen_8x8_in_0:for (j = 0; j < 4; j++) {
	  strassen_8x8_in_1:for (k = 0; k < 4; k++) {
#pragma HLS PIPELINE
//#pragma HLS PIPELINE
		  inputs[0][j][k] = InA[j][k] + InA[j+4][k+4]; // A11+A22
		  inputs[1][j][k] = InB[j][k] + InB[j+4][k+4]; // B11+B22
		  inputs[2][j][k] = InA[j+4][k] + InA[j+4][k+4]; // A21+A22
		  inputs[3][j][k] = InB[j][k]; // B11
		  inputs[4][j][k] = InA[j][k]; // A11
		  inputs[5][j][k] = InB[j][k+4] - InB[j+4][k+4]; // B12-B22
		  inputs[6][j][k] = InA[j+4][k+4]; // A22
		  inputs[7][j][k] = InB[j+4][k] - InB[j][k]; // B21-B11
		  inputs[8][j][k] = InA[j][k] + InA[j][k+4]; // A11+A12
		  inputs[9][j][k] = InB[j+4][k+4]; // B22
		  inputs[10][j][k] = InA[j+4][k] - InA[j][k]; // A21-A11
		  inputs[11][j][k] = InB[j][k] + InB[j][k+4]; // B11+B12
		  inputs[12][j][k] = InA[j][k+4] - InA[j+4][k+4]; // A12-A22
		  inputs[13][j][k] = InB[j+4][k] + InB[j+4][k+4]; // B21+B22
	  }
	}

	strassen_8x8_solve:for (i = 0; i < 7; i++) {
//#pragma HLS UNROLL factor=2

    #ifdef NON_RECURSIVE_4
      mmm_4x4(inputs[2*i], inputs[2*i+1], mults[i]);
    #else
	    strassen_4x4(inputs[2*i], inputs[2*i+1], mults[i]);
    #endif
	}

	// create outputs
  strassen_8x8_out_0:for (j = 0; j < 4; j++) {
		strassen_8x8_out_1:for (k = 0; k < 4; k++) {
#pragma HLS PIPELINE
//#pragma HLS PIPELINE
//			OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[6][j][k];  // C11 THIS WAS AN ISSUE
			OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];
			OutC[j][k+4] = mults[2][j][k] + mults[4][j][k];  // C12
			OutC[j+4][k] = mults[1][j][k] + mults[3][j][k];  // C21
			OutC[j+4][k+4] = mults[0][j][k] - mults[1][j][k] + mults[2][j][k] + mults[5][j][k];  // C22
		}
  }
}

void strassen_4x4(cnndata_t InA[4][4],
                    cnndata_t InB[4][4],
                    cnndata_t OutC[4][4]) {

  index_t i, j, k;

  cnndata_t inputs[14][2][2];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=lutram

  cnndata_t mults[7][2][2];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=lutram


	strassen_4x4_in_0:for (j = 0; j < 2; j++) {
	  strassen_4x4_in_1:for (k = 0; k < 2; k++) {
#pragma HLS UNROLL
		  inputs[0][j][k] = InA[j][k] + InA[j+2][k+2]; // A11+A22
		  inputs[1][j][k] = InB[j][k] + InB[j+2][k+2]; // B11+B22
		  inputs[2][j][k] = InA[j+2][k] + InA[j+2][k+2]; // A21+A22
		  inputs[3][j][k] = InB[j][k]; // B11
		  inputs[4][j][k] = InA[j][k]; // A11
		  inputs[5][j][k] = InB[j][k+2] - InB[j+2][k+2]; // B12-B22
		  inputs[6][j][k] = InA[j+2][k+2]; // A22
		  inputs[7][j][k] = InB[j+2][k] - InB[j][k]; // B21-B11
		  inputs[8][j][k] = InA[j][k] + InA[j][k+2]; // A11+A12
		  inputs[9][j][k] = InB[j+2][k+2]; // B22
		  inputs[10][j][k] = InA[j+2][k] - InA[j][k]; // A21-A11
		  inputs[11][j][k] = InB[j][k] + InB[j][k+2]; // B11+B12
		  inputs[12][j][k] = InA[j][k+2] - InA[j+2][k+2]; // A12-A22
		  inputs[13][j][k] = InB[j+2][k] + InB[j+2][k+2]; // B21+B22
	  }
	}

	strassen_4x4_solve: for (i = 0; i < 7; i++) {
#pragma HLS UNROLL
    #ifdef NON_RECURSIVE_2
      mmm_2x2(inputs[2*i], inputs[2*i+1], mults[i]);
    #else
	    strassen_2x2(inputs[2*i], inputs[2*i+1], mults[i]);
    #endif

	}

	// create outputs
  strassen_4x4_out_0:for (j = 0; j < 2; j++) {
   strassen_4x4_out_1:for (k = 0; k < 2; k++) {
#pragma HLS UNROLL
			OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
			OutC[j][k+2] = mults[2][j][k] + mults[4][j][k];  // C12
			OutC[j+2][k] = mults[1][j][k] + mults[3][j][k];  // C21
			OutC[j+2][k+2] = mults[0][j][k] - mults[1][j][k] + mults[2][j][k] + mults[5][j][k];  // C22

		}
  }
}

void strassen_2x2(cnndata_t InA[2][2], cnndata_t InB[2][2], cnndata_t OutC[2][2]) {
 index_t i, j, k;
  cnndata_t inputs[14];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=lutram
  cnndata_t mults[7];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=lutram
      inputs[0]= InA[0][0] + InA[1][1]; // A11+A22
      inputs[1]= InB[0][0] + InB[1][1]; // B11+B22
      inputs[2]= InA[1][0] + InA[1][1]; // A21+A22
      inputs[3]= InB[0][0]; // B11
      inputs[4]= InA[0][0]; // A11
      inputs[5]= InB[0][1] - InB[1][1]; // B12-B22
      inputs[6]= InA[1][1]; // A22
      inputs[7]= InB[1][0] - InB[0][0]; // B21-B11
      inputs[8]= InA[0][0] + InA[0][1]; // A11+A12
      inputs[9]= InB[1][1]; // B22
      inputs[10] = InA[1][0] - InA[0][0]; // A21-A11
      inputs[11] = InB[0][0] + InB[0][1]; // B11+B12
      inputs[12] = InA[0][1] - InA[1][1]; // A12-A22
      inputs[13] = InB[1][0] + InB[1][1]; // B21+B22

// create outputs
      strassen_2x2_solve:for (i = 0; i < 7; i++) {
#pragma HLS UNROLL
        mults[i] = inputs[2*i]*inputs[2*i+1];
      }

        OutC[0][0] = mults[0] + mults[3] - mults[4] + mults[6];  // C11
        OutC[0][1] = mults[2] + mults[4];  // C12
        OutC[1][0] = mults[1] + mults[3];  // C21
        OutC[1][1] = mults[0] - mults[1] + mults[2] + mults[5];  // C22
}

void mmm_512x512(cnndata_t InA[512][512], cnndata_t InB[512][512], cnndata_t OutC[512][512]) {
  std::cout << "--- non-recursive mmm_512x512 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_512x512_0:for (i = 0; i < 512; i++) {
    mmm_512x512_1:for (j = 0; j < 512; j++) {
      output = 0;
      mmm_512x512_2:for (k = 0; k < 512; k++){
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}

void mmm_256x256(cnndata_t InA[256][256], cnndata_t InB[256][256], cnndata_t OutC[256][256]) {
  std::cout << "--- non-recursive mmm_256x256 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_256x256_0:for (i = 0; i < 256; i++) {
    mmm_256x256_1:for (j = 0; j < 256; j++) {
      output = 0;
      mmm_256x256_2:for (k = 0; k < 256; k++){
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}

void mmm_128x128(cnndata_t InA[128][128], cnndata_t InB[128][128], cnndata_t OutC[128][128]) {
  std::cout << "--- non-recursive mmm_128x128 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_128x128_0:for (i = 0; i < 128; i++) {
    mmm_128x128_1:for (j = 0; j < 128; j++) {
      output = 0;
      mmm_128x128_2:for (k = 0; k < 128; k++){
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}

void mmm_64x64(cnndata_t InA[64][64], cnndata_t InB[64][64], cnndata_t OutC[64][64]) {
  std::cout << "--- non-recursive mmm_64x64 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_64x64_0:for (i = 0; i < 64; i++) {
    mmm_64x64_1:for (j = 0; j < 64; j++) {
      output = 0;
      mmm_64x64_2:for (k = 0; k < 64; k++){
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}

void mmm_32x32(cnndata_t InA[32][32], cnndata_t InB[32][32], cnndata_t OutC[32][32]) {
  std::cout << "--- non-recursive mmm_32x32 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_32x32_0:for (i = 0; i < 32; i++) {
    mmm_32x32_1:for (j = 0; j < 32; j++) {
      output = 0;
      mmm_32x32_2:for (k = 0; k < 32; k++) {
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}

void mmm_16x16(cnndata_t InA[16][16], cnndata_t InB[16][16], cnndata_t OutC[16][16]) {
  std::cout << "--- non-recursive mmm_16x16 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_16x16_0:for (i = 0; i < 16; i++) {
    mmm_16x16_1:for (j = 0; j < 16; j++) {
      output = 0;
      mmm_16x16_2:for (k = 0; k < 16; k++) {
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}

void mmm_8x8(cnndata_t InA[8][8], cnndata_t InB[8][8], cnndata_t OutC[8][8]) {
  std::cout << "--- non-recursive mmm_8x8 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_8x8_0:for (i = 0; i < 8; i++) {
    mmm_8x8_1:for (j = 0; j < 8; j++) {
      output = 0;
      mmm_8x8_2:for (k = 0; k < 8; k++) {
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}


void mmm_4x4(cnndata_t InA[4][4], cnndata_t InB[4][4], cnndata_t OutC[4][4]) {
  std::cout << "--- non-recursive mmm_4x4 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_4x4_0:for (i = 0; i < 4; i++) {
    mmm_4x4_1:for (j = 0; j < 4; j++) {
      output = 0;
      mmm_4x4_2:for (k = 0; k < 4; k++) {
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}

void mmm_2x2(cnndata_t InA[2][2], cnndata_t InB[2][2], cnndata_t OutC[2][2]) {
  std::cout << "--- non-recursive mmm_2x2 ---" << std::endl;
  index_t i, j, k;
  cnndata_t output;
	mmm_2x2_0:for (i = 0; i < 2; i++) {
    mmm_2x2_1:for (j = 0; j < 2; j++) {
      output = 0;
      mmm_2x2_2:for (k = 0; k < 2; k++) {
        output += InA[i][k]*InB[k][j];
      }
      OutC[i][j] = output;
    }
  }
}
