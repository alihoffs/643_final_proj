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

// Prevent aliasing
#undef BATCH_SIZE
#undef R_OFM
#undef C_OFM
#undef R_IFM
#undef C_IFM
#undef M_OFM
#undef N_IFM

#include "util643.h"

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
//void strassen_2x2(cnndata_t InA[2][2],
//                  cnndata_t InB[2][2],
//                  cnndata_t OutC[2][2]);
void strassen_2x2(cnndata_t InA11, cnndata_t InA12, cnndata_t InA21, cnndata_t InA22,
                    cnndata_t InB11, cnndata_t InB12, cnndata_t InB21, cnndata_t InB22,
					cnndata_t OutC11, cnndata_t OutC12, cnndata_t OutC21, cnndata_t OutC22);

#ifdef __VITIS_CL__
extern "C" {
#endif
void krnl_cnn_layerX(const cnndata_t* inA, const cnndata_t* inB,
        cnndata_t* OutC) {

  index_t i, j, k;

  cnndata_t inputs[14][32][32];
  cnndata_t mults[7][32][32];
  cnndata_t elem_a, elem_b;

  // initialize inputs
  // initialize A21
  for (j = 0; j < 32; j++) {
    for (k = 0; k < 32; k++) {
    	elem_a = ARRAYi_X(inA, j+32, k, 64, 64);
    	elem_b = ARRAYi_X(inB, j+32, k, 64, 64);
        inputs[2][j][k] = elem_a; // A21+A22
        inputs[10][j][k] = elem_a; // A21-A11

        inputs[7][j][k] = elem_b; // B21-B11
        inputs[13][j][k] = elem_b; // B21+B22
    }
  }

  // initialize A12
  for (j = 0; j < 32; j++) {
    for (k = 0; k < 32; k++) {
    	elem_a = ARRAYi_X(inA, j, k+32, 64, 64);
    	elem_b = ARRAYi_X(inB, j, k+32, 64, 64);
        inputs[8][j][k] = elem_a;// A11+A12
        inputs[12][j][k] = elem_a; // A12-A22

        inputs[5][j][k] = elem_b; // B12-B22
        inputs[11][j][k] = elem_b; // B11+B12
    }
  }

  // initialize A11
  for (j = 0; j < 32; j++) {
    for (k = 0; k < 32; k++) {
       	elem_a = ARRAYi_X(inA, j, k, 64, 64);
		elem_b = ARRAYi_X(inB, j, k, 64, 64);
        inputs[0][j][k] = elem_a; // A11+A22
        inputs[4][j][k] = elem_a; // A11
        inputs[8][j][k] +=  elem_a;// A11+A12
        inputs[10][j][k] -= elem_a; // A21-A11

        inputs[1][j][k] = elem_b; // B11+B22
        inputs[3][j][k] = elem_b; // B11
        inputs[7][j][k] -= elem_b; // B21-B11
        inputs[11][j][k] += elem_b; // B11+B12
    }
  }

  // initialize A22
  for (j = 0; j < 32; j++) {
    for (k = 0; k < 32; k++) {
       	elem_a = ARRAYi_X(inA, j+32, k+32, 64, 64);
		elem_b = ARRAYi_X(inB, j+32, k+32, 64, 64);
        inputs[0][j][k] += elem_a; // A11+A22
        inputs[2][j][k] += elem_a; // A21+A22
        inputs[6][j][k] = elem_a; // A22
        inputs[12][j][k] -= elem_a; // A12-A22

        inputs[1][j][k] += elem_b; // B11+B22
        inputs[5][j][k] -= elem_b; // B12-B22
        inputs[9][j][k] = elem_b; // B22
        inputs[13][j][k] += elem_b; // B21+B22


    }
  }

//  //initialize B21
//  for (j = 0; j < 32; j++) {
//    for (k = 0; k < 32; k++) {
//        inputs[7][j][k] = ARRAYi_X(inB, j+32, k, 64, 64); // B21-B11
//        inputs[13][j][k] = ARRAYi_X(inB, j+32, k, 64, 64); // B21+B22
//    }
//  }
//
//  // initialize B12
//  for (j = 0; j < 32; j++) {
//    for (k = 0; k < 32; k++) {
//        inputs[5][j][k] = ARRAYi_X(inB, j, k+32, 64, 64); // B12-B22
//        inputs[11][j][k] = ARRAYi_X(inB, j, k+32, 64, 64); // B11+B12
//    }
//  }
//
//  // initialize B11
//  for (j = 0; j < 32; j++) {
//    for (k = 0; k < 32; k++) {
//        inputs[1][j][k] = ARRAYi_X(inB, j, k, 64, 64); // B11+B22
//        inputs[3][j][k] = ARRAYi_X(inB, j, k, 64, 64); // B11
//        inputs[7][j][k] -= ARRAYi_X(inB, j, k, 64, 64); // B21-B11
//        inputs[11][j][k] += ARRAYi_X(inB, j, k, 64, 64); // B11+B12
//    }
//  }
//
//  // initialize B22
//  for (j = 0; j < 32; j++) {
//    for (k = 0; k < 32; k++) {
//        inputs[1][j][k] += ARRAYi_X(inB, j+32, k+32, 64, 64); // B11+B22
//        inputs[5][j][k] -= ARRAYi_X(inB, j+32, k+32, 64, 64); // B12-B22
//        inputs[9][j][k] = ARRAYi_X(inB, j+32, k+32, 64, 64); // B22
//        inputs[13][j][k] += ARRAYi_X(inB, j+32, k+32, 64, 64); // B21+B22
//    }
//  }
  // for (j = 0; j < 16; j++) {
  //   for (k = 0; k < 16; k++) {
  //       inputs[0][j][k] = ARRAYi_X(inA, j, k, 64, 64) + ARRAYi_X(inA, j+32, k+32, 64, 64); // A11+A22
  //       inputs[1][j][k] = ARRAYi_X(inB, j, k, 64, 64) + ARRAYi_X(inB, j+32, k+32, 64, 64); // B11+B22
  //       inputs[2][j][k] = ARRAYi_X(inA, j+32, k, 64, 64) + ARRAYi_X(inA, j+32, k+32, 64, 64); // A21+A22
  //       inputs[3][j][k] = ARRAYi_X(inB, j, k, 64, 64); // B11
  //       inputs[4][j][k] = ARRAYi_X(inA, j, k, 64, 64); // A11
  //       inputs[5][j][k] = ARRAYi_X(inB, j, k+32, 64, 64) - ARRAYi_X(inB, j+32, k+32, 64, 64); // B12-B22
  //       inputs[6][j][k] = ARRAYi_X(inA, j+32, k+32, 64, 64); // A22
  //       inputs[7][j][k] = ARRAYi_X(inB, j+32, k, 64, 64) - ARRAYi_X(inB, j, k, 64, 64); // B21-B11
  //       inputs[8][j][k] = ARRAYi_X(inA, j, k, 64, 64) +  ARRAYi_X(inA, j, k+32, 64, 64);// A11+A12
  //       inputs[9][j][k] = ARRAYi_X(inB, j+32, k+32, 64, 64); // B22
  //       inputs[10][j][k] = ARRAYi_X(inA, j+32, k, 64, 64) - ARRAYi_X(inA, j, k, 64, 64); // A21-A11
  //       inputs[11][j][k] = ARRAYi_X(inB, j, k, 64, 64) + ARRAYi_X(inB, j+32, k+32, 64, 64); // B11+B12
  //       inputs[12][j][k] = ARRAYi_X(inA, j, k+32, 64, 64) - ARRAYi_X(inA, j+32, k+32, 64, 64); // A12-A22
  //       inputs[13][j][k] = ARRAYi_X(inB, j+32, k, 64, 64) + ARRAYi_X(inB, j+32, k+32, 64, 64); // B21+B22
  //   }
  // }

  for (i = 0; i < 7; i++) {
    strassen_32x32(inputs[2*i], inputs[2*i+1], mults[i]);
  }

// create outputs
/*
    output[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
    output[j][k+8] = mults[2][j][k] + mults[4][j][k];  // C12
    output[j+8][k] = mults[3][j][k] + mults[5][j][k];  // C21
    output[j+8][k+8] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22
*/
  for (j = 0; j < 32; j++) {
    for (k = 0; k < 32; k++) {
      ARRAYi_X(OutC, j, k, 64, 64) = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
    }
  }
  for (j = 0; j < 32; j++) {
    for (k = 0; k < 32; k++) {
      ARRAYi_X(OutC, j, k+32, 64, 64) = mults[2][j][k] + mults[4][j][k];  // C12
    }
  }
  for (j = 0; j < 32; j++) {
    for (k = 0; k < 32; k++) {
      ARRAYi_X(OutC, j+32, k, 64, 64) = mults[3][j][k] + mults[5][j][k];  // C21
    }
  }
  for (j = 0; j < 32; j++) {
    for (k = 0; k < 32; k++) {
      ARRAYi_X(OutC, j+32, k+32, 64, 64) = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22
    }
  }

}

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif
void strassen_32x32(cnndata_t InA[32][32],
                    cnndata_t InB[32][32],
                    cnndata_t OutC[32][32]) {

  index_t i, j, k;

  cnndata_t inputs[14][16][16];
  cnndata_t mults[7][16][16];

// initialize inputs
// // initialize A21
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[2][j][k] = InA[j+8][k]; // A21+A22
//         inputs[10][j][k] = InA[j+8][k]; // A21-A11
//     }
//   }

//   // initialize A12
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[8][j][k] = InA[j][k+8];// A11+A12
//         inputs[12][j][k] = InA[j][k+8]; // A12-A22
//     }
//   }

//   // initialize A11
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[0][j][k] = InA[j][k]; // A11+A22
//         inputs[4][j][k] = InA[j][k]; // A11
//         inputs[8][j][k] +=  InA[j][k]; // A11+A12
//         inputs[10][j][k] -= InA[j][k]; // A21-A11

//     }
//   }

//   // initialize A22
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[0][j][k] += InA[j+8][k+8]; // A11+A22
//         inputs[2][j][k] += InA[j+8][k+8]; // A21+A22
//         inputs[6][j][k] = InA[j+8][k+8]; // A22
//         inputs[12][j][k] -= InA[j+8][k+8]; // A12-A22
//     }
//   }

//   //initialize B21
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[7][j][k] = InB[j+8][k]; // B21-B11
//         inputs[13][j][k] = InB[j+8][k]; // B21+B22
//     }
//   }

//   // initialize B12
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[5][j][k] = InB[j][k+8]; // B12-B22
//         inputs[11][j][k] = InB[j][k+8]; // B11+B12
//     }
//   }

//   // initialize B11
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[1][j][k] = InB[j][k]; // B11+B22
//         inputs[3][j][k] = InB[j][k]; // B11
//         inputs[7][j][k] -= InB[j][k];// B21-B11
//         inputs[11][j][k] += InB[j][k]; // B11+B12
//     }
//   }

//   // initialize B22
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[1][j][k] += InB[j+8][k+8]; // B11+B22
//         inputs[5][j][k] -= InB[j+8][k+8]; // B12-B22
//         inputs[9][j][k] = InB[j+8][k+8]; // B22
//         inputs[13][j][k] += InB[j+8][k+8]; // B21+B22
//     }
//   }
for (j = 0; j < 16; j++) {
  for (k = 0; k < 16; k++) {
#pragma HLS UNROLL factor=8
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

for (i = 0; i < 7; i++) {
  strassen_16x16(inputs[2*i], inputs[2*i+1], mults[i]);
}

// create outputs
  for (j = 0; j < 16; j++) {
    for (k = 0; k < 16; k++) {
#pragma HLS UNROLL factor=8
        OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        OutC[j][k+16] = mults[2][j][k] + mults[4][j][k];  // C12
        OutC[j+16][k] = mults[3][j][k] + mults[5][j][k];  // C21
        OutC[j+16][k+16] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22

    }
  }
}


void strassen_16x16(cnndata_t InA[16][16],
                    cnndata_t InB[16][16],
                    cnndata_t OutC[16][16]) {

  index_t i, j, k;

  cnndata_t inputs[14][8][8];
  cnndata_t mults[7][8][8];

// initialize inputs
// // initialize A21
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[2][j][k] = InA[j+8][k]; // A21+A22
//         inputs[10][j][k] = InA[j+8][k]; // A21-A11
//     }
//   }

//   // initialize A12
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[8][j][k] = InA[j][k+8];// A11+A12
//         inputs[12][j][k] = InA[j][k+8]; // A12-A22
//     }
//   }

//   // initialize A11
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[0][j][k] = InA[j][k]; // A11+A22
//         inputs[4][j][k] = InA[j][k]; // A11
//         inputs[8][j][k] +=  InA[j][k]; // A11+A12
//         inputs[10][j][k] -= InA[j][k]; // A21-A11

//     }
//   }

//   // initialize A22
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[0][j][k] += InA[j+8][k+8]; // A11+A22
//         inputs[2][j][k] += InA[j+8][k+8]; // A21+A22
//         inputs[6][j][k] = InA[j+8][k+8]; // A22
//         inputs[12][j][k] -= InA[j+8][k+8]; // A12-A22
//     }
//   }

//   //initialize B21
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[7][j][k] = InB[j+8][k]; // B21-B11
//         inputs[13][j][k] = InB[j+8][k]; // B21+B22
//     }
//   }

//   // initialize B12
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[5][j][k] = InB[j][k+8]; // B12-B22
//         inputs[11][j][k] = InB[j][k+8]; // B11+B12
//     }
//   }

//   // initialize B11
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[1][j][k] = InB[j][k]; // B11+B22
//         inputs[3][j][k] = InB[j][k]; // B11
//         inputs[7][j][k] -= InB[j][k];// B21-B11
//         inputs[11][j][k] += InB[j][k]; // B11+B12
//     }
//   }

//   // initialize B22
//   for (j = 0; j < 8; j++) {
//     for (k = 0; k < 8; k++) {
//         inputs[1][j][k] += InB[j+8][k+8]; // B11+B22
//         inputs[5][j][k] -= InB[j+8][k+8]; // B12-B22
//         inputs[9][j][k] = InB[j+8][k+8]; // B22
//         inputs[13][j][k] += InB[j+8][k+8]; // B21+B22
//     }
//   }
for (j = 0; j < 8; j++) {
  for (k = 0; k < 8; k++) {
#pragma HLS UNROLL factor=4
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

for (i = 0; i < 7; i++) {
  strassen_8x8(inputs[2*i], inputs[2*i+1], mults[i]);
}

// create outputs
  for (j = 0; j < 8; j++) {
    for (k = 0; k < 8; k++) {
#pragma HLS UNROLL factor=4
        OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        OutC[j][k+8] = mults[2][j][k] + mults[4][j][k];  // C12
        OutC[j+8][k] = mults[3][j][k] + mults[5][j][k];  // C21
        OutC[j+8][k+8] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22

    }
  }
}

void strassen_8x8(cnndata_t InA[8][8],
                    cnndata_t InB[8][8],
                    cnndata_t OutC[8][8]) {

  index_t i, j, k;

  cnndata_t inputs[14][4][4];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p

  cnndata_t mults[7][4][4];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p



// initialize inputs
// // initialize A21
//   for (j = 0; j < 4; j++) {
//     for (k = 0; k < 4; k++) {
//         inputs[2][j][k] = InA[j+4][k]; // A21+A22
//         inputs[10][j][k] = InA[j+4][k]; // A21-A11
//     }
//   }

//   // initialize A12
//   for (j = 0; j < 4; j++) {
//     for (k = 0; k < 4; k++) {
//         inputs[8][j][k] = InA[j][k+4];// A11+A12
//         inputs[12][j][k] = InA[j][k+4]; // A12-A22
//     }
//   }

//   // initialize A11
//   for (j = 0; j < 4; j++) {
//     for (k = 0; k < 4; k++) {
//         inputs[0][j][k] = InA[j][k]; // A11+A22
//         inputs[4][j][k] = InA[j][k]; // A11
//         inputs[8][j][k] +=  InA[j][k]; // A11+A12
//         inputs[10][j][k] -= InA[j][k]; // A21-A11

//     }
//   }

//   // initialize A22
//   for (j = 0; j < 4; j++) {
//     for (k = 0; k < 4; k++) {
//         inputs[0][j][k] += InA[j+4][k+4]; // A11+A22
//         inputs[2][j][k] += InA[j+4][k+4]; // A21+A22
//         inputs[6][j][k] = InA[j+4][k+4]; // A22
//         inputs[12][j][k] -= InA[j+4][k+4]; // A12-A22
//     }
//   }

//   //initialize B21
//   for (j = 0; j < 4; j++) {
//     for (k = 0; k < 4; k++) {
//         inputs[7][j][k] = InB[j+4][k]; // B21-B11
//         inputs[13][j][k] = InB[j+4][k]; // B21+B22
//     }
//   }

//   // initialize B12
//   for (j = 0; j < 4; j++) {
//     for (k = 0; k < 4; k++) {
//         inputs[5][j][k] = InB[j][k+4]; // B12-B22
//         inputs[11][j][k] = InB[j][k+4]; // B11+B12
//     }
//   }

//   // initialize B11
//   for (j = 0; j < 4; j++) {
//     for (k = 0; k < 4; k++) {
//         inputs[1][j][k] = InB[j][k]; // B11+B22
//         inputs[3][j][k] = InB[j][k]; // B11
//         inputs[7][j][k] -= InB[j][k];// B21-B11
//         inputs[11][j][k] += InB[j][k]; // B11+B12
//     }
//   }

//   // initialize B22
//   for (j = 0; j < 4; j++) {
//     for (k = 0; k < 4; k++) {
//         inputs[1][j][k] += InB[j+4][k+4]; // B11+B22
//         inputs[5][j][k] -= InB[j+4][k+4]; // B12-B22
//         inputs[9][j][k] = InB[j+4][k+4]; // B22
//         inputs[13][j][k] += InB[j+4][k+4]; // B21+B22
//     }
//   }
	for (j = 0; j < 4; j++) {

	  for (k = 0; k < 4; k++) {
#pragma HLS UNROLL factor=2
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
	  strassen_4x4(inputs[2*i], inputs[2*i+1], mults[i]);
	}

	// create outputs
	  for (j = 0; j < 4; j++) {

		for (k = 0; k < 4; k++) {
#pragma HLS UNROLL factor=2
			OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
			OutC[j][k+4] = mults[2][j][k] + mults[4][j][k];  // C12
			OutC[j+4][k] = mults[3][j][k] + mults[5][j][k];  // C21
			OutC[j+4][k+4] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22

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


// initialize inputs
// // initialize A21
//   for (j = 0; j < 2; j++) {
//     for (k = 0; k < 2; k++) {
//         inputs[2][j][k] = InA[j+2][k]; // A21+A22
//         inputs[10][j][k] = InA[j+2][k]; // A21-A11
//     }
//   }

//   // initialize A12
//   for (j = 0; j < 2; j++) {
//     for (k = 0; k < 2; k++) {
//         inputs[8][j][k] = InA[j][k+2];// A11+A12
//         inputs[12][j][k] = InA[j][k+2]; // A12-A22
//     }
//   }

//   // initialize A11
//   for (j = 0; j < 2; j++) {
//     for (k = 0; k < 2; k++) {
//         inputs[0][j][k] = InA[j][k]; // A11+A22
//         inputs[4][j][k] = InA[j][k]; // A11
//         inputs[8][j][k] +=  InA[j][k]; // A11+A12
//         inputs[10][j][k] -= InA[j][k]; // A21-A11

//     }
//   }

//   // initialize A22
//   for (j = 0; j < 2; j++) {
//     for (k = 0; k < 2; k++) {
//         inputs[0][j][k] += InA[j+2][k+2]; // A11+A22
//         inputs[2][j][k] += InA[j+2][k+2]; // A21+A22
//         inputs[6][j][k] = InA[j+2][k+2]; // A22
//         inputs[12][j][k] -= InA[j+2][k+2]; // A12-A22
//     }
//   }

//   //initialize B21
//   for (j = 0; j < 2; j++) {
//     for (k = 0; k < 2; k++) {
//         inputs[7][j][k] = InB[j+2][k]; // B21-B11
//         inputs[13][j][k] = InB[j+2][k]; // B21+B22
//     }
//   }

//   // initialize B12
//   for (j = 0; j < 2; j++) {
//     for (k = 0; k < 2; k++) {
//         inputs[5][j][k] = InB[j][k+2]; // B12-B22
//         inputs[11][j][k] = InB[j][k+2]; // B11+B12
//     }
//   }

//   // initialize B11
//   for (j = 0; j < 2; j++) {
//     for (k = 0; k < 2; k++) {
//         inputs[1][j][k] = InB[j][k]; // B11+B22
//         inputs[3][j][k] = InB[j][k]; // B11
//         inputs[7][j][k] -= InB[j][k];// B21-B11
//         inputs[11][j][k] += InB[j][k]; // B11+B12
//     }
//   }

//   // initialize B22
//   for (j = 0; j < 2; j++) {
//     for (k = 0; k < 2; k++) {
//         inputs[1][j][k] += InB[j+2][k+2]; // B11+B22
//         inputs[5][j][k] -= InB[j+2][k+2]; // B12-B22
//         inputs[9][j][k] = InB[j+2][k+2]; // B22
//         inputs[13][j][k] += InB[j+2][k+2]; // B21+B22
//     }
//   }
	for (j = 0; j < 2; j++) {

	  for (k = 0; k < 2; k++) {
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

	  strassen_2x2(inputs[2*i][0][0], inputs[2*i][0][1], inputs[2*i][1][0], inputs[2*i][1][1],
			  	   inputs[2*i+1][0][0], inputs[2*i+1][0][1], inputs[2*i+1][1][0], inputs[2*i+1][1][1],
				  mults[i][0][0], mults[i][0][1], mults[i][1][0], mults[i][1][1]);
	}

	// create outputs
	  for (j = 0; j < 2; j++) {
		for (k = 0; k < 2; k++) {
#pragma HLS UNROLL
			OutC[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
			OutC[j][k+2] = mults[2][j][k] + mults[4][j][k];  // C12
			OutC[j+2][k] = mults[3][j][k] + mults[5][j][k];  // C21
			OutC[j+2][k+2] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22

		}
	  }
}

void strassen_2x2(cnndata_t InA11, cnndata_t InA12, cnndata_t InA21, cnndata_t InA22,
                    cnndata_t InB11, cnndata_t InB12, cnndata_t InB21, cnndata_t InB22,
					cnndata_t OutC11, cnndata_t OutC12, cnndata_t OutC21, cnndata_t OutC22) {

  index_t i, j, k;
  cnndata_t inputs[14];
#pragma HLS BIND_STORAGE variable=inputs type=ram_2p impl=lutram
  cnndata_t mults[7];
#pragma HLS BIND_STORAGE variable=mults type=ram_2p impl=lutram
      inputs[0]= InA11 + InA22; // A11+A22
      inputs[1]= InB11 + InB22; // B11+B22
      inputs[2]= InA21 + InA22; // A21+A22
      inputs[3]= InB11; // B11
      inputs[4]= InA11; // A11
      inputs[5]= InB12 - InB22; // B12-B22
      inputs[6]= InA22; // A22
      inputs[7]= InB21 - InB11; // B21-B11
      inputs[8]= InA11 + InA12; // A11+A12
      inputs[9]= InB22; // B22
      inputs[10] = InA21 - InA11; // A21-A11
      inputs[11] = InB11 + InB12; // B11+B12
      inputs[12] = InA12 - InA22; // A12-A22
      inputs[13] = InB21 + InB22; // B21+B22
  
// create outputs
      strassen_2x2_solve:for (i = 0; i < 7; i++) {
#pragma HLS UNROLL
        mults[i] = inputs[2*i]*inputs[2*i+1];
      }

        OutC11 = mults[0] + mults[3] - mults[4] + mults[6];  // C11
        OutC12 = mults[2] + mults[4];  // C12
        OutC21 = mults[3] + mults[5];  // C21
        OutC22 = mults[1] + mults[3] - mults[2] + mults[5];  // C22
}
