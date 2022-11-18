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

#ifdef __VITIS_CL__
extern "C" {
#endif
void strassen_32x32(const cnndata_t* inA, const cnndata_t* inB,
        cnndata_t* outC, /*uint64_t batch_size,*/ uint64_t r_ofm, uint64_t c_ofm,
        uint64_t m_ofm, uint64_t n_ifm) {

  index_t i, j, k;

  cnndata_t inputs[14][16][16];
  cnndata_t mults[7][16][16];

// initialize inputs
for (j = 0; j < 16; j++) {
  for (k = 0; k < 16; k++) {
      inputs[0][j][k] = ARRAYi_X(inA, j, k, 32, 32) + ARRAYi_X(inA, j+16, k+16, 32, 32); // A11+A22
      inputs[1][j][k] = ARRAYi_X(inB, j, k, 32, 32) + ARRAYi_X(inB, j+16, k+16, 32, 32); // B11+B22
      inputs[2][j][k] = ARRAYi_X(inA, j+16, k, 32, 32) + ARRAYi_X(inA, j+16, k+16, 32, 32); // A21+A22
      inputs[3][j][k] = ARRAYi_X(inB, j, k, 32, 32); // B11
      inputs[4][j][k] = ARRAYi_X(inA, j, k, 32, 32); // A11
      inputs[5][j][k] = ARRAYi_X(inB, j, k+16, 32, 32) - ARRAYi_X(inB, j+16, k+16, 32, 32); // B12-B22
      inputs[6][j][k] = ARRAYi_X(inA, j+16, k+16, 32, 32); // A22
      inputs[7][j][k] = ARRAYi_X(inB, j+16, k, 32, 32) - ARRAYi_X(inB, j, k, 32, 32); // B21-B11
      inputs[8][j][k] = ARRAYi_X(inA, j, k, 32, 32) +  ARRAYi_X(inA, j, k+16, 32, 32);// A11+A12
      inputs[9][j][k] = ARRAYi_X(inB, j+16, k+16, 32, 32); // B22
      inputs[10][j][k] = ARRAYi_X(inA, j+16, k, 32, 32) - ARRAYi_X(inA, j, k, 32, 32); // A21-A11
      inputs[11][j][k] = ARRAYi_X(inB, j, k, 32, 32) + ARRAYi_X(inB, j+16, k+16, 32, 32); // B11+B12
      inputs[12][j][k] = ARRAYi_X(inA, j, k+16, 32, 32) - ARRAYi_X(inA, j+16, k+16, 32, 32); // A12-A22
      inputs[13][j][k] = ARRAYi_X(inB, j+16, k, 32, 32) + ARRAYi_X(inB, j+16, k+16, 32, 32); // B21+B22
  }
}

for (i = 0; i < 7; i++) {
  strassen_16x16(inputs[2*i], inputs[2*i+1], mults[i]);
}

// create outputs
/*
    output[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
    output[j][k+8] = mults[2][j][k] + mults[4][j][k];  // C12
    output[j+8][k] = mults[3][j][k] + mults[5][j][k];  // C21
    output[j+8][k+8] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22
*/
for (j = 0; j < 16; j++) {
  for (k = 0; k < 16; k++) {
    ARRAYi_X(outC, j, k, 32, 32) = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
    ARRAYi_X(outC, j, k+16, 32, 32) = mults[2][j][k] + mults[4][j][k];  // C12
    ARRAYi_X(outC, j+16, k, 32, 32) = mults[3][j][k] + mults[5][j][k];  // C21
    ARRAYi_X(outC, j+16, k+16, 32, 32) = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22

  }
}

  // for(iter = 0; iter < batch_size; iter++) {      // Batch Loop
  //   for(row = 0; row < r_ofm; row += TR_X) {      // Tiled Row Loop
  //     for(col = 0; col < c_ofm; col += TC_X) {    // Tiled Column Loop
  //       for(to = 0; to < m_ofm; to += TM_X) {     // Tiled Output Channel Loop
  //         // Temporary versions of incremented indices;
  //         // Same usage as in ZhangIsfpga_2()
  //         index_t trr, tcc, too, tii;

  //         // Only need to zero BufO in this loop ordering
  //         {
  //           // Indices internal to the block: count from 0
  //           index_t ioo, icc, irr;

  //           for(ioo = 0; ioo < TM_X; ioo++) {
  //             for(irr = 0; irr < TR_X; irr++) {
  //               for(icc = 0; icc < TC_X; icc++) {
  //                 BufO[ioo][irr][icc] = 0;
  //               }
  //             }
  //           }
  //         }

  //         // Tiled Input Channel Loop
  //         for(ti = 0; ti < n_ifm; ti += TN_X) {
  //           // Load active input feature map into local buffer
  //           {
  //             // Indices internal to the block: count from 0
  //             index_t irr, icc, iii;

  //             // Incremented temporary indices for input row and col
  //             index_t xrr, xcc;

  //             // Loop bounds
  //             index_t tii_max, xrr_max, xcc_max;
  //             tii_max = MIN(ti + TN_X, n_ifm);
  //             xrr_max = MIN(row + TR_X, r_ofm) * S_WTS + K_WTS - S_WTS;
  //             xcc_max = MIN(col + TC_X, c_ofm) * S_WTS + K_WTS - S_WTS;

  //             for(tii = ti, iii = 0; tii < tii_max; tii++, iii++) {
  //               for(xrr = row * S_WTS, irr = 0; xrr < xrr_max; xrr++, irr++) {
  //                 for(xcc = col * S_WTS, icc = 0; xcc < xcc_max; xcc++, icc++) {
  //                   BufI[iii][irr][icc] = ARRAYi_X(input, iter, tii, xrr, xcc,
  //                     batch_size, n_ifm, r_ifm, c_ifm);
  //                 }
  //               }
  //             }
  //           }

  //           // Load active weights into local buffer
  //           {
  //             // Indices internal to the block: count from 0
  //             index_t ioo, iii, irr, icc;

  //             // Loop bounds
  //             index_t too_max, tii_max;
  //             too_max = MIN(to + TM_X, m_ofm);
  //             tii_max = MIN(ti + TN_X, n_ifm);

  //             for(too = to, ioo = 0; too < too_max; too++, ioo++) {
  //               for(tii = ti, iii = 0; tii < tii_max; tii++, iii++) {
  //                 for(irr = 0; irr < K_WTS; irr++) {
  //                   for(icc = 0; icc < K_WTS; icc++) {
  //                     BufW[ioo][iii][irr][icc] = ARRAYw_X(weights, too, tii,
  //                       irr, icc, m_ofm, n_ifm, K_WTS, K_WTS);
  //                   }
  //                 }
  //               }

  //               /* Write 0s into over-run regions at the end;
  //                * This way convolve_kernel() accumulates correctly
  //                * without needing a special case
  //                */
  //               if (iii < TN_X) {
  //                 for(; iii < TN_X; iii++) {
  //                   for(irr = 0; irr < K_WTS; irr++) {
  //                     for(icc = 0; icc < K_WTS; icc++) {
  //                       BufW[ioo][iii][irr][icc] = 0;
  //                     }
  //                   }
  //                 }
  //               }
  //             }
  //           }

  //           // Call the blocked cnn kernel
  //           cnn_blocked_kernel(BufI, BufO, BufW);
  //         }

  //         // Unload finished active intermedaite output feature map from local
  //         // to full buffer
  //         {
  //           // Indices internal to the block: count from 0
  //           index_t ioo, icc, irr;

  //           // Loop bounds
  //           index_t too_max, tcc_max, trr_max;
  //           too_max = MIN(to + TM_X, m_ofm);
  //           tcc_max = MIN(col + TC_X, c_ofm);
  //           trr_max = MIN(row + TR_X, r_ofm);

  //           for(too = to, ioo = 0; too < too_max; too++, ioo++) {
  //             for(trr = row, irr = 0; trr < trr_max; trr++, irr++) {
  //               for(tcc = col, icc = 0; tcc < tcc_max; tcc++, icc++) {
  //                 ARRAYo_X(output, iter, too, trr, tcc, batch_size, m_ofm,
  //                   r_ofm, c_ofm) = BufO[ioo][irr][icc];
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
}

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif

void strassen_16x16(cnndata_t InA[16][16],
                    cnndata_t InB[16][16],
                    cnndata_t OutC[16][16]) {

  index_t i, j, k;

  cnndata_t inputs[14][8][8];
  cnndata_t mults[7][8][8];

// initialize inputs
for (j = 0; j < 8; j++) {
  for (k = 0; k < 8; k++) {
      inputs[0][j][k] = InA[j][k] + InA[j+8][k+8]; // A11+A22
      inputs[1][j][k] = InB[j][k] + InB[j+8][k+8]; // B11+B22
      inputs[2][j][k] = InA[j+8][k] + InA[j+8][k+8]; // A21+A22
      inputs[3][j][k] = InB[j][k]; // B11
      inputs[4][j][k] = InA[j][k]; // A11
      inputs[5][j][k] = InB[j][k+8] - InB[j+8][k+8]; // B12-B22
      inputs[6][j][k] = InA[j+8][k+8] // A22
      inputs[7][j][k] = InB[j+8][k] - InB[j][k]; // B21-B11
      inputs[8][j][k] = InA[j][k] + InA[j][k+8]; // A11+A12
      inputs[9][j][k] = InB[j+8][k+8]; // B22
      inputs[10][j][k] = InA[j+8][k] - InA[j][k]; // A21-A11
      inputs[11][j][k] = InB[j][k] + InB[j][k+8]; // B11+B12
      inputs[12][j][k] = InA[j][k+8] - InA[j+8][k+8]; // A12-A22
      inputs[13][j][k] = InB[j+8][k] + InB[j+8][k+8] // B21+B22
  }
}

for (i = 0; i < 7; i++) {
  strassen_8x8(inputs[2*i], inputs[2*i+1], mults[i]);
}

// create outputs
  for (j = 0; j < 8; j++) {
    for (k = 0; k < 8; k++) {
        output[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        output[j][k+8] = mults[2][j][k] + mults[4][j][k];  // C12
        output[j+8][k] = mults[3][j][k] + mults[5][j][k];  // C21
        output[j+8][k+8] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22

    }
  }
}

void strassen_8x8(cnndata_t InA[8][8],
                    cnndata_t InB[8][8],
                    cnndata_t OutC[8][8]) {

  index_t i, j, k;

  cnndata_t inputs[14][4][4];
  cnndata_t mults[7][4][4];

// initialize inputs
for (j = 0; j < 4; j++) {
  for (k = 0; k < 4; k++) {
      inputs[0][j][k] = InA[j][k] + InA[j+4][k+4]; // A11+A22
      inputs[1][j][k] = InB[j][k] + InB[j+4][k+4]; // B11+B22
      inputs[2][j][k] = InA[j+4][k] + InA[j+4][k+4]; // A21+A22
      inputs[3][j][k] = InB[j][k]; // B11
      inputs[4][j][k] = InA[j][k]; // A11
      inputs[5][j][k] = InB[j][k+4] - InB[j+4][k+4]; // B12-B22
      inputs[6][j][k] = InA[j+4][k+4] // A22
      inputs[7][j][k] = InB[j+4][k] - InB[j][k]; // B21-B11
      inputs[8][j][k] = InA[j][k] + InA[j][k+4]; // A11+A12
      inputs[9][j][k] = InB[j+4][k+4]; // B22
      inputs[10][j][k] = InA[j+4][k] - InA[j][k]; // A21-A11
      inputs[11][j][k] = InB[j][k] + InB[j][k+4]; // B11+B12
      inputs[12][j][k] = InA[j][k+4] - InA[j+4][k+4]; // A12-A22
      inputs[13][j][k] = InB[j+4][k] + InB[j+4][k+4] // B21+B22
  }
}

for (i = 0; i < 7; i++) {
  strassen_4x4(inputs[2*i], inputs[2*i+1], mults[i]);
}

// create outputs
  for (j = 0; j < 4; j++) {
    for (k = 0; k < 4; k++) {
        output[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        output[j][k+4] = mults[2][j][k] + mults[4][j][k];  // C12
        output[j+4][k] = mults[3][j][k] + mults[5][j][k];  // C21
        output[j+4][k+4] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22

    }
  }
}

void strassen_4x4(cnndata_t InA[4][4],
                    cnndata_t InB[4][4],
                    cnndata_t OutC[4][4]) {

  index_t i, j, k;

  cnndata_t inputs[14][2][2];
  cnndata_t mults[7][2][2];

// initialize inputs
for (j = 0; j < 2; j++) {
  for (k = 0; k < 2; k++) {
      inputs[0][j][k] = InA[j][k] + InA[j+2][k+2]; // A11+A22
      inputs[1][j][k] = InB[j][k] + InB[j+2][k+2]; // B11+B22
      inputs[2][j][k] = InA[j+2][k] + InA[j+2][k+2]; // A21+A22
      inputs[3][j][k] = InB[j][k]; // B11
      inputs[4][j][k] = InA[j][k]; // A11
      inputs[5][j][k] = InB[j][k+2] - InB[j+2][k+2]; // B12-B22
      inputs[6][j][k] = InA[j+2][k+2] // A22
      inputs[7][j][k] = InB[j+2][k] - InB[j][k]; // B21-B11
      inputs[8][j][k] = InA[j][k] + InA[j][k+2]; // A11+A12
      inputs[9][j][k] = InB[j+2][k+2]; // B22
      inputs[10][j][k] = InA[j+2][k] - InA[j][k]; // A21-A11
      inputs[11][j][k] = InB[j][k] + InB[j][k+2]; // B11+B12
      inputs[12][j][k] = InA[j][k+2] - InA[j+2][k+2]; // A12-A22
      inputs[13][j][k] = InB[j+2][k] + InB[j+2][k+2] // B21+B22
  }
}

for (i = 0; i < 7; i++) {
  strassen_2x2(inputs[2*i], inputs[2*i+1], mults[i]);
}

// create outputs
  for (j = 0; j < 2; j++) {
    for (k = 0; k < 2; k++) {
        output[j][k] = mults[0][j][k] + mults[3][j][k] - mults[4][j][k] + mults[6][j][k];  // C11
        output[j][k+2] = mults[2][j][k] + mults[4][j][k];  // C12
        output[j+2][k] = mults[3][j][k] + mults[5][j][k];  // C21
        output[j+2][k+2] = mults[1][j][k] + mults[3][j][k] - mults[2][j][k] + mults[5][j][k];  // C22

    }
  }
}

void strassen_2x2(cnndata_t InA[2][2],
                    cnndata_t InB[2][2],
                    cnndata_t OutC[2][2]) {

  index_t i, j, k;
  cnndata_t inputs[14];
  cnndata_t mults[7];
      inputs[0]= InA[0][0] + InA[1][1]; // A11+A22
      inputs[1]= InB[0][0] + InB[1][1]; // B11+B22
      inputs[2]= InA[1][0] + InA[1][1]; // A21+A22
      inputs[3]= InB[0][0]; // B11
      inputs[4]= InA[0][0]; // A11
      inputs[5]= InB[0][1] - InB[1][1]; // B12-B22
      inputs[6]= InA[1][1] // A22
      inputs[7]= InB[1][0] - InB[0][0]; // B21-B11
      inputs[8]= InA[0][0] + InA[0][1]; // A11+A12
      inputs[9]= InB[1][1]; // B22
      inputs[10] = InA[1][0] - InA[0][0]; // A21-A11
      inputs[11] = InB[0][0] + InB[0][1]; // B11+B12
      inputs[12] = InA[0][1] - InA[1][1]; // A12-A22
      inputs[13] = InB[1][0] + InB[1][1] // B21+B22
  
// create outputs
      for (i = 0; i < 7; i++) {
        mults[i] = inputs[2*i]*inputs[2*i+1];
      }

        output[0][0] = mults[0] + mults[3] - mults[4] + mults[6];  // C11
        output[0][1] = mults[2] + mults[4];  // C12
        output[1][0] = mults[3] + mults[5];  // C21
        output[1][1] = mults[1] + mults[3] - mults[2] + mults[5];  // C22
}
