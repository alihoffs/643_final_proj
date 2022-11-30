/****************************************************************
 * Copyright (c) 2017~2022, 18-643 Course Staff, CMU
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

#include "lab3_kernels.h"
#include "cnn_helper.h"
#include <string.h>

// Set kernel arguments and execute it
#ifdef __VITIS_CL__
void cnn_run_kernel(cl_object &cl_obj, krnl_object &krnl_obj0,
        krnl_object &krnl_obj1) {
    cl_int err;

    //
    // Layer 0
    //

    std::cout << "Running kernel for layer 0..." << std::endl;

    // Get i/o buffers from kernel object
    cl::Buffer *buffer_inA = &cl_obj.buffers[0];
    cl::Buffer *buffer_inB = &cl_obj.buffers[1];
    cl::Buffer *buffer_out = &cl_obj.buffers[2];

    // Set the kernel Arguments
    uint64_t narg = 0;
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_inA));            // input
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_inB));           // weights
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_out));           // output

    // Data will be migrated to kernel space
    OCL_CHECK(err, err = cl_obj.q.enqueueMigrateMemObjects({*buffer_inA, *buffer_inB}, 0/* 0 means from host*/)); // flushes kernel cache

    // Launch the Kernel; this is nonblocking.
    OCL_CHECK(err, err = cl_obj.q.enqueueTask(*cl_obj.krnl));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    OCL_CHECK(err, cl_obj.q.enqueueMigrateMemObjects({*buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST)); // flushes kernel cache

    // Wait for all tasks to finish
    OCL_CHECK(err, cl_obj.q.finish());


    std::cout << "Kernel executions completed" << std::endl;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// DO NOT MODIFY BELOW THIS LINE ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static const std::string cnn_error_message =
        "Error: Result mismatch:\n"
        "i = %d CPU result = %d Device result = %d\n";

// Verify a single batch of data

bool verify(cnndata_t *ref, cnndata_t *checkit) {
    uint64_t row, col;


    for(row = 0; row < 128; row++) {
        for(col = 0; col < 128 ; col++) {
            cnndata_t refval = ARRAY2(ref,row,col,128,128);

            cnndata_t checkval = ARRAY2(checkit,row,col,128,128) ;

            if (!nearlyEqual(checkval, refval)) {
                printf("\n***Result does not match reference:  "
                        "row = %lu, col = %lu, refval = %lu, checkval = %lu***\n", row, col, refval, checkval);
                return 0;
            } else {
            	printf("\n it do match: %lu\n", checkval);
            }
        }
    }
    
    return 1;
}


bool cnn_check(cnndata_t *ptr_inA, cnndata_t *ptr_inB, cnndata_t *ptr_output,
        cnndata_t *ref_inA, cnndata_t *ref_inB, cnndata_t *ref_output,
        uint64_t layer) {


    uint64_t row, col, k;
    bool mismatch;

    std::cout << "Verifying inputs are the same..." << std::endl;
    if (!verify(ref_inA, ptr_inA)){
    	std::cout << "inA does not match..." << std::endl;
    }

    if (!verify(ref_inB, ptr_inB)){
    	std::cout << "inB does not match..." << std::endl;
    }
    std::cout << "Verifying matrix multiply result..." << std::endl;
    // initialize ref_output to 0 for easy multiply accumulate
    for (row = 0; row < 128; row++) {
        for (col = 0; col < 128; col++) {
            ref_output[row*128+col] = 0;
        }
    }

    // compute the correct result
    for (row = 0; row < 128; row++) {
        for (col = 0; col < 128; col++) {
            for (k = 0; k < 128; k++) {
                ref_output[row*128 + col] += ref_inA[row*128 + k]*ref_inB[k*128 + col];
            }
        }
    }

    if (!verify(ref_output, ptr_output)){
        mismatch = 1;
    }

    return mismatch;
}
/*
void print_params(uint64_t layer) {
    std::cout << "===== Printing the CNN parameters Layer "
              << layer << " ======" << std::endl;

    std::cout << "Batch size: " << (uint64_t) BATCH_SIZE << std::endl;

    printf("Layer Parameters: \nK_wts: \t%d\tS_wts:\t%d\nR_ofm:\t%d\tC_ofm:"
           "\t%d\tM_ofm:\t%d\tN_ifm:\t%d\n", K_WTS, S_WTS, R_OFM(layer),
           C_OFM(layer), M_OFM(layer), N_IFM(layer));

#ifdef ENABLE_DFX
    printf("Kernel Parameters: \nTr: \t%d\tTc:\t%d\tTm:\t%d\tTn:\t%d\n\n",
            layer ? TR_1 : TR_0, layer ? TC_1 : TC_0, layer ? TM_1 : TM_0,
            layer ? TN_1 : TN_0);
#else
    printf("Kernel Parameters: \nTr: \t%d\tTc:\t%d\tTm:\t%d\tTn:\t%d\n\n",
            TR_X, TC_X, TM_X, TN_X);
#endif
}
*/
void initialize_buffer(cnndata_t *ptr, unsigned size, bool notzero) {
    for (unsigned i = 0; i < size; i++) {
        ptr[i] = notzero ? (rand() % VRANGE) : 0;
    }
}
/*
void ZhangIsfpga15_1_fp(cnndata_t *input, cnndata_t *output, cnndata_t *weights,
        uint64_t iter, uint64_t layer) {
    uint64_t row, col, to, ti;

    for(row = 0; row < R_OFM(layer); row++) {
        for(col = 0; col < C_OFM(layer); col++) {
            for(to = 0; to < M_OFM(layer); to++) {
                ARRAY4(output, iter, to, row, col, BATCH_SIZE, M_OFM(layer),
                        R_OFM(layer), C_OFM(layer)) = 0;
            }
        }
    }

    for(row = 0; row < R_OFM(layer); row++) {
      for(col = 0; col < C_OFM(layer); col++) {
        for(to = 0; to < M_OFM(layer); to++) {
          for(ti = 0; ti < N_IFM(layer); ti++) {
            uint64_t i, j;
            for(i = 0; i < K_WTS; i++) {
              for(j = 0; j < K_WTS; j++) {
                ARRAY4(output, iter, to, row, col, BATCH_SIZE, M_OFM(layer),
                    R_OFM(layer), C_OFM(layer))
                    += ARRAY4(weights, to, ti, i, j, M_OFM(layer), N_IFM(layer),
                    K_WTS, K_WTS)
                    * ARRAY4(input, iter, ti, S_WTS * row + i, S_WTS * col + j,
                       BATCH_SIZE, N_IFM(layer), R_IFM(layer), C_IFM(layer));
              }
            }
          }
        }
      }
    }
}
*/
