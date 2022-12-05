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
#include <sys/time.h>

int main(int argc, char* argv[]) {
    struct timeval start_time, end_time;
    bool mismatch[2] = {false, false};

    cnndata_t *ptr_inA, *ptr_inB, *ptr_output;
    cnndata_t *ref_inA, *ref_inB, *ref_output;

    // TODO: Change to size
    // Compute the size of array in bytes
    uint64_t num_elem_inA = 256 * 256; //BATCH_SIZE * N_IFM(0) * R_IFM(0) * C_IFM(0);
    uint64_t num_elem_inB = 256 * 256;
    uint64_t num_elem_outputs = 256 * 256;

#ifdef __VITIS_CL__
    // Hard coding xclbin filenames, ignoring command line arguments
    std::string xclbinFilename[3] = {
        "binary_container_X.xclbin",
//        "binary_container_0.xclbin",
//        "binary_container_1.xclbin"
    };
#endif

    // print_params(0);
    // print_params(1);

    cl_object cl_obj;

#ifdef ENABLE_DFX
    krnl_object cnn_obj[2];
    cnn_obj[0].index = 0;
    cnn_obj[0].name = "krnl_cnn_layer0";

    cnn_obj[1].index = 1;
    cnn_obj[1].name = "krnl_cnn_layer1";
#else
    krnl_object cnn_obj;
    cnn_obj.index = 0;
    cnn_obj.name = "krnl_cnn_layerX";
#endif

#ifdef __VITIS_CL__
    std::cout << "===== Initialize device ======" << std::endl;
    initialize_device(cl_obj);

    std::cout << "===== Reading xclbin ======" << std::endl;
    // Read cnn
#ifdef ENABLE_DFX
    read_xclbin(xclbinFilename[1], cl_obj.bins);
    read_xclbin(xclbinFilename[2], cl_obj.bins);
#else
    read_xclbin(xclbinFilename[0], cl_obj.bins);
#endif

    std::cout << "\n===== Programming kernel ======" << std::endl;
#ifdef ENABLE_DFX
    program_kernel(cl_obj, cnn_obj[0]);
#else
    program_kernel(cl_obj, cnn_obj);
#endif
#endif

    std::cout << "\n===== Allocating buffers ======" << std::endl;
    uint64_t bufIdx=0;
    allocate_readonly_mem(cl_obj, (void**) &ptr_inA, bufIdx++,
            num_elem_inA * sizeof(cnndata_t));
    allocate_readonly_mem(cl_obj, (void**) &ptr_inB, bufIdx++,
            num_elem_inB * sizeof(cnndata_t));
    allocate_readwrite_mem(cl_obj, (void**) &ptr_output, bufIdx++,
            num_elem_outputs * sizeof(cnndata_t));
    // allocate_readonly_mem(cl_obj, (void**) &ptr_weight[1], bufIdx++,
    //         num_elem_weights[1] * sizeof(cnndata_t));
    // allocate_readwrite_mem(cl_obj, (void**) &ptr_output[1], bufIdx++,
    //         num_elem_outputs[1] * sizeof(cnndata_t));

    MALLOC_CHECK(ref_inA = new cnndata_t[num_elem_inA]);
    MALLOC_CHECK(ref_inB = new cnndata_t[num_elem_inB]);
    MALLOC_CHECK(ref_output = new cnndata_t[num_elem_outputs]);
    // MALLOC_CHECK(ref_weight[1] = new cnndata_t[num_elem_weights[1]]);
    // MALLOC_CHECK(ref_output[1] = new cnndata_t[num_elem_outputs[1]]);

    // Set randomized inputs in reference copy
    initialize_buffer(ref_inA, num_elem_inA, true);
    initialize_buffer(ref_inB, num_elem_inB, true);
    // initialize_buffer(ref_weight[1], num_elem_weights[1], true);

#ifdef ENABLE_DFX
    // copy ref copy to kernel use copy, converting to kernel expected layout
    // COPY_BUF4D(ref_input, ARRAY4, ptr_input, ARRAYi_0,
    //         BATCH_SIZE, N_IFM(0), R_IFM(0),  C_IFM(0));
    // COPY_BUF4D(ref_weight[0], ARRAY4, ptr_weight[0], ARRAYw_0,
    //         M_OFM(0), N_IFM(0), K_WTS, K_WTS);
    // COPY_BUF4D(ref_weight[1], ARRAY4, ptr_weight[1], ARRAYw_1,
    //         M_OFM(1), N_IFM(1), K_WTS, K_WTS);
#else
    // copy ref copy to kernel use copy, converting to kernel expected layout
    // TODO: need to change COPY_BUF4D
    COPY_BUF2D(ref_inA, ARRAY2, ptr_inA, ARRAYi_X,
            256,  256);
    COPY_BUF2D(ref_inB, ARRAY2, ptr_inB, ARRAYi_X,
            256, 256);
    // COPY_BUF2D(ref_weight[1], ARRAY4, ptr_weight[1], ARRAYw_X,
    //         64, 64);
#endif

    // Random initialize output for kernel use
    initialize_buffer(ptr_output, num_elem_outputs, false); // cannot assume 0'ed
    // initialize_buffer(ptr_output[1], num_elem_outputs[1], true); // cannot assume 0'ed

    std::cout << "\n===== Execution and Timing starts ======" << std::endl;
    gettimeofday(&start_time, NULL);

#ifdef __VITIS_CL__
#ifdef ENABLE_DFX
    cnn_run_kernel(cl_obj, cnn_obj[0], cnn_obj[1]);
#else
    cnn_run_kernel(cl_obj, cnn_obj, cnn_obj);
#endif
#else
#ifdef ENABLE_DFX
    krnl_cnn_layer0(ptr_input, ptr_weight[0], ptr_output[0], BATCH_SIZE);
    krnl_cnn_layer1(ptr_output[0], ptr_weight[1], ptr_output[1], BATCH_SIZE);
#else
    krnl_cnn_layerX(ptr_inA, ptr_inB, ptr_output);
    // krnl_cnn_layerX(ptr_output[0], ptr_weight[1], ptr_output[1], BATCH_SIZE, R_OFM(1), C_OFM(1), M_OFM(1), N_IFM(1));
#endif
#endif

    gettimeofday(&end_time, NULL);
    std::cout << "Execution and Timing finished!\n" << std::endl;

    std::cout << "===== Verification starts ======" << std::endl;
    mismatch[0] = cnn_check(ptr_inA, ptr_inB, ptr_output,
             ref_inA, ref_inB, ref_output, 0);
    std::cout << "STRASSEN TEST " << (mismatch[0] ? "FAILED" : "PASSED") << "\n" << std::endl;
    // mismatch[1] = cnn_check(ptr_output[0], ptr_weight[1], ptr_output[1],
    //         ref_output[0], ref_weight[1], ref_output[1], 1);
    // std::cout << "CNN layer 1 TEST " << (mismatch[1] ? "FAILED" : "PASSED") << "\n" << std::endl;

    delete[] ref_inA;
    delete[] ref_inB;
    delete[] ref_output;
    // delete[] ref_weight[1];
    // delete[] ref_output[1];

    deallocate_mem(cl_obj, ptr_inA, 0);
    deallocate_mem(cl_obj, ptr_inB, 1);
    deallocate_mem(cl_obj, ptr_output, 2);
    // deallocate_mem(cl_obj, ptr_weight[1], 3);
    // deallocate_mem(cl_obj, ptr_output[1], 4);

    std::cout << "===== Reporting measured throughput ======" << std::endl;
    float timeusec=(end_time.tv_sec - start_time.tv_sec)*1e6 + (end_time.tv_usec - start_time.tv_usec);
    printf("Runtime = %0.1f (microsec) \n\n", timeusec);
    double num_operations[2] = {
            BATCH_SIZE * (double)2.0 * M_OFM(0) * R_OFM(0) * C_OFM(0) * N_IFM(0) * K_WTS * K_WTS,
            BATCH_SIZE * (double)2.0 * M_OFM(1) * R_OFM(1) * C_OFM(1) * N_IFM(1) * K_WTS * K_WTS
    };
    printf("# of operations = %.0f + %.0f \n", num_operations[0], num_operations[1]);
    printf("Throughput: %.5f GigaOP/sec\n",
            (double)1.0e-3 * (num_operations[0] + num_operations[1]) / timeusec);

    std::cout << "\n===== Exiting ======" << std::endl;
    return (mismatch[0]);
}
