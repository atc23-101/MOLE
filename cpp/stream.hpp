/* This file is adapted from FastMoE

*/

#ifndef _STREAM_HPP_
#define _STREAM_HPP_

#include "helper_cuda.h"
#include <unordered_map>
#include <cassert>
#include <thread>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <vector>
#include <torch/extension.h>

/*  Modification from FastMoE  */
c10::cuda::CUDAStream tran_stream(int device);
c10::cuda::CUDAStream comp_stream(int device);
c10::cuda::CUDAStream comm_stream(int device);
c10::cuda::CUDAStream tranback_stream(int device);
c10::cuda::CUDAStream default_stream(int device);

class _MOLEStreamManager
{
public:
    int device;
    cublasHandle_t *handles;
    cudaStream_t *streams;
    std::vector<c10::cuda::CUDAStream> c10streams;
    c10::cuda::CUDAStream defaultstream;
    bool use_default;

    char ncclgood;
    ncclComm_t ncclcomm;

    bool multi_stream;

public:
    _MOLEStreamManager(int device_) : device(device_), use_default(false), defaultstream(c10::cuda::getCurrentCUDAStream())
    {
        this->setup(device);
        // defaultstream = comp_stream(device_);
    }

    void setup(int);
    void sync(int = 0);
    void destroy();
    c10::cuda::CUDAStream cudastream(int idx)
    {
        return this->c10streams[idx];
    }
    cudaStream_t stream(size_t = 0);
    cublasHandle_t handle(size_t = 0);

    ~_MOLEStreamManager()
    {
        this->destroy();
    }
};

_MOLEStreamManager *_getCudaStreamManager(const int device);

void _default_wait_tran();

void _default_wait_comm();

void _default_wait_comp();

void _comm_wait_default();

void _comp_wait_default();

void _tran_back_wait_tran();

void _synchronize_tran();

void _disable_multi_stream(int device);

void _enable_multi_stream(int device);

#endif