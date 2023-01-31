/* This file is adapted from FastMoE
*/

#include "stream.hpp"
#include <ATen/cuda/CUDAEvent.h>
#define SMGR_N_STREAMS 16

cudaStream_t _MOLEStreamManager::stream(size_t idx)
{
    if (this->use_default)
    {
        return c10::cuda::getCurrentCUDAStream().stream();
    }
    return this->streams[idx % SMGR_N_STREAMS];
}

cublasHandle_t _MOLEStreamManager::handle(size_t idx)
{
    if (this->use_default)
    {
        return at::cuda::getCurrentCUDABlasHandle();
    }
    return this->handles[idx % SMGR_N_STREAMS];
}

void _MOLEStreamManager::sync(int idx)
{
    if (this->use_default)
    {
        return;
    }
    for (int i = 0; i < idx && i < SMGR_N_STREAMS; ++i)
    {
        cudaStreamSynchronize(streams[i]);
    }
}

void _MOLEStreamManager::setup(const int device)
{
    this->ncclgood = 0;

    this->device = device;
    checkCudaErrors(cudaSetDevice(device));
    streams = new cudaStream_t[SMGR_N_STREAMS];
    handles = new cublasHandle_t[SMGR_N_STREAMS];
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i)
    {
        checkCudaErrors(cudaStreamCreate(streams + i));
        checkCudaErrors(cublasCreate(handles + i));
        cublasSetStream(handles[i], streams[i]);
        this->c10streams.push_back(c10::cuda::getStreamFromExternal(streams[i], device));
    }

    this->multi_stream = true;
}

void _MOLEStreamManager::destroy()
{
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i)
    {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
        checkCudaErrors(cublasDestroy(handles[i]));
    }
    delete[] streams;
    delete[] handles;
}

std::vector<_MOLEStreamManager *> smgrs;

_MOLEStreamManager *_getCudaStreamManager(const int device)
{

    if (smgrs.size() == 0)
    {

        auto smgr = new _MOLEStreamManager(device);
        smgrs.push_back(smgr);
        return smgr;
    }
    assert(device == smgrs[0].device);
    return smgrs[0];
}

c10::cuda::CUDAStream default_stream(int device)
{
    _MOLEStreamManager *mgr = _getCudaStreamManager(device);
    if (mgr->multi_stream)
    {
        return comp_stream(device);
    }
    else
    {
        return mgr->defaultstream;
    }
}
c10::cuda::CUDAStream tran_stream(int device)
{
    _MOLEStreamManager *mgr = _getCudaStreamManager(device);
    if (mgr->multi_stream)
    {
        return mgr->cudastream(1);
    }
    else
    {
        return mgr->defaultstream;
    }
}

c10::cuda::CUDAStream comp_stream(int device)
{
    _MOLEStreamManager *mgr = _getCudaStreamManager(device);
    if (mgr->multi_stream)
    {
        return mgr->cudastream(2);
    }
    else
    {
        return mgr->defaultstream;
    }
}
c10::cuda::CUDAStream comm_stream(int device)
{
    _MOLEStreamManager *mgr = _getCudaStreamManager(device);
    if (mgr->multi_stream)
    {
        return mgr->cudastream(3);
    }
    else
    {
        return mgr->defaultstream;
    }
}
c10::cuda::CUDAStream tranback_stream(int device)
{
    _MOLEStreamManager *mgr = _getCudaStreamManager(device);
    if (mgr->multi_stream)
    {
        return mgr->cudastream(4);
    }
    else
    {
        return mgr->defaultstream;
    }
}

uint64_t _get_stream_cdata(int device, int idx)
{
    return _getCudaStreamManager(device)->c10streams[idx].pack();
}

int _get_device()
{
    return smgrs[0]->device;
}

void _default_wait_tran()
{
    int device = _get_device();
    at::cuda::CUDAEvent tran_event;
    tran_event.record(tran_stream(device));
    AT_CUDA_CHECK(cudaStreamWaitEvent(default_stream(device), tran_event, 0));
}

void _default_wait_comm()
{
    int device = _get_device();
    at::cuda::CUDAEvent comm_event;
    comm_event.record(comm_stream(device));
    AT_CUDA_CHECK(cudaStreamWaitEvent(default_stream(device), comm_event, 0));
}

void _default_wait_comp()
{
    int device = _get_device();
    at::cuda::CUDAEvent comp_event;
    comp_event.record(comp_stream(device));
    AT_CUDA_CHECK(cudaStreamWaitEvent(default_stream(device), comp_event, 0));
}

void _comp_wait_default()
{
    int device = _get_device();
    at::cuda::CUDAEvent default_event;
    default_event.record(default_stream(device));
    AT_CUDA_CHECK(cudaStreamWaitEvent(comp_stream(device), default_event, 0));
}
void _tran_back_wait_tran()
{
    int device = _get_device();
    at::cuda::CUDAEvent tran_event;
    tran_event.record(tran_stream(device));
    AT_CUDA_CHECK(cudaStreamWaitEvent(tranback_stream(device), tran_event, 0));
}
void _comm_wait_default()
{
    int device = _get_device();
    at::cuda::CUDAEvent default_event;
    default_event.record(default_stream(device));
    AT_CUDA_CHECK(cudaStreamWaitEvent(comm_stream(device), default_event, 0));
}

void _synchronize_tran()
{
    AT_CUDA_CHECK(cudaStreamSynchronize(tran_stream(_get_device())));
}

void _disable_multi_stream(int device)
{
    _getCudaStreamManager(device)->multi_stream = false;
}

void _enable_multi_stream(int device)
{
    _getCudaStreamManager(device)->multi_stream = true;
}