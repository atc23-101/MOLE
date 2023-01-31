#include "define.hpp"
#include "stream.hpp"

/*   FROM FASTER MOE  */
#define NCCL_SAFE_CALL(__fn__)                                                              \
    {                                                                                       \
        auto __res__ = __fn__;                                                              \
        if (__res__ != ncclSuccess)                                                         \
        {                                                                                   \
            fprintf(stderr, "NCCL Error at %s:%d value %d\n", __FILE__, __LINE__, __res__); \
            exit(-1);                                                                       \
        }                                                                                   \
    }
c10::cuda::CUDAStream c_data2Stream(uint64_t cdata)
{
    return c10::cuda::CUDAStream::unpack(cdata);
}

class HackNCCLGroup : public c10d::ProcessGroupNCCL
{
public:
    ncclComm_t getcomm()
    {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0)
        {
            ncclGetUniqueId(&ncclID);
        }
#if defined(TORCH_VERSION_MAJOR) && ((TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        broadcastUniqueNCCLID(&ncclID,
                              false,
                              "mole_nccl_comm",
                              rank);
#elif defined(TORCH_VERSION_MAJOR) && ((TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                              c10d::OpType::SEND,
                              "mole_nccl_comm",
                              rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        return comm;
    }
};

void _ensure_nccl(c10d::ProcessGroupNCCL &p, int device)
{
    auto smgr = _getCudaStreamManager(device);
    if (smgr->ncclgood)
    {
        return;
    }
    HackNCCLGroup *h = (HackNCCLGroup *)(void *)&p;
    smgr->ncclcomm = h->getcomm();
    if (smgr->ncclcomm != 0)
    {
        smgr->ncclgood = 1;
    }
    else
    {
        std::cerr << "Nccl initialization failed\n";
    }
}

void _MOLEAll2Allfp32(std::vector<at::Tensor> src, std::vector<at::Tensor> tgt, int device, int world_size, int stream_id, uint64_t s, std::string group, int world_rank)
{
    int rank;
    auto ctx = _getCudaStreamManager(device);
    NCCL_SAFE_CALL(ncclCommUserRank(ctx->ncclcomm, &rank));

    // verify that world_rank param equals to rank get from ncclcomm
    if (~world_rank)
    {
        assert(rank == world_rank);
    }

    auto stream = (stream_id == -1) ? c_data2Stream(s).stream() : ctx->stream(stream_id);
    NCCL_SAFE_CALL(ncclGroupStart());

    for (int i = 0; i < world_size; i++)
    {
        if (i == rank)
        {
            tgt[i].copy_(src[i]);
        }
        else
        {

            ncclSend(src[i].data_ptr(), src[i].numel(),
                     ncclFloat32, i, ctx->ncclcomm, stream);

            ncclRecv(tgt[i].data_ptr(), tgt[i].numel(),
                     ncclFloat32, i, ctx->ncclcomm, stream);
        }
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
}

/*   FROM FASTER MOE  */
