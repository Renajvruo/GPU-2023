/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 * In this sample:
 * Length of x / y / z is 8*2048.
 * Num of vector core used in sample is 8.
 * Length for each core to compute is 2048.
 * Tiles for each core is 8 which means we add 2048/8=256 elements in one loop.
 *
 * This is just a tile strategy for demonstration, in fact we can compute at most 128*255
 * elements in one loop for b16 type.
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // seperate to 2 parts, due to double buffer

class KernelSinh {
public:
    __aicore__ inline KernelSinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z)
    {
        //  YOUR CODE HERE !
        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half*)z + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(calcBuf,TILE_LENGTH * sizeof(half));
        //pipe.InitBuffer(mid, TILE_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        constexpr int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //  YOUR CODE HERE !
        // alloc tensor from queue memory
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //  YOUR CODE HERE !
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        LocalTensor<half> tLocal = calcBuf.Get<half>();
        // LocalTensor<half> tLocal = mid.AllocTensor<half>();
        // call Add instr for computation
        // 一会儿回来改
        Exp(zLocal, xLocal, TILE_LENGTH);
        Reciprocal(xLocal, zLocal, TILE_LENGTH);
        Sub(tLocal, zLocal, xLocal, TILE_LENGTH);
        half scalar = 0.5;
        Muls(zLocal, tLocal, scalar, TILE_LENGTH);
        // Add(zLocal, xLocal, yLocal, TILE_LENGTH);

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<half>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);       
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //  YOUR CODE HERE !
        // deque output tensor from VECOUT queue
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    //  YOUR CODE HERE !
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, zGm;

    TBuf<TPosition::VECIN> calcBuf;
    // TBuf<TPosition::VECCALC> mid;
};

extern "C" __global__ __aicore__ void SinhCustom(GM_ADDR x, GM_ADDR z)
{
    KernelSinh op;
    op.Init(x, z);
    op.Process();
}

#ifndef __CCE_KT_TEST__
void SinhCustom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* z)
{
    SinhCustom<<<blockDim, l2ctrl, stream>>>(x, z);
}
#endif