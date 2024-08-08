/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "reduce_scatter_ring_pub.h"
#include "all_gather_ring_pub.h"
#include "all_reduce_ring.h"

namespace hccl {
AllReduceRing::AllReduceRing(const HcclDispatcher dispatcher,
    const u64 reduceAttrBitMap)
    : ExecutorBase(dispatcher), reduceAttr_(reduceAttrBitMap)
{
}

AllReduceRing::~AllReduceRing()
{
}

// ringallreduce算法的函数入口
HcclResult AllReduceRing::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = PrepareRunAsync(rank, rankSize, links);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] count[%llu] "\
        "failed in PrepareRunAsync step", rank, count_), ret);
    CHK_PRT_RET(rankSize == 1, HCCL_INFO("[AllReduceRing][RunAsync] rankSize[%u], do nothing.",
        rankSize), HCCL_SUCCESS);

    // 先执行reducescater
    ret = RunReduceScatter(rank, rankSize, links);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in reducescater "\
        "step", rank, count_), ret);

    // 再执行allgather
    ret = RunAllGather(rank, rankSize, links);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in allgather "\
        "step", rank, count_), ret);

    HCCL_INFO("AllReduceRing finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRing::RunAsyncStaged(const u32 rank, const u32 rankSize, const std::vector<LINK> &links,
    RunStage stage)
{
    CHK_PRT_RET(rankSize == 1 && stage != RunStage::RUN_PREPARE,
        HCCL_INFO("[AllReduceRing][RunAsyncStaged] rankSize[%u], stage[%d], do nothing.",
        rankSize, stage), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    switch (stage) {
        case RunStage::RUN_PREPARE:
            ret = PrepareRunAsync(rank, rankSize, links);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceRing][RunAsyncStaged]rank[%u] count[%llu] failed in PrepareRunAsync step",
                rank, count_), ret);
            break;
        case RunStage::RUN_REDUCE_SCATTER:
            // 先执行reducescater
            ret = RunReduceScatter(rank, rankSize, links, true);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceRing][RunAsyncStaged]rank[%u] count[%llu] "\
                "failed in reducescater step", rank, count_), ret);
            break;
        case RunStage::RUN_ALLGATHER:
            // 再执行allgather
            ret = RunAllGather(rank, rankSize, links);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceRing][RunAsyncStaged]rank[%u] count[%llu] "\
                "failed in allgather step", rank, count_), ret);
            break;
        default:
            HCCL_ERROR("[AllReduceRing][RunAsyncStaged]stage[%d]is not support", stage);
            return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("AllReduceRing RunAsyncStaged stage[%d] finished: rank[%u] ranksize[%u]", stage, rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRing::PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO("AllReduceRing run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] memcpy async failed", rank), ret);
        }

        return ret;
    }
    // 计算reducescatter 阶段每个rank结果上的offset和size
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        u64 totalSize = count_ * SIZE_TABLE[dataType_];
        u64 sliceSizeCalculated = (totalSize + (rankSize - 1)) / rankSize;
        u64 sliceSizeAligned = ExecutorBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);
        u64 residueSize = totalSize;

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
            slices_[i].offset = totalSize - residueSize;
            residueSize -= slices_[i].size;
        }

        if (CheckDebugLogLevel()) {
            for (size_t j = 0; j < slices_.size(); j++) {
                HCCL_DEBUG("rank[%u] slice[%u]: offset[%llu] size[%llu]", rank, j, slices_[j].offset, slices_[j].size);
            }
        }
    }
    HCCL_INFO("AllReduceRing PrepareRunAsync finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}


HcclResult AllReduceRing::RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK> &links, bool needBarrier)
{
    if (links.size() < rankSize) {
        HCCL_ERROR("[AllReduceRing][RunReduceScatter]rank[%u] linkSize[%llu] is less than rankSize[%u]",
            rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    // 调用reducescattering算法
    ReduceScatterRing executor(dispatcher_, reduceAttr_);
    HCCL_INFO("rank[%u] executor reducescattering inputMem[%p] outputMem[%p] mem_size[%llu] "\
        "count[%llu] planeID:[%d]", \
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);
    if (!needBarrier) {
        executor.CloseBarrier();
    }
    CHK_RET(executor.Prepare(inputMem_, inputMem_, outputMem_, count_, dataType_, stream_,
        reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(executor.RegisterProfiler(
        profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return executor.RunAsync(rank, rankSize, links);
}

HcclResult AllReduceRing::RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK> &links)
{
    AllGatherRing executor(dispatcher_);
    HCCL_INFO("rank[%u] executor allgathering inputMem[%p] outputMem[%p] mem_size[%llu] "\
        "count[%llu] planeID:[%d]", rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(),
        count_, profilerInput_.planeID);
    // 判断是否关闭allgather的barrier
    if (!barrierSwitchOn_) {
        executor.CloseBarrier();
    }
    // 调用allgatherring的算法执行
    CHK_RET(executor.Prepare(inputMem_, outputMem_, outputMem_, count_, dataType_, stream_,
        reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(executor.RegisterProfiler(
        profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return executor.RunAsync(rank, rankSize, links);
}
}  // namespace hccl
