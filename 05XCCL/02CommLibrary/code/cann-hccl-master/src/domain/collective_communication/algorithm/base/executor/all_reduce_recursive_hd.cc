/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "reduce_scatter_halving_doubling_pub.h"
#include "all_gather_halving_doubling_pub.h"
#include "all_reduce_recursive_hd.h"

namespace hccl {
AllReduceRecursiveHalvingDoubling::AllReduceRecursiveHalvingDoubling(
    const HcclDispatcher dispatcher, const u64 reduceAttrBitMap)
    : RecursiveHalvingDoublingBase(dispatcher), reduceAttr(reduceAttrBitMap)
{
}

AllReduceRecursiveHalvingDoubling::~AllReduceRecursiveHalvingDoubling()
{
}

// 算法的主入口
HcclResult AllReduceRecursiveHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize,
                                                       const std::vector<LINK> &links)
{
    CHK_RET(PrepareRunAsync(rank, rankSize, links));
    CHK_PRT_RET(rankSize == 1, HCCL_INFO("[AllReduceRecursiveHalvingDoubling][RunAsync] "\
        "rankSize[%u], do nothing.", rankSize), HCCL_SUCCESS);

    CHK_RET(ReduceInPartOne(rank, links));

    CHK_RET(ReduceScatterInBlock(rank, rankSize, links));

    CHK_RET(AllGatherInBlock(rank, rankSize, links));

    CHK_RET(GatherInPartOne(rank, links));

    HCCL_INFO("AllReduceRecursiveHalvingDoubling finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::RunAsyncStaged(const u32 rank, const u32 rankSize,
    const std::vector<LINK> &links, RunStage stage)
{
    CHK_PRT_RET(rankSize == 1 && stage != RunStage::RUN_PREPARE,
        HCCL_INFO("[AllReduceRecursiveHalvingDoubling][RunAsyncStaged] rankSize[%u], stage[%d], do nothing.",
        rankSize, stage), HCCL_SUCCESS);
    switch (stage) {
        case RunStage::RUN_PREPARE:
            CHK_RET(PrepareRunAsync(rank, rankSize, links));
            break;
        case RunStage::RUN_REDUCE_SCATTER:
            // 先执行reducescater
            CHK_RET(ReduceInPartOne(rank, links));
            CHK_RET(ReduceScatterInBlock(rank, rankSize, links));
            break;
        case RunStage::RUN_ALLGATHER:
            // 再执行allgather
            CHK_RET(AllGatherInBlock(rank, rankSize, links));
            CHK_RET(GatherInPartOne(rank, links));
            break;
        default:
            HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][RunAsyncStaged]stage[%d]is not support", stage);
            return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("AllReduceRecursiveHalvingDoubling RunAsyncStaged stage[%d] finished: rank[%u] ranksize[%u]",
        stage, rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::PrepareRunAsync(const u32 rank, const u32 rankSize,
    const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][RunAsync]rank[%u] run_async inputmem or outputmem is null",
            rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO("AllReduceRecursiveHalvingDoubling run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]",
        rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    HcclResult ret = HCCL_SUCCESS;

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return ret;
    }

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr));
    CHK_SMART_PTR_NULL(reducerInfo_);

    bool bRetSize = (links.size() < rankSize);
    CHK_PRT_RET(bRetSize, HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is "\
        "error", rank, links.size()), HCCL_E_INTERNAL);

    CHK_RET(CalcPartOneSizeAndBlockSize(rankSize));

    u32 bytesPerData = SIZE_TABLE[dataType_];
    u64 dataBytes = count_ * bytesPerData;
    CHK_RET(CalculateSlices(dataBytes));
    HCCL_INFO("AllReduceRecursiveHalvingDoubling PrepareRunAsync finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::ReduceInPartOne(u32 rank, const std::vector<LINK> &links)
{
    // 本rank属于第一部分，并且是2的整数倍
    if (rank < part1Size_ && rank % 2 == 0) {  // 1.从下一个rank接收数据到output，2. reduce到本rank的input
        u32 peerRank = rank + 1;
        HCCL_DEBUG("rank[%u] outputMem receives from PeerRank[%u] inputMem, Offset[%llu], Size[%llu]", \
                   rank, peerRank, baseOffset_, outputMem_.size());

        if (peerRank < links.size()) {
            const LINK &link = links[peerRank];
            CHK_SMART_PTR_NULL(link);

            HcclResult ret = link->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = link->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            //  接收数据到本端的 output
            HCCL_DEBUG("send mem[%p] size[%llu] to peerank[%u]", outputMem_.ptr(), outputMem_.size(), peerRank);
            ret = link->TxAsync(UserMemType::INPUT_MEM, baseOffset_, outputMem_.ptr(), 0, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOneToEven]TxAsync: tx async size[%llu] "\
                "failed", 0), ret);
            CHK_RET(reducerInfo_->run(dispatcher_, link, baseOffset_,
                outputMem_, inputMem_, outputMem_, stream_));
            ret = link->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]RxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { //  向上一个rank的output发数据 2
        u32 peerRank = rank - 1;

        if (peerRank < links.size()) {
            const LINK &link = links[peerRank];
            CHK_SMART_PTR_NULL(link);
            HcclResult ret = link->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = link->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            //  发送到对端的output
            HCCL_DEBUG("rank[%u] sends inputMem[%p] to PeerRank[%u] Offset[%llu], Size[%llu]", \
                rank, inputMem_.ptr(), peerRank, baseOffset_, inputMem_.size());
            ret = senderInfo_->run(link, baseOffset_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]tx sync to peerank[%u] failed",
                peerRank), ret);
            ret = link->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_, inputMem_.ptr(), 0, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: rx async size[%llu] failed", 0), ret);
            ret = link->DataReceivedAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: data received ack failed"), ret);
            ret = link->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]TxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}


HcclResult AllReduceRecursiveHalvingDoubling::ReduceScatterInBlock(u32 rank, u32 rankSize,
    const std::vector<LINK> &links)
{
    u32 rankInBlock = 0;
    if (rank < part1Size_ && (rank % 2) == 1) {     // 模2判断奇偶性，本rank处于第一部分，并且为奇数rank
        return HCCL_SUCCESS;
    } else if (rank < part1Size_ && (rank % 2) == 0) {     // 模2判断奇偶性，本rank 处于第一部分，并且为偶数rank
        rankInBlock = rank / 2;                            // 除2计算block内的rank值
    } else {           // 本rank不属于第一部分
        rankInBlock = rank - part1Size_ / 2;               // 除2计算block内的part1的范围
    }
    // 直接调用block的reducscatterhd算法
    ReduceScatterHalvingDoubling executor(blockSize_, dispatcher_, reduceAttr,
        UserMemType::INPUT_MEM, UserMemType::OUTPUT_MEM);
    CHK_RET(executor.Prepare(inputMem_, outputMem_, outputMem_, count_, dataType_, stream_,
        reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(executor.RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step,
        stream_));

    // 重新建立reducscatterscatter需要的链接
    std::vector<LINK> subLinks;
    CHK_RET(BuildSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(subLinks.size() == 0,
        HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][ReduceScatterInBlock]rank[%u] BuildSubLinks "\
            "failed", rank), HCCL_E_PARA);
    CHK_RET(executor.RunAsync(rankInBlock, blockSize_, subLinks));

    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::AllGatherInBlock(u32 rank, u32 rankSize,
                                                               const std::vector<LINK> &links)
{
    u32 rankInBlock = 0;
    if (rank < part1Size_ && (rank % 2) == 1) {    // 模2判断奇偶性，本rank 处于第一部分，并且为奇数rank
        return HCCL_SUCCESS;
    } else if (rank < part1Size_ && (rank % 2) == 0) { // 模2判断奇偶性，本rank 处于第一部分，并且为偶数rank
        rankInBlock = rank / 2;                        // 在block内的rank为实际rank除以2
    } else {
        rankInBlock = rank - part1Size_ / 2;           // 除2计算block内的part1的范围
    }
    // 直接调用block的allgatherhd算法
    AllGatherHalvingDoubling executor(blockSize_, dispatcher_, UserMemType::OUTPUT_MEM, UserMemType::OUTPUT_MEM);
    CHK_RET(executor.Prepare(outputMem_, outputMem_, count_, dataType_, stream_,
        reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(executor.RegisterProfiler(
        profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    // 重新建立allgather需要的链接
    std::vector<LINK> subLinks;
    CHK_RET(BuildSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(subLinks.size() == 0,
        HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][AllGatherInBlock]rank[%u] build sub "\
            "links failed", rank), HCCL_E_PARA);

    CHK_RET(executor.RunAsync(rankInBlock, blockSize_, subLinks));

    return HCCL_SUCCESS;
}

HcclResult AllReduceRecursiveHalvingDoubling::GatherInPartOne(u32 rank, const std::vector<LINK> &links)
{
    if (rank < part1Size_ && rank % 2 == 0) {  // 模2判断奇偶性，本rank 处于第一部分，并且为偶数rank
        u32 peerRank = rank + 1;
        //  发送到对端的output
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            HCCL_DEBUG("rank[%u] outputMem[%p] sends to peerrank[%u] outputmem, offset[%llu], size[%llu]",
                       rank, outputMem_.ptr(), peerRank, baseOffset_, outputMem_.size());
            ret = ExecuteTxSync(links[peerRank], UserMemType::OUTPUT_MEM, baseOffset_, outputMem_.ptr(),
                outputMem_.size(), stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][GatherInPartOne]rank[%u] tx "\
                    "sync to PeerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][GatherInPartOne]TxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) {  // 模2判断奇偶性，本rank 处于第一部分，并且为奇数rank
        u32 peerRank = rank - 1;
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            // 等待对端可以接收数据
            HCCL_DEBUG("rank[%u] outputMem[%p] recieve from PeerRank[%u] outputMem, Offset[%llu], "\
                "Size[%llu]", rank, outputMem_.ptr(), peerRank, baseOffset_, outputMem_.size());
            ret = ExecuteRxSync(links[peerRank], UserMemType::OUTPUT_MEM, baseOffset_, outputMem_.ptr(),
                outputMem_.size(), stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][GatherInPartOne]rank[%u] rx "\
                    "sync from PeerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllReduceRecursiveHalvingDoubling][GatherInPartOne]RxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}
}  // namespace hccl
