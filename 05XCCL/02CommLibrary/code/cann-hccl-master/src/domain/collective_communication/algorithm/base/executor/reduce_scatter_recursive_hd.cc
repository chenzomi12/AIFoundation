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
#include "reduce_scatter_recursive_hd.h"

namespace hccl {
ReduceScatterRecursiveHalvingDoubling::ReduceScatterRecursiveHalvingDoubling(
    const HcclDispatcher dispatcher, const u64 reduceAttrBitMap)
    : RecursiveHalvingDoublingBase(dispatcher), reduceAttr(reduceAttrBitMap)
{
}

ReduceScatterRecursiveHalvingDoubling::~ReduceScatterRecursiveHalvingDoubling()
{
}

// reducescatter recursiveHD 入口函数
HcclResult ReduceScatterRecursiveHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("run: rank[%u] totalrank[%u] \
        inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize, \
              inputMem_.ptr(), outputMem_.ptr(), count_);
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ReduceScatterRecursiveHalvingDoubling][RunAsync]rank[%u] run_async inputmem or outputmem is null",
            rank);
        return HCCL_E_PTR;
    }

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
    CHK_PRT_RET(bRetSize, HCCL_ERROR("[ReduceScatterRecursiveHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is "\
        "error", rank, links.size()), HCCL_E_INTERNAL);

    ret = CalcPartOneSizeAndBlockSize(rankSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterRecursiveHalvingDoubling][RunAsync]calculate part1size[%u] "\
        "and blocksize[%u] Failed! rankSize[%u]", part1Size_, blockSize_, rankSize), ret);

    HCCL_DEBUG("rank[%u] calculate par1size[%u] blocksize[%u] ranksize[%u]", rank, part1Size_, blockSize_, rankSize);

    CHK_RET(ReduceInPartOne(rank, links));

    CHK_RET(CalculateSlices(dataBytes_, rankSize));

    CHK_RET(ReduceScatterInBlock(rank, rankSize, links));

    CHK_RET(ScatterInPartOne(rank, rankSize, links));

    HCCL_INFO("ReduceScatterRecursiveHalvingDoubling finished: rank[%u], finished", rank);
    return HCCL_SUCCESS;
}


HcclResult ReduceScatterRecursiveHalvingDoubling::CalculateSlices(u64 dataBytes, const u32 rankSize) const
{
    CHK_PRT_RET((blockSize_ == 0), HCCL_ERROR("[Calculate][Slices]calculate_slices para error"), HCCL_E_INTERNAL);

    slices_.resize(blockSize_);
    u64 bytesPerSlice = dataBytes / rankSize; // input大小 / server数 = 服务器内rank数*count (4p mesh以4*count为粒度)
    u32 i = 0;
    u32 halfPart1Size = (part1Size_ / 2); // 除2计算一半part1的大小

    /* 先给属于part1的block rank分配slice。每个rank有两份数据 */
    while (i < halfPart1Size) {
        slices_[i].size = 2 * bytesPerSlice; // 乘2计算2倍数据大小
        slices_[i].offset = i * 2 * bytesPerSlice; // 乘2计算2倍数据大小
        i++;
    }

    /* 再给剩余的block rank分配slice。每个rank有一份数据 */
    while (i < blockSize_) {
        slices_[i].size = bytesPerSlice;
        slices_[i].offset = (i * bytesPerSlice) + (halfPart1Size * bytesPerSlice);
        i++;
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRecursiveHalvingDoubling::ReduceInPartOne(u32 rank, const std::vector<LINK> &links)
{
    if (rank < part1Size_ && rank % 2 == 0) {  // 模2判断奇偶性，rank属于第一部分，并且为偶数rank
        u32 peerRank = rank + 1;
        HCCL_DEBUG("rank[%u] outputmem receives from peerrank[%u] inputmem, offset[%llu], size[%llu]", \
                   rank, peerRank, baseOffset_, scratchMem_.size());
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            // 接收数据到本端的input
            HCCL_DEBUG("send mem[%p] size[%llu] to peerank[%u]", \
                scratchMem_.ptr(), scratchMem_.size(), peerRank);
            ret = links[peerRank]->TxAsync(UserMemType::INPUT_MEM, baseOffset_, scratchMem_.ptr(), 0, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOneToEven]TxAsync: tx async size[%llu] "\
                "failed", 0), ret);
            CHK_RET(reducerInfo_->run(dispatcher_, links[peerRank], baseOffset_,
                inputMem_, inputMem_, scratchMem_, stream_));
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]RxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { //  向上一个rank的input发数据 2
        u32 peerRank = rank - 1;
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            //  发送到对端的input
            HCCL_DEBUG("rank[%u] sends inputMem[%p] to peerrank[%u] offset[%llu], size[%llu]", \
                rank, inputMem_.ptr(), peerRank, baseOffset_, inputMem_.size());
            ret = senderInfo_->run(links[peerRank], baseOffset_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]tx sync to peerank[%u] failed",
                peerRank), ret);
            ret = links[peerRank]->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_, inputMem_.ptr(), 0, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: rx async size[%llu] failed", 0), ret);
            ret = links[peerRank]->DataReceivedAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: data received ack failed"), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]TxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}


HcclResult ReduceScatterRecursiveHalvingDoubling::ReduceScatterInBlock(u32 rank, u32 rankSize,
    const std::vector<LINK> &links)
{
    u32 rankInBlock = 0;
    if (rank < part1Size_ && (rank % 2) == 1) { // rank号对2求余，rank为奇数
        return HCCL_SUCCESS;
    } else if (rank < part1Size_ && (rank % 2) == 0) { // rank对2求余，rank为偶数
        rankInBlock = rank / 2; // 直接除以2即为本rank的在block内的排序
    } else {
        rankInBlock = rank - part1Size_ / 2; // 通过rank减去part1除2的大小即不处于第一部分的block内rank号
    }

    ReduceScatterHalvingDoubling executor(blockSize_, dispatcher_, reduceAttr,
        UserMemType::INPUT_MEM, UserMemType::OUTPUT_MEM);
    CHK_RET(executor.Prepare(inputMem_, inputMem_, scratchMem_, count_, dataType_, stream_,
        reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(executor.RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    std::vector<LINK> subLinks;
    CHK_RET(BuildSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(subLinks.size() == 0,
        HCCL_ERROR("[ReduceScatterRecursiveHalvingDoubling][ReduceScatterInBlock]rank[%u] "\
            "build sub links failed", rank), HCCL_E_PARA);
    CHK_RET(executor.RunAsync(rankInBlock, blockSize_, subLinks));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRecursiveHalvingDoubling::ScatterInPartOne(u32 rank, u32 rankSize,
    const std::vector<LINK> &links)
{
    u32 bytesPerData = DataUnitSize(dataType_);
    u64 dataBytes = count_ * bytesPerData;
    u64 bytesPerSlice = dataBytes / rankSize;

    if (rank < part1Size_ && rank % 2 == 0) {  // 模2计算奇偶性，偶数rank把自己第二份数据给下一个奇数rank
        u32 peerRank = rank + 1;
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank,  peerRank), ret);
            //  发送到对端的input
            HCCL_DEBUG("rank[%u] sends inputmem[%p] to peerrank[%u] Offset[%llu], Size[%llu]", \
                rank, inputMem_.ptr(), peerRank, baseOffset_, inputMem_.size());

            u64 offset = peerRank * bytesPerSlice; // 计算对端rank的slice偏移
            void *srcAddr = reinterpret_cast<s8 *>(inputMem_.ptr()) + offset;
            ret = ExecuteTxSync(links[peerRank], UserMemType::INPUT_MEM, offset, srcAddr, bytesPerSlice, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Scatter][InPartOne]tx sync to peerank[%u] failed",
                peerRank), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]TxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) { // 模2计算奇偶性，奇数rank接收偶数rank发过来下半份的数据
        u32 peerRank = rank - 1;
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Scatter][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed", rank, peerRank), ret);
            //  接收数据到本端的 inputMem_
            HCCL_DEBUG("rx mem[%p] size[%llu] from peerank[%u]", inputMem_.ptr(), inputMem_.size(), peerRank);

            u64 offset = rank * bytesPerSlice; // 本rank slice偏移
            void *dstAddr = reinterpret_cast<s8 *>(inputMem_.ptr()) + offset;
            ret = ExecuteRxSync(links[peerRank], UserMemType::INPUT_MEM, offset, dstAddr, bytesPerSlice, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Scatter][InPartOne]rx sync from peerank[%u] failed",
                peerRank), ret);
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]RxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}
}  // namespace hccl
