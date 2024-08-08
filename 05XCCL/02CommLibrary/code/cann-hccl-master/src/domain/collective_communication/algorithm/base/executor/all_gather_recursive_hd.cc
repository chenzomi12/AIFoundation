/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_recursive_hd.h"
#include "all_gather_halving_doubling_pub.h"

namespace hccl {
AllGatherRecursiveHalvingDoubling::AllGatherRecursiveHalvingDoubling(const HcclDispatcher dispatcher)
    : RecursiveHalvingDoublingBase(dispatcher)
{
}

AllGatherRecursiveHalvingDoubling::~AllGatherRecursiveHalvingDoubling()
{
}

// 服务器间allreduce的入口函数
HcclResult AllGatherRecursiveHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize,
                                                       const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("AllGatherRecursiveHalvingDoubling run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]",
        rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    HcclResult ret = HCCL_SUCCESS;

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return ret;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllGatherRecursiveHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    ret = CalcPartOneSizeAndBlockSize(rankSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherRecursiveHalvingDoubling][RunAsync]Calculate Par1Size[%u] "\
        "And BlockSize[%u] Failed! rankSize[%u]", part1Size_, blockSize_, rankSize), ret);

    ret = CalculateSlices(dataBytes_, rankSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherRecursiveHalvingDoubling][RunAsync]Calculate slices failed, "\
            "dataBytes[%llu], rankSize[%u]", dataBytes_, rankSize), ret);

    CHK_RET(GatherInPartOneToEven(rank, links));

    CHK_RET(AllGatherInBlock(rank, rankSize, links));

    CHK_RET(GatherInPartOneToOdd(rank, links));

    HCCL_INFO("AllGatherRecursiveHalvingDoubling finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}


HcclResult AllGatherRecursiveHalvingDoubling::CalculateSlices(u64 dataBytes, const u32 rankSize) const
{
    slices_.resize(blockSize_);
    u64 bytesPerSlice = dataBytes;
    u64 totalBytes = dataBytes * rankSize;
    u64 bytesLeft = totalBytes;
    u32 i = 0;
    while (bytesLeft > 0 && i < part1Size_ / 2) { // 除2计算part1在做完操作后block内slice数
        slices_[i].size = 2 * bytesPerSlice < bytesLeft ? 2 * bytesPerSlice : bytesLeft; // 乘2表示slice为part2两倍
        slices_[i].offset = totalBytes - bytesLeft;
        bytesLeft -= slices_[i].size;
        i++;
    }

    while (bytesLeft > 0) {
        slices_[i].size = bytesPerSlice < bytesLeft ? bytesPerSlice : bytesLeft;
        slices_[i].offset = totalBytes - bytesLeft;
        bytesLeft -= slices_[i].size;
        i++;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherRecursiveHalvingDoubling::GatherInPartOneToEven(u32 rank, const std::vector<LINK> &links)
{
    if (rank < part1Size_ && rank % 2 == 0) {  // 模2判断奇偶性，从下一个rank的output收数据到output
        u32 peerRank = rank + 1;               // 加1计算下一个rank号
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);

            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed",
                    rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed",
                    rank, peerRank), ret);
            DeviceMem gatherOutputMem = outputMem_.range(dataBytes_ * rank, dataBytes_);
            //  接收数据到本端的 output
            HCCL_DEBUG(
                "rank[%u] outputMem[%p] recieve from PeerRank[%u] outputMem, Offset[%llu], Size[%llu]",
                rank, gatherOutputMem.ptr(), peerRank, baseOffset_ + dataBytes_ * rank,
                gatherOutputMem.size());

            ret = ExecuteRxSync(links[peerRank], UserMemType::OUTPUT_MEM, dataBytes_ * rank, gatherOutputMem.ptr(),
                dataBytes_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] rx sync from PeerRank[%u] "\
                "failed", rank, peerRank), ret);
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToEven]RxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) {  // 模2判断奇偶性，向上一个rank发送数据
        u32 peerRank = rank - 1;                      // 减1计算上一个rank号
        //  发送到对端的output
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx ack from peerank[%u] failed",
                    rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            //  等待对端可以接收数据
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] rx ack from peerank[%u] failed",
                    rank, peerRank), ret);
            //  设置gather的发送内存范围
            DeviceMem gatherOutputMem = outputMem_.range(dataBytes_ * rank, dataBytes_);
            //  发送数据到对端的 output
            HCCL_DEBUG("rank[%u] outputMem[%p] sends to PeerRank[%u] outputMem, Offset[%llu], Size[%llu]",
                rank, gatherOutputMem.ptr(), peerRank, baseOffset_ + dataBytes_ * rank,
                gatherOutputMem.size());

            ret = ExecuteTxSync(links[peerRank], UserMemType::OUTPUT_MEM, dataBytes_ * rank, gatherOutputMem.ptr(),
                dataBytes_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToEven]rank[%u] tx sync to PeerRank[%u] failed",
                    rank, peerRank), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToEven]TxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherRecursiveHalvingDoubling::GatherInPartOneToOdd(u32 rank, const std::vector<LINK> &links)
{
    if (rank < part1Size_ && rank % 2 == 0) {  // 模2判断奇偶性，向下一个rank发送数据
        u32 peerRank = rank + 1;               // 加1计算下一个rank号
        //  发送到对端的output
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] tx ack from peerank[%u] failed",
                    rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            //  等待对端可以接收数据
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] rx ack from peerank[%u] failed",
                    rank, peerRank), ret);

            HCCL_DEBUG("rank[%u] outputMem[%p] sends to PeerRank[%u] outputMem, Offset[%llu], Size[%llu]",
                       rank, outputMem_.ptr(), peerRank, baseOffset_, outputMem_.size());
            ret = ExecuteTxSync(links[peerRank], UserMemType::OUTPUT_MEM, baseOffset_, outputMem_.ptr(),
                outputMem_.size(), stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] tx sync to PeerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToOdd]TxWaitDone failed"), ret);
        }
    } else if (rank < part1Size_ && rank % 2 == 1) {  // 模2判断奇偶性，从上一个rank的output收数据到output
        u32 peerRank = rank - 1;                      // 减1计算上一个rank号
        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            //  知会对端本人可以接收数据
            HcclResult ret = links[peerRank]->TxAck(stream_);
            //  等待对端可以接收数据
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] tx ack from peerank[%u] failed",
                    rank, peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] rx ack from peerank[%u] failed",
                    rank, peerRank), ret);
            //  接收数据到本端的 output
            HCCL_DEBUG("rank[%u] outputMem[%p] recieve from PeerRank[%u] outputMem, Offset[%llu], "\
                "Size[%llu]", rank, outputMem_.ptr(), peerRank, baseOffset_, outputMem_.size());
            ret = ExecuteRxSync(links[peerRank], UserMemType::OUTPUT_MEM, baseOffset_, outputMem_.ptr(),
                outputMem_.size(), stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][InPartOneToOdd]rank[%u] rx sync from PeerRank[%u] failed", rank, peerRank), ret);
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InPartOneToOdd]RxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherRecursiveHalvingDoubling::AllGatherInBlock(u32 rank, u32 rankSize, const std::vector<LINK> &links)
{
    u32 rankInBlock = 0;
    if (rank < part1Size_ && (rank % 2) == 1) {        // 模2余1代表当前rank在part1的奇数位置上，不参与block内的计算
        return HCCL_SUCCESS;
    } else if (rank < part1Size_ && (rank % 2) == 0) { // 模2余0代表当前rank在part1的偶数位置上，参与block内的计算
        rankInBlock = rank / 2;                        // 除2计算出在block内的rank号
    } else {
        rankInBlock = rank - part1Size_ / 2;           // rank在part2中，用原始rank减part1除2，计算出在block内的rank号
    }

    AllGatherHalvingDoubling executor(blockSize_, dispatcher_, UserMemType::OUTPUT_MEM, UserMemType::OUTPUT_MEM);

    CHK_RET(executor.Prepare(outputMem_, outputMem_, count_, dataType_, stream_,
        reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(executor.RegisterProfiler(
        profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    std::vector<LINK> subLinks;
    CHK_RET(BuildSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(subLinks.size() == 0,
        HCCL_ERROR("[AllGatherRecursiveHalvingDoubling][AllGatherInBlock]rank[%u] BuildSubLinks failed",
            rank), HCCL_E_PARA);

    CHK_RET(executor.RunAsync(rankInBlock, blockSize_, subLinks));
    return HCCL_SUCCESS;
}
}  // namespace hccl
