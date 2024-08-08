/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_halving_doubling.h"

namespace hccl {
AllGatherHalvingDoubling::AllGatherHalvingDoubling(u32 blockSize,
    const HcclDispatcher dispatcher, UserMemType hdInputMemType, UserMemType hdOutputMemType)
    : ExecutorBase(dispatcher), blockSize_(blockSize), interRank_(0), interRankSize_(0),
      hdInputMemType_(hdInputMemType), hdOutputMemType_(hdOutputMemType)
{
}

AllGatherHalvingDoubling::~AllGatherHalvingDoubling()
{
}

u32 AllGatherHalvingDoubling::Log2(u32 antilogarithm) const
{
    // 求以2为底数的对数计算
    u32 logarithm = 0;
    while ((antilogarithm >> (logarithm + 1)) != 0) {
        logarithm++;
    }

    return logarithm;
}

HcclResult AllGatherHalvingDoubling::CalculateSlices(const std::vector<Slice> &inputSlices, u32 stepNum,
                                                     u32 rank, SliceType type, std::vector<Slice> &sliceOut)
{
    std::vector<Slice> slice(stepNum);

    for (u32 step = 0; step < stepNum; step++) {
        // all-gather操作, halving_bitmask从低往高循环, size倍增
        u32 halvingBitmask = (1 << step);
        u32 peerRank = rank ^ halvingBitmask;

        // 计算tx_slice/rx_slice
        u32 sliceId = (type == SliceType::SLICE_TYPE_RX) ? \
            (peerRank & (~(halvingBitmask - 1))) : (rank & (~(halvingBitmask - 1)));

        slice[step].offset = inputSlices[sliceId].offset;
        CHK_RET(Sum(inputSlices, sliceId, halvingBitmask, slice[step].size));

        HCCL_DEBUG("Slice Info: rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", \
                   rank, step, slice[step].offset, step, slice[step].size);
    }

    sliceOut = std::move(slice);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalvingDoubling::CalculateSlices(u64 size, u32 sliceNum, std::vector<Slice> &sliceOut) const
{
    // 不对size, count和slice_num做检查, 默认满足all-gather的要求
    std::vector<Slice> slices(sliceNum);

    for (u32 i = 0; i < sliceNum; i++) {
        slices[i].size = size;
        slices[i].offset = i * size;
        HCCL_DEBUG("Slice Info: slices[%u].offset=%llu, slices[%u].size=%llu", \
                   i, slices[i].offset, i, slices[i].size);
    }

    sliceOut = std::move(slices);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalvingDoubling::Rx(const LINK &link, const Slice &rxSlice)
{
    // 目前不考虑Halving-Doubling算法引入inlne-reduce
    DeviceMem rxMem = outputMem_.range(rxSlice.offset, rxSlice.size);
    // 接收数据到output
    HcclResult ret = link->RxAsync(hdOutputMemType_, rxSlice.offset + baseOffset_,
        rxMem.ptr(), rxSlice.size, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalvingDoubling][Rx]rank[%u] rx_async with rxMem[%p] Failed",
        interRank_, rxMem.ptr()), ret);
    ret = link->DataReceivedAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalvingDoubling][Rx]rank[%u] data_received_ack Failed",
        interRank_), ret);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalvingDoubling::Tx(const LINK &link, const Slice &txSlice)
{
    // 目前不考虑Halving-Doubling算法引入inlne-reduce
    HcclResult ret = link->RxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalvingDoubling][Tx]rank[%u] txSlice.size[%llu]Link rx_ack "\
        "Failed", interRank_, txSlice.size), ret);
    DeviceMem txMem = inputMem_.range(txSlice.offset, txSlice.size);
    // input的数据发送
    ret = link->TxAsync(hdOutputMemType_, txSlice.offset + baseOffset_,
                        txMem.ptr(), txSlice.size, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalvingDoubling][Tx]rank[%u] tx_async txMem[%p] Failed", \
        interRank_, txMem.ptr()), ret);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    HCCL_INFO("AllGatherHD Run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    HcclResult ret = HCCL_SUCCESS;
    interRank_ = rank;
    interRankSize_ = rankSize;

    // 检查rank, rank_size合法性
    // 仅一个rank, 则直接input拷贝到output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }

        return ret;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllGatherHalvingDoubling][RunAsync]rank[%u] link size error", rank);
        return HCCL_E_INTERNAL;
    }
    // 检查是否已对数据分片
    if (slices_.size() != rankSize) {
        CHK_RET(CalculateSlices(inputMem_.size(), rankSize, slices_));
    }

    // 计算每个step的数据size
    u32 stepNum = Log2(blockSize_);
    CHK_RET(CalculateSlices(slices_, stepNum, rank, SliceType::SLICE_TYPE_TX, txSlices_));

    CHK_RET(CalculateSlices(slices_, stepNum, rank, SliceType::SLICE_TYPE_RX, rxSlices_));

    CHK_RET(RunAllGather(rank, stepNum, links));

    HCCL_INFO("AllGatherHD finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}


HcclResult AllGatherHalvingDoubling::RunAllGather(u32 rank, u32 stepNum,
                                                  const std::vector<LINK> &links)
{
    HcclResult ret = HCCL_SUCCESS;

    if (rxSlices_.size() < stepNum || txSlices_.size() < stepNum) {
        HCCL_ERROR("[Run][AllGather]rank[%u] rxslice size or tx slice size error", rank);
        return HCCL_E_INTERNAL;
    }

    for (u32 step = 0; step < stepNum; step++) {
        // all-gather操作, peer_rank_bitmask从低往高循环
        u32 peerRankBitmask = (1 << step);
        u32 peerRank = rank ^ peerRankBitmask;
        CHK_SMART_PTR_NULL(links[peerRank]);

        ret = links[peerRank]->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]rank[%u] step[%u] tx ack to dstrank[%u] Failed", \
            rank, step, peerRank), ret);

        // 本rank的发送侧的动作
        HCCL_DEBUG("Rank[%u] send to PeerRank[%u] in Round[%u], silce.offset[%llu], slice.size[%llu]", \
            rank, peerRank, step, txSlices_[step].offset, txSlices_[step].size);
        ret = Tx(links[peerRank], txSlices_[step]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]Rank[%u] end to PeerRank[%u] run tx failed ", \
            rank, peerRank), ret);
        HCCL_DEBUG("Rank[%u]receive from PeerRank[%u] in Round[%u], silce.offset[%llu], slice.size[%llu]", \
                   rank, peerRank, step, rxSlices_[step].offset, rxSlices_[step].size);
        // 本rank的接收侧的动作
        ret = Rx(links[peerRank], rxSlices_[step]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]Rank[%u] rx from PeerRank[%u] failed", \
            rank, peerRank), ret);
        ret = links[peerRank]->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]RxWaitDone failed"), ret);
        ret = links[peerRank]->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]TxWaitDone failed"), ret);
    }
    return ret;
}
}  // namespace hccl
