/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_ring.h"

namespace hccl {
BroadcastRing::BroadcastRing(const HcclDispatcher dispatcher)
    : ExecutorBase(dispatcher)
{
}

BroadcastRing::~BroadcastRing()
{
}

HcclResult BroadcastRing::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("BroadcastRing run: rank[%u] totalrank[%u] count[%llu]", \
              rank, rankSize, count_);

    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    // 获取ring algorithm所需的通信连接
    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    u32 ringNextRank = (rank + 1) % rankSize;

    if (links.size() < rankSize) {
        HCCL_ERROR("[BroadcastRing][RunAsync]rank[%u] linksize[%llu] is less than rank size", \
            rank, links.size());
        return HCCL_E_INTERNAL;
    }
    linkLeft_ = links[ringPrevRank];
    CHK_SMART_PTR_NULL(linkLeft_);

    linkRight_ = links[ringNextRank];
    CHK_SMART_PTR_NULL(linkRight_);

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[BroadcastRing][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    if (rank == root_) {
        CHK_PRT_RET(!inputMem_, HCCL_ERROR("[BroadcastRing][RunAsync]rank[%u] inputmem is null", rank), HCCL_E_PTR);
        scratch_ = DeviceMem::create(inputMem_.ptr(), inputMem_.size());
    } else {
        CHK_PRT_RET(!outputMem_, HCCL_ERROR("[BroadcastRing][RunAsync]rank[%u] outputmem is null", rank),
            HCCL_E_PTR);
        scratch_ = DeviceMem::create(outputMem_.ptr(), outputMem_.size());
    }
    HCCL_DEBUG("root[%u] scratch[%p] memsize[%llu]", \
        root_, scratch_.ptr(), scratch_.size());

    //  所有的数据平均到每个rank上
    u64 sizeAvg = ((count_ + rankSize - 1) / rankSize) * unitSize;

    u64 sizePerSlice = ExecutorBase::RoundUpWithDivisor(sizeAvg, HCCL_MIN_SLICE_ALIGN);
    u64 sizePerRound = 0;
    HCCL_DEBUG("bcast total count[%llu] sizeAverage[%llu] sizePerSlice after aligns[%llu]",
        count_, sizeAvg, sizePerSlice);
    CHK_RET(linkLeft_->TxAck(stream_));
    CHK_RET(linkRight_->RxAck(stream_));

    HcclResult ret = HCCL_SUCCESS;
    if (rank == root_) { // root节点，数据发送下一个节点
        DeviceMem localSrc;

        u64 sizeResidue = count_ * unitSize;
        for (u32 round = 0; round < rankSize; round++, sizeResidue -= sizePerRound) {  // 固定循环次数，避免子图复用出错
            sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
            localSrc = scratch_.range(count_ * unitSize - sizeResidue, sizePerRound);

            // 数据向下一个rank发送
            u64 dstOffset = count_ * unitSize - sizeResidue + baseOffset_;
            ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, dstOffset, localSrc.ptr(), sizePerRound, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BroadcastRing][RunAsync]root[%u] right link tx async srcmem "\
                "[%p] to offset[%llu] failed", rank, localSrc.ptr(), dstOffset), ret);
            HCCL_DEBUG("root rank[%u] send scratchmem[%p] to offset[%llu] sendsize[%llu]", rank, \
                       localSrc.ptr(), dstOffset, sizePerRound);

            // 等待后一节点同步信号
            CHK_RET(linkRight_->RxAck(stream_));
            ret = linkRight_->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BroadcastRing][RunAsync]TxWaitDone failed"), ret);
        }
    } else if (ringNextRank == root_) { // 最后一个节点，只接收数据
        DeviceMem localSrc;

        u64 sizeResidue = count_ * unitSize;
        for (u32 round = 0; round < rankSize; round++, sizeResidue -= sizePerRound) {  // 固定循环次数，避免子图复用出错
            sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
            localSrc = scratch_.range(count_ * unitSize - sizeResidue, sizePerRound);
            // 从前一节点接收数据
            u64 dstOffset = count_ * unitSize - sizeResidue + baseOffset_;
            ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, dstOffset, localSrc.ptr(), sizePerRound, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BroadcastRing][RunAsync]rank[%u] rx data from offset[%llu] "\
                "with localmem[%p] failed", rank, dstOffset, localSrc.ptr()),
                ret);
            HCCL_DEBUG("last rank[%u]  rx_sync from  range[%llu] with localmem[%p] size:[%llu] ", rank, \
                       dstOffset, localSrc.ptr(), sizePerRound);

            // 给前一节点发送同步
            ret = linkLeft_->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[BroadcastRing][RunAsync]rank[%u] left link tx ack failed", rank), ret);
            ret = linkLeft_->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BroadcastRing][RunAsync]RxWaitDone failed"), ret);
        }
    } else { // 非root节点或者尾节点，先接收来自前一节点的数据，再发送至下一结点
        DeviceMem localSrc;

        u64 sizeResidue = count_ * unitSize;
        for (u32 round = 0; round < rankSize; round++, sizeResidue -= sizePerRound) {  // 固定循环次数，避免子图复用出错
            sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
            // 需要从前一节点接收数据
            localSrc = scratch_.range(count_ * unitSize - sizeResidue, sizePerRound);
            u64 dstOffset = count_ * unitSize - sizeResidue + baseOffset_;
            ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, dstOffset, localSrc.ptr(), sizePerRound, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, \
                HCCL_ERROR("[BroadcastRing][RunAsync]rank[%u] left link rx sync from offset[%llu] with "\
                "localmem[%p] failed", rank, dstOffset, localSrc.ptr()), ret);
            HCCL_DEBUG("rank[%u] rx_sync from  range[%u] with localmem[%p] size[%llu] ", rank, \
                dstOffset, localSrc.ptr(), sizePerRound);

            // 给前一节点发送同步
            CHK_RET(linkLeft_->TxAck(stream_));
            ret = linkLeft_->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BroadcastRing][RunAsync]RxWaitDone failed"), ret);

            // 数据向下一个节点发送
            CHK_RET(linkRight_->TxAsync(UserMemType::OUTPUT_MEM, dstOffset, localSrc.ptr(), sizePerRound, stream_));

            HCCL_DEBUG("rank[%u] tx_sync from localmem[%p] to offset[%llu] size[%llu] ", \
                rank, localSrc.ptr(), dstOffset, sizePerRound);
            // 等待后一节点同步信号
            CHK_RET(linkRight_->RxAck(stream_));
            ret = linkRight_->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BroadcastRing][RunAsync]TxWaitDone failed"), ret);
        }
    }
    CHK_RET(linkRight_->TxDataSignal(stream_));

    CHK_RET(linkLeft_->RxDataSignal(stream_));
    HCCL_INFO("BroadcastRing finished: rank[%u] end count[%llu]", rank, count_);
    return HCCL_SUCCESS;
}
}  // namespace hccl
