/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh.h"

namespace hccl {
ReduceScatterMesh::ReduceScatterMesh(const HcclDispatcher dispatcher,
    const u64 reduceAttrBitMap, const u32 streamIndex)
    : ExecutorBase(dispatcher), interRank_(0), interRankSize_(0), reduceAttr_(reduceAttrBitMap),
      streamIndex_(streamIndex)
{
}

ReduceScatterMesh::~ReduceScatterMesh()
{
}

HcclResult ReduceScatterMesh::RunSourceReducer(const LINK &link, const Slice &txSlice, const Slice &dstSlice)
{
    // 发送inputmem
    DeviceMem srcMem = inputMem_.range(txSlice.offset, txSlice.size);
    HCCL_INFO("rank[%u] inputSlice range[%llu], size[%llu]", \
        interRank_, txSlice.offset, txSlice.size);
    CHK_RET(senderInfo_->run(link, baseOffset_ + dstSlice.offset, srcMem, stream_));

    return HCCL_SUCCESS;
}
HcclResult ReduceScatterMesh::RunDestRducer(const LINK &link, const Slice &rxSlice, const Slice &dstSlice)
{
    // 使用scratchmem接收数据，并同inputmem数据做reduce
    DeviceMem dstMem = inputMem_.range(rxSlice.offset, rxSlice.size);
    DeviceMem srcMem = scratchMem_.range(dstSlice.offset, dstSlice.size);
    HCCL_INFO("rank[%u] rxSlice offset[%llu], size[%llu] dstSlice offset[%llu] size[%llu] ",
        interRank_, rxSlice.offset, rxSlice.size, dstSlice.offset, dstSlice.size);

    HcclResult ret = reducerInfo_->run(dispatcher_, link, baseOffset_ + rxSlice.offset,
        dstMem, dstMem, srcMem, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][DestRducer]rank[%u] reducer info run failed", \
        interRank_), ret);

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMesh::RunReduceScatter(const std::vector<LINK> &links,
    const std::vector<Slice> &inputSlices, const std::vector<Slice> &scratchSlices)
{
    std::vector<u32> txRankOpOrder;
    std::vector<u32> rxRankOpOrder;
    // 计算默认的每轮接收的源rank和发送的目的rank
    for (u32 round = 1; round < interRankSize_; round++) {
        u32 srcRank = ForwardRank(interRank_, interRankSize_, round);
        u32 dstRank = BackwardRank(interRank_, interRankSize_, round);
        HCCL_INFO("<multiDie>RunReduceScatter:srcRank[%u] dstRank[%u]", srcRank, dstRank);
        rxRankOpOrder.push_back(srcRank);
        txRankOpOrder.push_back(dstRank);
    }

    HcclResult ret = HCCL_SUCCESS;
    for (u32 round = 1; round < interRankSize_; round++) {
        // 不同的stream依次轮训默认的顺序数组
        u32 orderIndex = (round + streamIndex_ - 1) % (interRankSize_ - 1);
        u32 srcRank = rxRankOpOrder[orderIndex];
        s32 dstRank = txRankOpOrder[orderIndex];
        CHK_SMART_PTR_NULL(links[srcRank]);
        HCCL_INFO("rank[%u] will tx_ack to rank[%u] ", interRank_, srcRank);
        ret = links[srcRank]->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][ReduceScatter]rank[%u] tx ack to rank[%u] failed", interRank_, srcRank), ret);
        CHK_SMART_PTR_NULL(links[dstRank]);
        HCCL_INFO("rank[%u] will rx_ack from rank[%d]", interRank_, dstRank);

        ret = links[dstRank]->RxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][ReduceScatter]rank[%u] rx ack from rank[%d] failed", interRank_, dstRank), ret);
        HCCL_INFO("rank:%u round[%u] send to rank:[%d], inputSlices offset[%llu]"
            "size[%llu] scratchSlice offset[%llu] size[%llu] ",
            interRank_, round, dstRank, inputSlices[dstRank].offset, inputSlices[dstRank].size,
            scratchSlices[dstRank].offset, scratchSlices[dstRank].size);
        // 发送数据
        ret = RunSourceReducer(links[dstRank], inputSlices[dstRank], scratchSlices[dstRank]);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][ReduceScatter]rank:%u round[%u] reducer src run failed", interRank_, round), ret);
        HCCL_INFO("rank[%u] round[%u] rx from rank[%u], inSlicesoffset[%llu] size[%llu] \
                scratchSlices offset[%llu] size[%llu]",
            interRank_, round, srcRank, inputSlices[interRank_].offset, inputSlices[interRank_].size,
            scratchSlices[interRank_].offset, scratchSlices[interRank_].size);

        ret = RunDestRducer(links[srcRank], inputSlices[interRank_], scratchSlices[interRank_]);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][ReduceScatter]rank[%u] round[%u] reducer dst run failed", interRank_, round), ret);

        ret = links[srcRank]->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]RxWaitDone failed"), ret);
        ret = links[dstRank]->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]TxWaitDone failed"), ret);
    }
    if (barrierSwitchOn_) {
        for (u32 round = 1; round < interRankSize_; round++) {
            u32 orderIndex = (round + streamIndex_ - 1) % (interRankSize_ - 1);
            u32 srcRank = rxRankOpOrder[orderIndex];
            s32 dstRank = txRankOpOrder[orderIndex];

            ret = ExecuteBarrier(links[srcRank], links[dstRank]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][ReduceScatter]rank[%u] run reduce scatter executor barrier "\
                    "failed. srcRank:%u dstRank:%d", interRank_, srcRank, dstRank), ret);
        }
    }

    return HCCL_SUCCESS;
}

// reducescatter的入口函数
HcclResult ReduceScatterMesh::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ReduceScatterMesh][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO("ReduceScatterMesh run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    interRank_ = rank;
    interRankSize_ = rankSize;

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    if (interRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[ReduceScatterMesh][RunAsync]rank[%u] linksize error", rank);
        return HCCL_E_INTERNAL;
    }

    if (streamIndex_ >= interRankSize_ - 1) {
        HCCL_ERROR("[ReduceScatterMesh][RunAsync]rank[%u] stream index[%u] is out of range when ranksize[%u]",
            rank, streamIndex_, rankSize);
        return HCCL_E_INTERNAL;
    }

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[ReduceScatterMesh][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> scratchSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        scratchSlices.resize(rankSize);

        // 生成std::vector<Slice> slices_
        u64 sliceSize = count_ * unitSize;

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = (i * sliceSize);
            scratchSlices[i].size = sliceSize;
            scratchSlices[i].offset = (inputMem_.size() > outputMem_.size()) ? 0 : (i * sliceSize);
            HCCL_DEBUG("rank[%u], slices[%u].offset=[%llu] slices[%u].size=[%llu]", \
                       rank, i, slices_[i].offset, i, slices_[i].size);
        }
    }
    // 运行reduce-scatter, mesh算法
    CHK_RET(RunReduceScatter(links, slices_, scratchSlices));

    if (inputMem_ != outputMem_) {
        DeviceMem src = inputMem_.range(slices_[interRank_].offset, slices_[interRank_].size);
        HCCL_DEBUG("rank[%u] copy result from to output[%p] offset[%llu] size[%llu] ", \
                   interRank_, outputMem_.ptr(), slices_[interRank_].offset, \
                   slices_[interRank_].size);
        HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, src, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterMesh][RunAsync]rank[%u] memcpy async from mem[%p] "\
            "to ouputmem[%p] failed", rank, src.ptr(), outputMem_.ptr()), ret);
    }

    HCCL_INFO("ReduceScatterMesh finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}
}  // namespace hccl
