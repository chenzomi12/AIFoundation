/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_halving_doubling.h"

namespace hccl {
ReduceScatterHalvingDoubling::ReduceScatterHalvingDoubling(const u32 blockSize,
                                                           const HcclDispatcher dispatcher,
                                                           const u64 reduceAttrBitMap,
                                                           const UserMemType hdInputMemType,
                                                           const UserMemType hdOutputMemType)
    : ExecutorBase(dispatcher), blockSize_(blockSize), reduceAttr_(reduceAttrBitMap),
      hdInputMemType_(hdInputMemType), hdOutputMemType_(hdOutputMemType)
{
}

ReduceScatterHalvingDoubling::~ReduceScatterHalvingDoubling()
{
}

u32 ReduceScatterHalvingDoubling::GetBlockStep(u32 blocksize) const
{
    // 求以2为底数的对数计算
    u32 step = 0;
    while ((blocksize >> (step + 1)) != 0) {
        step++;
    }

    return step;
}

HcclResult ReduceScatterHalvingDoubling::CalculateSlices(const u64 size, const u32 sliceNum,
                                                         std::vector<Slice> &slicesOut)
{
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[Calculate][Slices]calculate_slices failed"), HCCL_E_INTERNAL);

    // 不对size, count和slice_num做检查, 默认满足reduce-scatter的要求
    std::vector<Slice> slices(sliceNum);

    u64 sliceSize = size / sliceNum;

    for (u32 i = 0; i < sliceNum; i++) {
        slices[i].size = sliceSize;
        slices[i].offset = i * sliceSize;
    }

    slicesOut = std::move(slices);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHalvingDoubling::CalcStepSlices(const std::vector<Slice> &inputSlices,
    const u32 stepNum, const u32 rank, const SliceType type, std::vector<Slice> &slicesOut)
{
    std::vector<Slice> slice(stepNum);

    for (u32 step = 0; step < stepNum; step++) {
        // reduce-scatter操作, halving_bitmask从高往低循环, size倍减
        u32 halvingBitmask = (1 << (stepNum - step - 1));
        u32 peerRank = rank ^ halvingBitmask;

        // 计算tx_slice/rx_slice
        u32 sliceId = (type == SliceType::SLICE_TYPE_TX) ? \
            (peerRank & (~(halvingBitmask - 1))) : (rank & (~(halvingBitmask - 1)));

        slice[step].offset = inputSlices[sliceId].offset;
        HcclResult ret = Sum(inputSlices, sliceId, halvingBitmask, slice[step].size);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Calc][StepSlices]rank[%u] reduce scatter Halving Doubling sum error", rank), ret);
        HCCL_DEBUG("rank[%u] step[%u] type[%u] slice[%u].offset[%llu] slice[%u].size[%llu] ", \
                   rank, step, type, step, slice[step].offset, step, slice[step].size);
    }

    slicesOut = std::move(slice);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHalvingDoubling::RunSourceReducer(const LINK &link, const Slice &txSlice)
{
    HcclResult ret = link->RxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][SourceReducer]txSlice.size[%llu] rx ack run failed",
        txSlice.size), ret);

    DeviceMem txMem = inputMem_.range(txSlice.offset, txSlice.size);

    HCCL_DEBUG("tx_slice.offset[%llu],base_offset[%llu],tx_slice.size[%llu]",
        txSlice.offset, baseOffset_, txSlice.size);

    // 发送到对端的output
    CHK_RET(senderInfo_->run(link, txSlice.offset + baseOffset_, txMem, stream_));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHalvingDoubling::RunDestRducer(const LINK &link,
                                                       const HcclDispatcher dispatcher,
                                                       const Slice &rxSlice, const DstMemType reduceDst)
{
    DeviceMem rxMem = scratchMem_.range(rxSlice.offset, rxSlice.size);
    HCCL_DEBUG("rx_mem.offset[%llu],base_offset[%llu],rx_slice.size[%llu]",
        rxSlice.offset, baseOffset_, rxSlice.size);

    DeviceMem reduceSrcMem;
    DeviceMem reduceDstMem;

    if (DataUnitSize(dataType_) == 0) {
        HCCL_ERROR("[Run][DestRducer]DataUnitSize(data_type_) == 0, error");
        return HCCL_E_INTERNAL;
    }

    if (link->IsSpInlineReduce() && (INLINE_REDUCE_BITMASK & reduceAttr_)) {
        reduceSrcMem = inputMem_.range(rxSlice.offset, rxSlice.size);
    } else {
        reduceSrcMem = outputMem_.range(rxSlice.offset, rxSlice.size);
    }
    reduceDstMem = inputMem_.range(rxSlice.offset, rxSlice.size);

    HcclResult ret = reducerInfo_->run(dispatcher, link, rxSlice.offset + baseOffset_,
        reduceSrcMem, reduceDstMem, rxMem, stream_, DstMemType::RESULT_INPUT_MEM);

    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][DestRducer]offset[%llu] reduce_async failed",
        rxSlice.offset), ret);

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ReduceScatterHalvingDoubling][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }

    HCCL_INFO("ReduceScatterHalvingDoubling run: rank[%u] totalrank[%u] \
        inputMem[%p] outputMem[%p] count[%llu]", \
        rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);
    HcclResult ret = HCCL_SUCCESS;

    // 仅一个rank, 则直接input拷贝到output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return ret;
    }

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);

    bool bRetSize = (links.size() < rankSize);
    CHK_PRT_RET(bRetSize, HCCL_ERROR("[ReduceScatterHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is error",
        rank, links.size()), HCCL_E_INTERNAL);

    // 检查是否已对数据分片
    if (slices_.size() != rankSize) {
        CHK_RET(CalculateSlices(inputMem_.size(), rankSize, slices_));
    }

    // 计算每个step的数据size
    u32 stepNum = GetBlockStep(rankSize);
    CHK_RET(CalcStepSlices(slices_, stepNum, rank, SliceType::SLICE_TYPE_TX, txSlices_));

    CHK_RET(CalcStepSlices(slices_, stepNum, rank, SliceType::SLICE_TYPE_RX, rxSlices_));

    CHK_RET(RunReduceScatter(rank, stepNum, dispatcher_, links));

    HCCL_INFO("ReduceScatterHalvingDoubling finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}


HcclResult ReduceScatterHalvingDoubling::RunReduceScatter(const u32 rank, const u32 stepNum,
                                                          const HcclDispatcher dispatcher,
                                                          const std::vector<LINK> &links)
{
    HcclResult ret = HCCL_SUCCESS;

    for (u32 step = 0; step < stepNum; step++) {
        // reduce-scatter操作, peer_rank_bitmask从高往低循环
        u32 peerRankBitmask = 1 << (stepNum - step - 1);
        u32 peerRank = rank ^ peerRankBitmask;

        CHK_SMART_PTR_NULL(links[peerRank]);

        ret = links[peerRank]->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]rank[%u] tx ack to peerrank[%u] in step[%u] "\
            "run failed", rank, peerRank, step), ret);

        HCCL_DEBUG("rank[%u] send to peerrank[%u] in step[%u], silce.offset[%llu], slice.size[%llu]", \
                   rank, peerRank, step, txSlices_[step].offset, txSlices_[step].size);
        // 本rank作为reducer的发送侧的动作
        ret = RunSourceReducer(links[peerRank], txSlices_[step]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]rank[%u] to peerrank[%u] reducer_src_run "\
            "failed", rank, peerRank), ret);

        // 本rank作为reducer的接收侧的动作, 结果除最后一轮存放至output外, 均存放至input
        DstMemType reduceDst = (step == stepNum - 1) ? \
            DstMemType::RESULT_OUTPUT_MEM : DstMemType::RESULT_INPUT_MEM;

        HCCL_DEBUG("rank[%u] Reduce from peerrank[%u] in step[%u], silce.offset[%llu], slice.size[%llu]", \
                   rank, peerRank, step, rxSlices_[step].offset, rxSlices_[step].size);

        ret = RunDestRducer(links[peerRank], dispatcher, rxSlices_[step], reduceDst);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]rank[%u] to peerrank[%u] reducer_dst_run "\
            "failed", rank, peerRank), ret);
        ret = links[peerRank]->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]RxWaitDone failed"), ret);
        ret = links[peerRank]->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]TxWaitDone failed"), ret);
    }
    DeviceMem reduceSrcMem = inputMem_.range(rxSlices_[stepNum-1].offset, rxSlices_[stepNum-1].size);
    DeviceMem reduceDstMem = outputMem_.range(rxSlices_[stepNum-1].offset, rxSlices_[stepNum-1].size);
    ret = HcclD2DMemcpyAsync(dispatcher_, reduceDstMem, reduceSrcMem, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]HcclD2DMemcpyAsync failed"), ret);

    return ret;
}
}  // namespace hccl
