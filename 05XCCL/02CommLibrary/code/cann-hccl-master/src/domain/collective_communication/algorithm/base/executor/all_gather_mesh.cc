/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_mesh.h"
#include "externalinput_pub.h"

namespace hccl {
AllGatherMesh::AllGatherMesh(const HcclDispatcher dispatcher, std::vector<Stream> &meshStreams,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 interRank,
    u32 interRankSize, u32 userRank)
    : ExecutorBase(dispatcher),
      meshStreams_(meshStreams),
      meshSignal_(meshSignal),
      meshSignalAux_(meshSignalAux),
      interRank_(interRank),
      interRankSize_(interRankSize),
      userRank_(userRank)
{}

AllGatherMesh::~AllGatherMesh() {}

HcclResult AllGatherMesh::Tx(const LINK &link, const Slice &txSlice, const Slice &dstSlice, Stream stream)
{
    DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);

    HCCL_DEBUG("rank[%u] tx srcMem[%p] output's offset[%llu] size[%llu]  to dstrank's offset[%llu]", interRank_,
        srcMem.ptr(), txSlice.offset, txSlice.size, dstSlice.offset);
    HcclResult ret =
        link->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + dstSlice.offset, srcMem.ptr(), txSlice.size, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherMesh][Tx]rank[%u] tx srcMem[%p] tx_async Failed", interRank_, srcMem.ptr()), ret);
    return HCCL_SUCCESS;
}

HcclResult AllGatherMesh::Rx(const LINK &link, const Slice &srcSlice, const Slice &rxSlice, Stream stream)
{
    DeviceMem rcvMem = outputMem_.range(rxSlice.offset, rxSlice.size);

    HCCL_DEBUG("rank[%u] rx rcvMem[%p] output's offset[%llu] size[%llu] rcv datat from srcrank's"
        "offset[%llu] ",
        interRank_, rcvMem.ptr(), rxSlice.offset, rxSlice.size, srcSlice.offset);
    HcclResult ret =
        link->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + srcSlice.offset, rcvMem.ptr(), rxSlice.size, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherMesh][Tx]rank[%u] rcvMem[%p] rx_async Failed", interRank_, rcvMem.ptr()), ret);
    return HCCL_SUCCESS;
}

HcclResult AllGatherMesh::RunAllGatherHighPerf(const std::vector<LINK> &links, const std::vector<Slice> &outputSlices,
    const std::vector<Slice> &inputSlices)
{
    Stream subStream;
    for (u32 round = 1; round < interRankSize_; round++) {
        u32 dstRank = BackwardRank(interRank_, interRankSize_, round);

        subStream = (round == interRankSize_ - 1) ? stream_ : meshStreams_[round - 1];

        s32 streamID = 0;
        HcclResult rtRet = hrtGetStreamId(subStream.ptr(), streamID);
        CHK_PRT_RET(rtRet != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]Call hrtGetStreamId error[%d]", rtRet),
            HCCL_E_INTERNAL);
        profilerInput_.streamID = streamID;
        profilerInput_.planeID = round - 1;
        profilerInput_.step = HCCL_EXEC_STEP_NOT_SET;

        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
        void *srcMemPtr = nullptr;
        // 从对端的input内存拿数据，input==output也没有关系
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &srcMemPtr));
        DeviceMem srcDevMem(static_cast<s8 *>(srcMemPtr) + baseOffset_ + inputSlices[dstRank].offset,
            inputSlices[dstRank].size);
        DeviceMem dstDevMem = outputMem_.range(outputSlices[dstRank].offset, outputSlices[dstRank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstDevMem, srcDevMem, subStream,
            links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        CHK_RET(links[dstRank]->TxDataSignal(subStream)); // Post remoteSendReadyNotify
        CHK_RET(links[dstRank]->RxDataSignal(subStream)); // Wait localSendReadyNotify
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherMesh::RunAllGather(const std::vector<LINK> &links, const std::vector<Slice> &outputSlices,
    const std::vector<Slice> &inputSlices)
{
    Stream subStream;
    HcclResult ret = HCCL_SUCCESS;
    for (u32 round = 1; round < interRankSize_; round++) {
        u32 dstRank = BackwardRank(interRank_, interRankSize_, round);

        subStream = (round == interRankSize_ - 1) ? stream_ : meshStreams_[round - 1];

        s32 streamID = 0;
        HcclResult rtRet = hrtGetStreamId(subStream.ptr(), streamID);
        CHK_PRT_RET(rtRet != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]Call hrtGetStreamId error[%d]", rtRet),
            HCCL_E_INTERNAL);
        profilerInput_.streamID = streamID;
        profilerInput_.planeID = round - 1;
        profilerInput_.step = HCCL_EXEC_STEP_NOT_SET;

        CHK_SMART_PTR_NULL(links[dstRank]);
        HCCL_DEBUG("rank[%u] round[%u] tx ack to srcRank[%u] ", interRank_, round, dstRank);
        ret = links[dstRank]->TxAck(subStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]rank[%u] tx ack to rank[%u] Failed", interRank_, dstRank), ret);
        CHK_SMART_PTR_NULL(links[dstRank]);

        HCCL_DEBUG("rank[%u] round[%u] rx ack from Rank[%u] ", interRank_, round, dstRank);
        ret = links[dstRank]->RxAck(subStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]rank[%u] rx_ack from rank[%u] Failed", interRank_, dstRank), ret);

        HCCL_DEBUG("rank[%u] tx to rank[%u] inputslice offset[%llu] size[%llu] outputSlices "
            "offset[%llu] size[%llu] ",
            interRank_, dstRank, inputSlices[interRank_].offset, inputSlices[interRank_].size,
            outputSlices[interRank_].offset, outputSlices[interRank_].size);

        ret = Tx(links[dstRank], inputSlices[interRank_], outputSlices[interRank_], subStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]rank[%u] tx to rank[%u] failed", interRank_, dstRank), ret);
        HCCL_DEBUG("rank[%u] will rcv from rank[%u] inputSlices offset[%llu] size[%llu] outputSlices"
            "offset[%llu] size[%llu]",
            interRank_, dstRank, inputSlices[dstRank].offset, inputSlices[dstRank].size, outputSlices[dstRank].offset,
            outputSlices[dstRank].size);

        ret = Rx(links[dstRank], inputSlices[dstRank], outputSlices[dstRank], subStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]rank[%u] rx from srcRank[%u] run failed", interRank_, dstRank), ret);
        ret = ExecuteBarrier(links[dstRank], subStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]destrank[%u] all gather mesh run executor barrier "
            "failed",
            dstRank),
            ret);
        ret = links[dstRank]->RxWaitDone(subStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]RxWaitDone failed"), ret);
        ret = links[dstRank]->TxWaitDone(subStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGather]TxWaitDone failed"), ret);
    }
    return HCCL_SUCCESS;
}

// allgather的入口函数
HcclResult AllGatherMesh::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("AllGatherMesh run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    interRank_ = rank;
    interRankSize_ = rankSize;

    if (interRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            HCCL_DEBUG("rank[%u] mem copy async from input to output", rank);
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return ret;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllGatherMesh][RunAsync]rank[%u] linksize error", rank);
        return HCCL_E_INTERNAL;
    }
    u32 subStreamSize = interRankSize_ - 2; // 子流大小等于ranksize-2
    if (meshStreams_.size() < subStreamSize || meshSignal_.size() < subStreamSize ||
        meshSignalAux_.size() < subStreamSize) {
        HCCL_ERROR("[AllGatherMesh][RunAsync]AllGatherMesh stream size error: "
            "rank[%u] totalrank:%u substreamsize[%llu] signalsize[%llu], signal_aux size[%llu]",
            rank, rankSize, meshStreams_.size(), meshSignal_.size(), meshSignalAux_.size());
        return HCCL_E_PARA;
    }

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherMesh][RunAsync]rank[%u] Unit Data Size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(interRankSize_);
        inputSlices.resize(rankSize);

        // 生成std::vector<Slice> slices_
        u64 sliceSize = count_ * unitSize;

        for (u32 i = 0; i < interRankSize_; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = (i * sliceSize);

            inputSlices[i].size = sliceSize;
            inputSlices[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
            HCCL_DEBUG("rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", rank, i, slices_[i].offset, i,
                slices_[i].size);
        }
    }

    for (u32 i = 0; i < interRankSize_; i++) {
        HCCL_DEBUG("[AllGatherMesh][Outputslice]: size[%llu] offset[%llu]   inputslice: size[%llu]  offset[%llu]",
            slices_[i].size, slices_[i].offset, inputSlices[i].size, inputSlices[i].offset);
    }

    if (inputMem_ != outputMem_) {
        DeviceMem dst = outputMem_.range(slices_[rank].offset, slices_[rank].size);
        DeviceMem src = inputMem_.range(inputSlices[rank].offset, inputSlices[rank].size);

        HCCL_INFO("inputMem != outputMem: rank[%u] copy src[%p] offset[%llu] size[%llu] to dst[%p] offset[%llu] "
            "size[%llu]",
            rank, src.ptr(), inputSlices[rank].offset, inputSlices[rank].size, dst.ptr(), slices_[rank].offset,
            slices_[rank].size);

        // 拷贝到自身rank的output_mem
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize-2: stream num
        HCCL_DEBUG("rank[%u] streamindex[%u] wait signalaux[%p]",
            rank, streamIndex, meshSignalAux_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_, meshSignalAux_[streamIndex],
            profilerInput_.stage));

        HCCL_DEBUG("rank[%u] siganl_aux index[%u] signal record signalaux[%p] ", rank, streamIndex,
            meshSignalAux_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux_[streamIndex],
            profilerInput_.stage));
    }
    if (GetExternalInputHcclHighPerfEnable() != 0) {
        CHK_RET(RunAllGatherHighPerf(links, slices_, inputSlices));
    } else {
        CHK_RET(RunAllGather(links, slices_, inputSlices));
    }

    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize - 2 stream num
        HCCL_DEBUG("rank[%u] streamindex[%u] wait signal[%p] ",
            rank, streamIndex, meshSignal_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal_[streamIndex], profilerInput_.stage));

        HCCL_DEBUG("rank[%u] streamindex[%u] record signal[%p]", rank, streamIndex, meshStreams_[streamIndex].ptr());
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal_[streamIndex],
            profilerInput_.stage));
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    HCCL_INFO("AllGatherMesh finished: rank[%u", rank);
    return HCCL_SUCCESS;
}
} // namespace hccl
