/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh_atomic.h"
#include "externalinput_pub.h"

namespace hccl {
using namespace std;

ReduceScatterMeshAtomic::ReduceScatterMeshAtomic(const HcclDispatcher dispatcher,
    const u64 reduceAttrBitMap, std::vector<Stream> &meshStreams,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 userRank)
    : ExecutorBase(dispatcher),
      reduceAttr_(reduceAttrBitMap),
      localRank_(0),
      localRankSize_(0),
      userRank_(userRank),
      meshStreams_(meshStreams),
      meshSignal_(meshSignal),
      meshSignalAux_(meshSignalAux)
{}

ReduceScatterMeshAtomic::~ReduceScatterMeshAtomic() {}

HcclResult ReduceScatterMeshAtomic::RunReduceScatterHighPerf(const std::vector<LINK> &links)
{
    // 拼接所有stream
    HCCL_INFO("[ReduceScatterMeshAtomic]RunReduceScatterHighPerf");
    vector<Stream> streamVct;
    streamVct.reserve(localRankSize_ - 1); // 有ranksize-1个对端，每个对端对应一条stream
    streamVct.push_back(stream_);          // 增加主stream
    streamVct.insert(streamVct.end(), meshStreams_.begin(), meshStreams_.end()); // 增加从stream

    // 每个stream只负责一个对端的交互
    for (u32 streamIndex = 0; streamIndex < localRankSize_ - 1; streamIndex++) {
        u32 remoteRank = (streamIndex + localRank_ + 1) % localRankSize_;
        const LINK &dstLink = links[remoteRank];
        Stream &stream = streamVct[streamIndex];
        Slice &rxSlice = slices_[localRank_];

        CHK_RET(dstLink->TxAck(stream)); // record remoteSendDoneNotify_
        CHK_RET(dstLink->RxAck(stream)); // wait localSendDoneNotify_
        DeviceMem dstMem = inputMem_.range(rxSlice.offset, rxSlice.size);
        DeviceMem srcMem = scratchMem_.range(scratchSlices_[remoteRank].offset, scratchSlices_[remoteRank].size);
        void *remoteMem = nullptr;
        CHK_RET(dstLink->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
        if ((INLINE_REDUCE_BITMASK & reduceAttr_) != 0) {
            CHK_RET(HcclReduceAsync(dispatcher_, static_cast<s8 *>(remoteMem) + baseOffset_ + rxSlice.offset,
                rxSlice.size / SIZE_TABLE[dataType_], dataType_, reductionOp_, stream,
                dstMem.ptr(), dstLink->GetRemoteRank(), dstLink->GetLinkType(), INLINE_REDUCE_BIT));
        } else {
            DeviceMem srcDevMem(static_cast<s8 *>(remoteMem) + baseOffset_ + rxSlice.offset, rxSlice.size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, srcMem, srcDevMem, stream,
                dstLink->GetRemoteRank(), dstLink->GetLinkType()));
        }
        CHK_RET(dstLink->TxDataSignal(stream));
        CHK_RET(dstLink->RxDataSignal(stream));
        CHK_RET(dstLink->RxWaitDone(stream));
        CHK_RET(dstLink->TxWaitDone(stream));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshAtomic::RunReduceScatter(const std::vector<LINK> &links)
{
    // 拼接所有stream
    vector<Stream> streamVct;
    streamVct.reserve(localRankSize_ - 1); // 有ranksize-1个对端，每个对端对应一条stream
    streamVct.push_back(stream_);          // 增加主stream
    streamVct.insert(streamVct.end(), meshStreams_.begin(), meshStreams_.end()); // 增加从stream

    // 每个stream只负责一个对端的交互
    for (u32 streamIndex = 0; streamIndex < localRankSize_ - 1; streamIndex++) {
        u32 remoteRank = (streamIndex + localRank_ + 1) % localRankSize_;
        const LINK &dstLink = links[remoteRank];
        Stream &stream = streamVct[streamIndex];

        CHK_RET(dstLink->TxAck(stream));
        CHK_RET(dstLink->RxAck(stream));
    }

    for (u32 streamIndex = 0; streamIndex < localRankSize_ - 1; streamIndex++) {
        u32 remoteRank = (streamIndex + localRank_ + 1) % localRankSize_;
        const LINK &dstLink = links[remoteRank];
        Stream &stream = streamVct[streamIndex];
        profilerInput_.streamID = stream.id();
        profilerInput_.planeID = streamIndex - 1;
        profilerInput_.step = HCCL_EXEC_STEP_NOT_SET;

        Slice &rxSlice = slices_[localRank_];
        if (streamIndex == 0) {
            for (u32 signalIndex = 0; signalIndex < localRankSize_ - 2; signalIndex++) { // rankSize-2: stream num
                CHK_RET(LocalNotify::Wait(stream, dispatcher_, meshSignal_[signalIndex], profilerInput_.stage));
            }
            for (u32 signalIndex = 0; signalIndex < localRankSize_ - 2; signalIndex++) { // rankSize-2: stream num
                CHK_RET(LocalNotify::Post(stream, dispatcher_, meshSignalAux_[signalIndex],
                    profilerInput_.stage));
            }
        } else {
            u32 signalIndex = streamIndex - 1;
            CHK_RET(LocalNotify::Post(stream, dispatcher_, meshSignal_[signalIndex],
                profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(stream, dispatcher_, meshSignalAux_[signalIndex], profilerInput_.stage));
        }

        HCCL_DEBUG(
            "ReduceScatterMeshAtomic RX rank[%u] inputMemSize[%llu], rxSlice.offset[%llu], rxSlice.size[%llu] "
            "baseOffset_[%llu]",
            localRank_, inputMem_.size(), rxSlice.offset, rxSlice.size, baseOffset_);
        DeviceMem dstMem = inputMem_.range(rxSlice.offset, rxSlice.size);
        void *remoteMem = nullptr;
        CHK_RET(dstLink->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
        CHK_RET(HcclReduceAsync(dispatcher_, static_cast<s8 *>(remoteMem) + baseOffset_ + rxSlice.offset,
            dstMem.size() / SIZE_TABLE[dataType_], dataType_, reductionOp_, stream,
            dstMem.ptr(), dstLink->GetRemoteRank(), dstLink->GetLinkType(), INLINE_REDUCE_BIT));

        CHK_RET(dstLink->TxDataSignal(stream));
        CHK_RET(dstLink->RxDataSignal(stream));
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, streamVct[0], dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshAtomic::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("ReduceScatterMeshAtomic run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    localRank_ = rank;
    localRankSize_ = rankSize;

    if (localRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[ReduceScatterMeshAtomic][RunAsync]rank[%u] linksize[%zu] error", rank, links.size());
        return HCCL_E_INTERNAL;
    }
    CHK_RET(MemSlice());

    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize-2: stream num
        HCCL_DEBUG("rank[%u] streamindex[%u] wait signalaux[%p]",
            rank, streamIndex, meshSignalAux_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_, meshSignalAux_[streamIndex],
            profilerInput_.stage));
    }
    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize-2: stream num
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux_[streamIndex],
            profilerInput_.stage));
    }

    if (GetExternalInputHcclHighPerfEnable() != 0) {
        CHK_RET(RunReduceScatterHighPerf(links));
    } else {
        CHK_RET(RunReduceScatter(links));
    }

    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize - 2 stream num
        HCCL_DEBUG("rank[%u] streamindex[%u] wait signal[%p] ",
            rank, streamIndex, meshSignal_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal_[streamIndex], profilerInput_.stage));
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal_[streamIndex],
            profilerInput_.stage));
    }

    // 添加空task,保证执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    if ((GetExternalInputHcclHighPerfEnable() != 0) &&
        (INLINE_REDUCE_BITMASK & reduceAttr_) == 0) {
        for (u32 streamIndex = 0; streamIndex < localRankSize_ - 1; streamIndex++) {
            u32 remoteRank = (streamIndex + localRank_ + 1) % localRankSize_;
            Slice &rxSlice = slices_[localRank_];
            DeviceMem dstMem = inputMem_.range(rxSlice.offset, rxSlice.size);
            DeviceMem srcMem = scratchMem_.range(scratchSlices_[remoteRank].offset, scratchSlices_[remoteRank].size);
            CHK_RET(HcclReduceAsync(dispatcher_, srcMem.ptr(), rxSlice.size / SIZE_TABLE[dataType_], dataType_,
                reductionOp_, stream_, dstMem.ptr(), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, reduceAttr_));
        }
    }
    if (inputMem_ != outputMem_) {
        DeviceMem src = inputMem_.range(slices_[localRank_].offset, slices_[localRank_].size);
        HCCL_DEBUG("rank[%u] copy result from to output[%p] offset[%llu] size[%llu] ", localRank_, outputMem_.ptr(),
            slices_[localRank_].offset, slices_[localRank_].size);
        HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, src, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterMeshAtomic][RunAsync]rank[%u] memcpy async from mem[%p] "
            "to ouputmem[%p] failed",
            rank, src.ptr(), outputMem_.ptr()),
            ret);
    }

    HCCL_INFO("ReduceScatterMeshAtomic finished: rank[%u", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshAtomic::MemSlice()
{
    u32 unitSize = SIZE_TABLE[dataType_];
    if (unitSize == 0) {
        HCCL_ERROR("[ReduceScatterMeshAtomic][RunAsync]rank[%u] unit data size is zero", localRank_);
        return HCCL_E_INTERNAL;
    }

    if (CheckDebugLogLevel()) {
        for (size_t i = 0; i < slices_.size(); i++) {
            HCCL_DEBUG("[ReduceScatterMeshAtomic] rank[%u] index[%zu] size[%llu] offset[%llu]", localRank_, i,
                slices_[i].size, slices_[i].offset);
        }
        HCCL_DEBUG("[ReduceScatterMeshAtomic] localRankSize[%u]", localRankSize_);
    }

    if (slices_.size() == 0) {
        slices_.resize(localRankSize_);
        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < localRankSize_; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = (i * sliceSize);
        }
    }

    scratchSlices_.resize(localRankSize_);
    if (GetExternalInputHcclHighPerfEnable() != 0) {
        for (u32 i = 0; i < localRankSize_; i++) {
            scratchSlices_[i].size = slices_[localRank_].size;
            scratchSlices_[i].offset = slices_[localRank_].size * i;
        }
    } else {
        for (u32 i = 0; i < localRankSize_; i++) {
            scratchSlices_[i].size = slices_[i].size;
            scratchSlices_[i].offset = (scratchMem_.size() < inputMem_.size()) ? 0 : slices_[i].offset;
        }
    }

    if (CheckDebugLogLevel()) {
        for (size_t i = 0; i < slices_.size(); i++) {
            HCCL_DEBUG("ReduceScatterMeshAtomic rank[%u] index[%zu] \
                size[%llu] offset[%llu] scratch_size[%llu] scratch_offset[%llu]",
                localRank_, i, slices_[i].size, slices_[i].offset, scratchSlices_[i].size, scratchSlices_[i].offset);
        }
    }

    return HCCL_SUCCESS;
}
}