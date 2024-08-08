/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_mesh_atomic.h"

namespace hccl {
AllGatherMeshAtomic::AllGatherMeshAtomic(const HcclDispatcher dispatcher,
    std::vector<Stream> &meshStreams, const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 interRank, u32 interRankSize, u32 userRank)
    : AllGatherMesh(dispatcher, meshStreams, meshSignal, meshSignalAux, interRank, interRankSize, userRank)
{}

AllGatherMeshAtomic::~AllGatherMeshAtomic() {}

HcclResult AllGatherMeshAtomic::RunAllGather(const std::vector<LINK> &links, const std::vector<Slice> &outputSlices,
    const std::vector<Slice> &inputSlices)
{
    for (u32 round = 1; round < interRankSize_; round++) {
        u32 dstRank = BackwardRank(interRank_, interRankSize_, round);
        Stream& subStream = (round == interRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    for (u32 round = 1; round < interRankSize_; round++) {
        u32 dstRank = BackwardRank(interRank_, interRankSize_, round);
        Stream& subStream = (round == interRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        profilerInput_.streamID = subStream.id();
        profilerInput_.planeID = round - 1;
        profilerInput_.step = HCCL_EXEC_STEP_NOT_SET;

        if (round == interRankSize_ - 1) {
            for (u32 signalIndex = 0; signalIndex < interRankSize_ - 2; signalIndex++) { // rankSize-2: stream num
                    CHK_RET(LocalNotify::Wait(subStream, dispatcher_, meshSignal_[signalIndex], profilerInput_.stage));
            }
            // 为子图增加一个从stream到主stream的附着点
            DeviceMem src = DeviceMem::create(inputMem_.ptr(), 0);
            DeviceMem dst = DeviceMem::create(outputMem_.ptr(), 0);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
            for (u32 signalIndex = 0; signalIndex < interRankSize_ - 2; signalIndex++) { // rankSize-2: stream num
                CHK_RET(LocalNotify::Post(subStream, dispatcher_, meshSignalAux_[signalIndex],
                    profilerInput_.stage));
            }
        } else {
            u32 signalIndex = round - 1;
            CHK_RET(LocalNotify::Post(subStream, dispatcher_, meshSignal_[signalIndex],
                profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(subStream, dispatcher_, meshSignalAux_[signalIndex], profilerInput_.stage));
        }
        // 本rank要收数据
        void *srcMemPtr = nullptr;
        // 从对端的input内存拿数据，input==output也没有关系
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &srcMemPtr));
        DeviceMem srcDevMem(static_cast<s8 *>(srcMemPtr) + baseOffset_ + inputSlices[dstRank].offset,
            inputSlices[dstRank].size);
        DeviceMem dstDevMem = outputMem_.range(outputSlices[dstRank].offset, outputSlices[dstRank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstDevMem, srcDevMem, subStream,
            links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    return HCCL_SUCCESS;
}
} // namespace hccl
