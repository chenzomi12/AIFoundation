/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_mesh_direct.h"
// userin -> dmaout -> userout
namespace hccl {
AllgatherMeshDirect::AllgatherMeshDirect(const HcclDispatcher dispatcher,
    std::vector<Stream> &meshStreams, const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 interRank,
    u32 interRankSize, u32 userRank, HcomCollOpInfo *opInfo)
    : ExecutorBase(dispatcher),
      meshStreams_(meshStreams),
      meshSignal_(meshSignal),
      meshSignalAux_(meshSignalAux),
      interRank_(interRank),
      interRankSize_(interRankSize),
      userRank_(userRank),
      opInfo_(opInfo)
{}

AllgatherMeshDirect::~AllgatherMeshDirect() {}

HcclResult AllgatherMeshDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAux_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux_[signalIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAux_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_, meshSignalAux_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignal_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignal_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// allgather的入口函数
HcclResult AllgatherMeshDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("AllGatherMesh run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]",
        rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);
    u32 unitSize = DataUnitSize(dataType_);
    u64 sdmaSize = count_ * unitSize; // 当前count
    u64 sliceSize = opInfo_->count * unitSize; // 总输入count
    DeviceMem src;
    DeviceMem dst;

    char* curUerMemInPtr = static_cast<char *>(opInfo_->inputAddr);
    char* curUerMemOutPtr = static_cast<char *>(opInfo_->outputAddr);
    char* curCommMemOutPtr = static_cast<char *>(outputMem_.ptr());

    if (rankSize == 1) {
        if (opInfo_->inputAddr != opInfo_->outputAddr) {
            DeviceMem userMemIn = DeviceMem::create(curUerMemInPtr, sdmaSize);
            DeviceMem userMemOut = DeviceMem::create(curUerMemOutPtr, sdmaSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_));
        }
        return HCCL_SUCCESS;
    }

    src = DeviceMem::create(curUerMemInPtr, sdmaSize);
    u64 localOffsetByte = (sliceSize * rank) % HCCL_MIN_SLICE_ALIGN_910B;
    dst = DeviceMem::create(curCommMemOutPtr + localOffsetByte, sdmaSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(SubWaitMain());
    CHK_RET(MainRecordSub());

    src = dst;
    dst = DeviceMem::create(curUerMemOutPtr + rank * sliceSize, sdmaSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = meshStreams_[round - 1];
        // 本rank要收数据
        void *remMemPtr = nullptr;
        // 从对端的input内存拿数据，input==output也没有关系
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        u64 remoteOffsetByte = (sliceSize * dstRank) % HCCL_MIN_SLICE_ALIGN_910B;
        src = DeviceMem::create(static_cast<char *>(remMemPtr) + remoteOffsetByte, sdmaSize);
        dst = DeviceMem::create(curUerMemOutPtr + dstRank * sliceSize, sdmaSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStream,
            links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }
    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    HCCL_INFO("AllGatherMesh finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
} // namespace hccl
