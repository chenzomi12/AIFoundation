/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh_atomic_opbase.h"
#include "externalinput_pub.h"

namespace hccl {
using namespace std;

ReduceScatterMeshDirect::ReduceScatterMeshDirect(const HcclDispatcher dispatcher,
    const u64 reduceAttrBitMap, std::vector<Stream> &meshStreams,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 userRank, HcomCollOpInfo *opInfo)
    : ExecutorBase(dispatcher),
      reduceAttr_(reduceAttrBitMap),
      localRank_(0),
      localRankSize_(0),
      userRank_(userRank),
      meshStreams_(meshStreams),
      meshSignal_(meshSignal),
      meshSignalAux_(meshSignalAux),
      opInfo_(opInfo)
{}

ReduceScatterMeshDirect::~ReduceScatterMeshDirect() {}

HcclResult ReduceScatterMeshDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAux_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux_[signalIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAux_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_,
            meshSignalAux_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignal_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignal_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("ReduceScatterMeshDirect run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];
    u64 sliceSize = count_ * unitSize;

    DeviceMem userMemIn =
        DeviceMem::create(static_cast<char *>(opInfo_->inputAddr) + ((opInfo_->count) * unitSize * rank), sliceSize);
    DeviceMem userMemOut = DeviceMem::create(static_cast<char *>(opInfo_->outputAddr), sliceSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    dst = commMemOut.range(0, sliceSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, userMemIn, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    // 每个stream只负责一个对端的交互
    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        const LINK &dstLink = links[dstRank];
        Stream &subStream = meshStreams_[round - 1];
        CHK_RET(dstLink->TxAck(subStream));
        CHK_RET(dstLink->RxAck(subStream));
    }
    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    // 为子图增加一个从stream到主stream的附着点
    DeviceMem srcZero = DeviceMem::create(inputMem_.ptr(), 0);
    DeviceMem dstZero = DeviceMem::create(outputMem_.ptr(), 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstZero, srcZero, stream_));

    CHK_RET(SubWaitMain());
    CHK_RET(MainRecordSub());

    // inline执行notice reduce
    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        const LINK &dstLink = links[dstRank];
        Stream &subStream = meshStreams_[round - 1];
        // 本rank要发数据
        void *remMemPtr = nullptr;
        // 获取远端的commoutMem
        CHK_RET(dstLink->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
        dst = DeviceMem::create(static_cast<char *>(remMemPtr), sliceSize);
        u64 suboffset = (opInfo_->count) * unitSize * dstRank;
        src = DeviceMem::create(static_cast<char *>(opInfo_->inputAddr) + suboffset, sliceSize);
        CHK_RET(HcclReduceAsync(dispatcher_, static_cast<void *>(src.ptr()), count_, dataType_, reductionOp_,
            subStream, static_cast<void *>(dst.ptr()), dstLink->GetRemoteRank(), dstLink->GetLinkType(),
            INLINE_REDUCE_BIT));

        CHK_RET(dstLink->TxDataSignal(subStream));
        CHK_RET(dstLink->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    // commout--> useroutput
    DeviceMem srcMem = commMemOut.range(0, sliceSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemOut, srcMem, stream_));

    HCCL_INFO("ReduceScatterMeshDirect finished: rank[%u", rank);
    return HCCL_SUCCESS;
}
}