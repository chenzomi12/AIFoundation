/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_staged_mesh.h"
#include "log.h"

namespace hccl {
using namespace std;

AlltoAllVStagedMesh::AlltoAllVStagedMesh(const HcclDispatcher dispatcher, Stream &stream,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain,
    u32 userRank, std::vector<Stream> &subStreams)
    : AlltoAllVStagedBase(dispatcher, stream),
      meshSignalMainToSub_(meshSignalMainToSub),
      meshSignalSubToMain_(meshSignalSubToMain),
      userRank_(userRank),
      subStreams_(subStreams)
{}

AlltoAllVStagedMesh::~AlltoAllVStagedMesh() {}

// 图模式Prepare入口
HcclResult AlltoAllVStagedMesh::Prepare(DeviceMem &sendMem, DeviceMem &recvMem, StageAlltoAllVAddrInfo &sendAddrInfo,
    StageAlltoAllVAddrInfo &recvAddrInfo, bool isAlltoAllZCopyMode, const std::vector<Stream> &subStreams)
{
    sendMem_ = sendMem;
    recvMem_ = recvMem;
    sendAddrInfo_ = sendAddrInfo;
    recvAddrInfo_ = recvAddrInfo;
    subStreams_ = subStreams;
    isAlltoAllZCopyMode_ = isAlltoAllZCopyMode;

    HCCL_DEBUG("[AlltoAllVStagedMesh][Prepare] finished and isAlltoAllZCopyMode_[%d]", isAlltoAllZCopyMode_);
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVStagedMesh::Prepare(DeviceMem &sendMem, DeviceMem &recvMem, DeviceMem &scratchInputMem,
    DeviceMem &scratchOutputMem, StageAlltoAllVAddrInfo &sendAddrInfo, StageAlltoAllVAddrInfo &recvAddrInfo,
    bool isAlltoAllZCopyMode, const std::vector<Stream> &subStreams)
{
    HCCL_ERROR("AlltoAllv Staged Mesh is not supported in Op base mode!");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AlltoAllVStagedMesh::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("[AlltoAllVStagedMesh][RunAsync]: rank[%u] transportSize[%llu]", rank, links.size());
    CHK_SMART_PTR_NULL(dispatcher_);

    CHK_PRT_RET(rankSize == 0, HCCL_ERROR("[AlltoAllVStagedMesh][Prepare] invilad rankSize[%u]", rankSize),
        HCCL_E_PARA);

    CHK_PRT_RET(rankSize != links.size(),
        HCCL_ERROR("[AlltoAllVStagedMesh][RunAsync]: rankSize[%u] and transport size[%llu] do not match", rankSize,
        links.size()),
        HCCL_E_PARA);

    CHK_PRT_RET(rankSize > 1 && subStreams_.size() < rankSize - 2, // 从流个数是rankSize - 2，ranksize为1时不需要校验
        HCCL_ERROR("[AlltoAllVStagedMesh][RunAsync]: rankSize[%u] and stream size[%llu] do not match", rankSize,
        subStreams_.size()),
        HCCL_E_PARA);

    bool sizeEqual = (sendAddrInfo_.size() == recvAddrInfo_.size() && sendAddrInfo_.size() == rankSize);
    CHK_PRT_RET(!sizeEqual,
        HCCL_ERROR("[AlltoAllVStagedMesh][RunAsync] invilad params: "\
        "sendAddrInfo size[%u] recvAddrInfo size[%u] rankSize[%u]",
        sendAddrInfo_.size(), recvAddrInfo_.size(), rankSize),
        HCCL_E_PARA);

    CHK_RET(LocalCopy(rank));
    CHK_RET(RunZCopyMode(rank, rankSize, links));

    return HCCL_SUCCESS;
}

void AlltoAllVStagedMesh::BuildSendRecvMemoryInfo(vector<TxMemoryInfo> &txMems, vector<RxMemoryInfo> &rxMems,
    u32 destRank)
{
    u32 index = 0;
    for (auto &addrInfo : sendAddrInfo_[destRank]) {
        txMems[index].dstMemType = UserMemType::OUTPUT_MEM;
        txMems[index].dstOffset = addrInfo.remoteOffset;
        txMems[index].src = static_cast<u8 *>(sendMem_.ptr()) + addrInfo.localOffset;
        txMems[index].len = addrInfo.localLength;
        index++;
    }

    index = 0;
    for (auto &addrInfo : recvAddrInfo_[destRank]) {
        rxMems[index].srcMemType = UserMemType::INPUT_MEM;
        rxMems[index].srcOffset = addrInfo.remoteOffset;
        rxMems[index].dst = static_cast<u8 *>(recvMem_.ptr()) + addrInfo.localOffset;
        rxMems[index].len = addrInfo.localLength;
        index++;
    }
    return;
}

HcclResult AlltoAllVStagedMesh::RunZCopyMode(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    // 从stream wait, 主stream record
    for (u32 i = 0; i < rankSize - 2; i++) { // 从stream 个数 = ranksize -2
        CHK_RET(LocalNotify::Wait(subStreams_[i], dispatcher_, meshSignalMainToSub_[i], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, meshSignalMainToSub_[i], INVALID_VALUE_STAGE));
    }

    for (u32 i = 1; i < rankSize; i++) {
        u32 destRank = (rank + i) % rankSize;
        shared_ptr<Transport> destTransport = links[destRank];
        Stream &currentStream = (i == 1) ? mainStream_ : subStreams_[i - 2];

        u32 sendDataNum = sendAddrInfo_[destRank].size();
        vector<TxMemoryInfo> txMems(sendDataNum);
        u32 recvDataNum = recvAddrInfo_[destRank].size();
        vector<RxMemoryInfo> rxMems(recvDataNum);

        BuildSendRecvMemoryInfo(txMems, rxMems, destRank);
        CHK_RET(LoadTask(destTransport, currentStream, txMems, rxMems));
    }
    // 主stream wait, 从stream record
    for (u32 i = 0; i < rankSize - 2; i++) { // 从stream 个数 = ranksize -2
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, meshSignalSubToMain_[i], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Post(subStreams_[i], dispatcher_, meshSignalSubToMain_[i], INVALID_VALUE_STAGE));
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(sendMem_, recvMem_, mainStream_, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVStagedMesh::LoadTask(shared_ptr<Transport> destTransport, Stream &currentStream,
    vector<TxMemoryInfo> &txMems, vector<RxMemoryInfo> &rxMems) const
{
    CHK_RET(destTransport->TxAck(currentStream));           // record send done
    CHK_RET(destTransport->RxAck(currentStream));           // wait send done
    CHK_RET(destTransport->TxAsync(txMems, currentStream)); // record Send Ready
    CHK_RET(destTransport->RxAsync(rxMems, currentStream)); // wait Send Ready + SDMA get
    CHK_RET(destTransport->TxAck(currentStream));           // record send done
    CHK_RET(destTransport->RxAck(currentStream));           // wait send done
    CHK_RET(destTransport->TxDataSignal(currentStream));    // record Send Ready
    CHK_RET(destTransport->RxDataSignal(currentStream));    // wait send Ready
    return HCCL_SUCCESS;
}
} // namespace hccl
