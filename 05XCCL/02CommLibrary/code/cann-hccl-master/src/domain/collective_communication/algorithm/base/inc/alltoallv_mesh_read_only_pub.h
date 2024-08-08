/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_MESH_READ_ONLY_PUB_H
#define ALLTOALL_V_MESH_READ_ONLY_PUB_H

#include "executor_base_pub.h"
#include "alltoallv_staged_calculator_pub.h"

// 区分单算子模式和图模式，单算子模式需要主从流并发，每步主流先将下一步需要 SDMA 发到其他卡的数据搬到 scratch，再唤醒从流
// 通过 txack 告知对端数据已ready，对端发起读，与此同时主流做下一步搬移
// 图模式由于对端数据已ready，直接开始sdma读
namespace hccl {

struct RemoteMem {
    DeviceMem remoteScratchPingMem;
    DeviceMem remoteScratchPongMem;
};

struct SendDataBlock {
    u64 sendLen;
    u64 userInOffset;
    u64 scratchOffset;
};

struct ReadDataBlock {
    u64 recvLen;
    u64 remoteOffset;
    u64 recvOffset;
};

struct AlltoallSendRecvInfo {
    std::vector<SendDataBlock> sendInfo;
    std::vector<ReadDataBlock> readInfo;
};

struct DataTrace {
    u32 dataIndex;
    u64 dataOffset;
};

class AlltoAllVMeshReadOnly : public ExecutorBase {
public:
    explicit AlltoAllVMeshReadOnly(const HcclDispatcher dispatcher, Stream &mainStream,
        std::vector<Stream> &subStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain,
        u32 userRank, u32 intraRankSize, const std::vector<LINK> &links,
        const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    ~AlltoAllVMeshReadOnly() override;
    HcclResult Prepare(DeviceMem &userInput, DeviceMem &userOutput, DeviceMem &scratchPingMem,
        DeviceMem &scratchPongMem, StageAlltoAllVAddrInfo &sendAddrInfo, StageAlltoAllVAddrInfo &recvAddrInfo,
        HcclWorkflowMode workMode);
    HcclResult RunAsync();

protected:
private:
    u64 GetGlobalMaxUserInSize();
    u64 GetGraphModeRemoteMemSize(u32 destRank);
    std::string GetStreamIndexString();
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    u32 CalcNumSubStep();
    HcclResult SendRecvData();
    void UpdateCurrRankSendInfo(u32 destRank, std::vector<SendDataBlock>& sendInfo);
    void UpdateCurrRankRecvInfo(u32 destRank, std::vector<ReadDataBlock>& readInfo);
    void UpdateOpBaseSubStreamInfo(u32 step);
    void UpdateGraphModeSubStreamInfo();
    HcclResult PrepareIntraData();
    HcclResult LocalCopy();
    HcclResult RunAlltoall();
    HcclResult RunAlltoallPingPong();

    Stream mainStream_;
    std::vector<Stream> subStreams_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalSubToMain_;
    u32 userRank_;
    u32 intraRank_;
    u32 intraRankSize_;
    const std::vector<LINK> links_;
    const std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo_;

    DeviceMem userInput_;
    DeviceMem userOutput_;
    DeviceMem scratchPingMem_;
    DeviceMem scratchPongMem_;
    std::map<u32, std::vector<OneSendRecvAddrInfo>> sendAddrInfo_;
    std::map<u32, std::vector<OneSendRecvAddrInfo>> recvAddrInfo_;
    HcclWorkflowMode workMode_;
    u64 dataBlockSize_;
    std::unordered_map<u32, RemoteMem> destRankRemoteMem_;

    bool useScratchPingMem_;
    std::unordered_map<u32, AlltoallSendRecvInfo> subStreamSendRecvInfo_;       // 从流当前收发长度和接收到的本地偏移
    std::unordered_map<u32, DataTrace> sendIndexTrace_;            // 当前步骤每个对端发送到第几个大块数据，中的第几个小块数据
    std::unordered_map<u32, DataTrace> recvIndexTrace_;            // 当前步骤每个对端接收到第几个大块数据，中的第几个小块数据
    std::unordered_map<u32, u32> sendNumSubStep_;                       // 需要向对应对端rank发几次数据
    std::unordered_map<u32, u32> recvNumSubStep_;                       // 需要从对应对端rank收几次数据
};
} // namespace hccl
#endif /* ALLTOALL_V_MESH_READ_ONLY_PUB_H */