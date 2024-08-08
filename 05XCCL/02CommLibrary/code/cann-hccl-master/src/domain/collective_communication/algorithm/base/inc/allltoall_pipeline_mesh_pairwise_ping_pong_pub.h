/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_PIPELINE_MESH_PAIRWISE_PING_PONG_PUB_H
#define ALLTOALL_PIPELINE_MESH_PAIRWISE_PING_PONG_PUB_H

#include "allltoall_pipeline_base_pub.h"

namespace hccl {

class AlltoallPipelineMeshPairwisePingPong : public AlltoallPipelineBase {
public:
    explicit AlltoallPipelineMeshPairwisePingPong(
        const HcclDispatcher dispatcher,
        const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        HcclWorkflowMode workMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    virtual ~AlltoallPipelineMeshPairwisePingPong();

    virtual HcclResult Prepare(u32 userRank, A2aPipelineMemory A2aPipelineMemory,
        std::unique_ptr<CommBase> &commOuter, std::unique_ptr<CommBase> &commInner,
        Stream &mainStream, std::vector<Stream> &subStream,
        std::vector<std::shared_ptr<LocalNotify>> &notifyMain, std::vector<std::shared_ptr<LocalNotify>> &notifySub);
    // 适配新CollExecutor接口
    virtual HcclResult Prepare(u32 userRank, A2aPipelineMemory A2aPipelineMemory,
        const SubCommInfo &outerCommInfo, const SubCommInfo &innerCommInfo,
        Stream &mainStream, std::vector<Stream> &subStream,
        std::vector<std::shared_ptr<LocalNotify>> &notifyMain, std::vector<std::shared_ptr<LocalNotify>> &notifySub);

    HcclResult RunAsync();

private:
    virtual HcclResult DeviceMemMapping();
    virtual HcclResult PreProcess();
    virtual HcclResult PostProcess();
    virtual u32 CalcInterNumSteps();
    HcclResult PrepareInterSendData(
        u32 mainStep,
        u32 subStep);
    HcclResult PrepareInterRecvData(
        u32 mainStep,
        u32 subStep);
    HcclResult PrepareInterData(
        u32 mainStep,
        u32 subStep);
    HcclResult PrepareIntraData(u32 subStep);
    void UpdateRemoteMemStatusIntra(u32 step);
    void UpdateRemoteMemStatusInter(u32 step);
    void UpdateIntraStreamInfo(
        u32 interRankDistance,
        u32 subStep);
    HcclResult SendRecvDataIntraMesh();
    HcclResult SendRecvDataInterMesh(
        u32 step,
        bool doSend,
        bool doRecv);
    HcclResult LocalCopyDataRecvFromInter(
        u32 mainStep,
        u32 subStep);

    void GetNumSubStep(
        u32 step,
        u32& interSendSubStep,
        u32& interRecvSubStep,
        u32& intraSubStep);
    virtual HcclResult PipelineSend(
        u32 step,
        bool isLastStep);

    // 主流搬数据同时准备好下次 mesh 间发送的信息
    std::vector<TxMemoryInfo> nextInterSendData_;
    std::vector<RxMemoryInfo> nextInterRecvData_;

    // 采用内存 ping-pong 的方法，单算子模式将 ccl 拆成 4 块，以达成，server 内收发，server 间收发，本地内存搬移三流并行的效果
    // interRecv 和 intraSend 使用同一块内存，在图模式时不需要 interSend 这块内存，也没有从 userIn 到 interSend 的拷贝，可以直接从 userIn 发到对端 interRecv
    u64 pingPongMemSize_;
    u64 intraDataBlockSize_;
    DeviceMem interSendPing_;
    DeviceMem interSendPong_;
    DeviceMem interRecvPing_;
    DeviceMem interRecvPong_;
    DeviceMem intraSendPing_;
    DeviceMem intraSendPong_;

    // 第 0 步将第一次需要 mesh 间收发的数据放到 interSendPing，将需要做 mesh 内收发的数据放到 intraSendPong;
    // 第一步将 interSendPing 的数据发到 mesh 间对端 interRecvPing，将 intraSendPong 的数据发到 mesh 内对端 userOut ，
    //      与此同时，主流将下一次做 mesh 间收发的数据搬到 interSendPong，
    // 第二步将 interSendPong 的数据发到 mesh 间对端 interRecvPong，将 intraSendPing 的数据发到 mesh 内对端 userOut ,
    //      与此同时，主流将下一次做 mesh 间收发的数据搬到 interSendPing
    // 以此类推
    bool interSendUsePingMem_ = true;
    bool interRecvUsePingMem_ = true;
    bool intraSendUsePingMem_ = false;
    // mesh 间对端和 mesh 内对端使用 ping-pong 的信息，每个大步骤更新
    bool recvFromInterSrcMemPing_ = true;
    bool sendToInterDstMemPing_ = true;
    std::vector<bool> memStatusInMesh_;
};
}  // namespace hccl

#endif /* ALLTOALL_PIPELINE_MESH_PAIRWISE_PING_PONG_PUB_H */