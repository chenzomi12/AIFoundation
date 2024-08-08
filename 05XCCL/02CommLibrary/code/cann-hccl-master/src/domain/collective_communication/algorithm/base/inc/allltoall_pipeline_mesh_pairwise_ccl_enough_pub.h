/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_PIPELINE_MESH_PAIRWISE_CCL_ENOUGH_PUB_H
#define ALLTOALL_PIPELINE_MESH_PAIRWISE_CCL_ENOUGH_PUB_H

#include "allltoall_pipeline_base_pub.h"

namespace hccl {

class AlltoallPipelineMeshPairwiseCCLEnough : public AlltoallPipelineBase {
public:
    explicit AlltoallPipelineMeshPairwiseCCLEnough(
        const HcclDispatcher dispatcher,
        const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        HcclWorkflowMode workMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    virtual ~AlltoallPipelineMeshPairwiseCCLEnough();

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
    virtual HcclResult PipelineSend(u32 step, bool isLastStep);
    virtual u32 CalcInterNumSteps();
    HcclResult GetIntraScratchOffset();
    HcclResult PrepareInterData(u32 step);
    HcclResult PrepareIntraData();
    void UpdateIntraStreamInfo(u32 interRankDistance);
    HcclResult SendRecvDataIntraMesh();
    HcclResult SendRecvDataInterMesh(u32 step);
    HcclResult LocalCopyDataRecvFromInter(u32 interRankDistance);

    // 主流搬数据同时准备好下次 mesh 间发送的信息
    std::vector<TxMemoryInfo> nextInterSendData_;
    std::vector<RxMemoryInfo> nextInterRecvData_;

    std::unordered_map<u32, std::vector<u64>> intraScratchOffsetMap_;
    std::unordered_map<u32, std::vector<u64>> intraScratchLengMap_;
    u64 localScratchOffset_;
};
}  // namespace hccl

#endif /* ALLTOALL_PIPELINE_MESH_PAIRWISE_CCL_ENOUGH_PUB_H */