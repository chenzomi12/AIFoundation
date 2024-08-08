/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COLL_RUN_ALLTOALLV_TWO_LEVEL_PIPELINE_H
#define COLL_RUN_ALLTOALLV_TWO_LEVEL_PIPELINE_H
#include "coll_all_to_all_executor.h"
namespace hccl {

class CollRunAlltoAllVTwoLevelPipeline : public CollAlltoAllExecutor {
public:
    CollRunAlltoAllVTwoLevelPipeline(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollRunAlltoAllVTwoLevelPipeline() = default;

private:
    u64 GetAlltoall2LevelPipelineScratchSize910B(u32 rank, std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    u64 GetAlltoall2LevelPipelineMaxScratchSize910B(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;
    HcclResult CalcStreamNum(u32& streamNum) override;

    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalNoScratchAlltoallCommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
};

} // namespace hccl
#endif