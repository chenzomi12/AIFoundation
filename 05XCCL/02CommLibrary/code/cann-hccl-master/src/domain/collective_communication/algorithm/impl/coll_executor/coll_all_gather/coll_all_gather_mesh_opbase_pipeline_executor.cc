/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_all_gather_mesh_opbase_pipeline_executor.h"

namespace hccl {
CollAllGatherMeshOpbasePipelineExecutor::CollAllGatherMeshOpbasePipelineExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllGatherMeshOpbasePipelineExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherMeshOpbasePipelineExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshOpbasePipelineExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshOpbasePipelineExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO("[CollAllGatherMeshOpbasePipelineExecutor][CalcTransportMemType]" \
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshOpbasePipelineExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

// PipeLine模式下使用Ring算法
HcclResult CollAllGatherMeshOpbasePipelineExecutor::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    return maxCountPerLoop;
}

bool CollAllGatherMeshOpbasePipelineExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize > RDMA_SEND_MAX_SIZE || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollAllGatherMeshOpbasePipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherMeshOpbasePipelineExecutor][KernelRun]AllGatherMeshOpbasePipelineExecutor begins.");

    // step 1 先获取 comm inner \ comm outer 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // DMA消减场景，打包opInfo
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };

    std::unique_ptr<AllGatherPipeline> executor;
    executor.reset(new (std::nothrow) AllGatherPipeline(dispatcher_));
    CHK_SMART_PTR_NULL(executor);
    CHK_RET(executor->Prepare(&opInfo, topoAttr_.userRank, execMem.count, execMem.inputMem, execMem.outputMem,
        outerCommInfo, innerCommInfo, const_cast<Stream&>(param.stream),
        streamInfo_.ringStreams, streamInfo_.ringSignal, streamInfo_.ringSignalAux));
    CHK_RET(executor->RunAsync());
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherMeshOpbasePipelineExecutor", AllGatherOpbasePipeline, CollAllGatherMeshOpbasePipelineExecutor);

} // namespace hccl
