/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mesh_opbase_pipeline_executor.h"

namespace hccl {
// 准入条件: pipeLine && 910B && 单算子 && sdmaReduce && rdmaReduce && 多Mesh && MeshTopo && 非确定性
CollAllReduceMeshOpbasePipelineExecutor::CollAllReduceMeshOpbasePipelineExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher): CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllReduceMeshOpbasePipelineExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllReduceMeshOpbasePipelineExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbasePipelineExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbasePipelineExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO("[CollAllReduceMeshOpbasePipelineExecutor][CalcTransportMemType]" \
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbasePipelineExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaInfo.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

// PipeLine模式下使用Ring算法
HcclResult CollAllReduceMeshOpbasePipelineExecutor::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllReduceMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    if (cclBuffSize <= HCCL_MIN_SLICE_ALIGN_910B) {
        return 0;
    }
    u64 maxCountPerLoop = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) /
        unitSize * topoAttr_.userRankSize;
    return maxCountPerLoop;
}

bool CollAllReduceMeshOpbasePipelineExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollAllReduceMeshOpbasePipelineExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    return false;
}

HcclResult CollAllReduceMeshOpbasePipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllReduceMeshOpbasePipelineExecutor][Run]CollAllReduceMeshOpbasePipelineExecutor begins.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };

    std::unique_ptr<AllReduceOpbasePipeline> executor;
    executor.reset(new (std::nothrow) AllReduceOpbasePipeline(dispatcher_, reduceAttr));
    CHK_SMART_PTR_NULL(executor);
    CHK_RET(executor->Prepare(&opInfo, execMem.inputMem, execMem.outputMem, execMem.count,
        innerCommInfo, outerCommInfo, const_cast<Stream&>(param.stream),
        streamInfo_.ringStreams, streamInfo_.ringSignal, streamInfo_.ringSignalAux));
    CHK_RET(executor->RunAsync());
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMeshOpbasePipelineExecutor",
    AllReduceMeshOpbasePipeline, CollAllReduceMeshOpbasePipelineExecutor);

} // namespace hccl