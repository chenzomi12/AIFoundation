
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_all_gather_mesh_executor.h"

namespace hccl {
CollAllGatherMeshExecutor::CollAllGatherMeshExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllGatherMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherMeshExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshExecutor::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = (topoAttr_.deviceType == DevType::DEV_TYPE_910B) &&
        !topoMatcher_->GetExternalInputHcclDeterministic() && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherMeshExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

HcclResult CollAllGatherMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    // 获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 outerRankSize = outerCommInfo.localRankSize;
    u32 commIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = innerCommInfo.localRank;

    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * outerRankSize;
    u64 outerOffset = commIndex * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(baseOffset + outerOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);
    //  第一步，将数据从input内存拷贝到output内存的对应位置
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherMeshExecutor][KernelRun]all gather 4PmeshHD memcpy Failed, Offset[%llu], Size[%llu].",
        baseOffset + outerOffset, inputMemSize), ret);

    // 第二步，各个AI Server 内 multi stream mesh all gather
    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    u32 sliceNum = outerRankSize;
    CHK_RET(PrepareAllgatherSlice(sliceNum, inputMemSize, dataSegsSlice));

    // mesh算法stream数量为server内rank数减1
    CHK_RET(ExecutorBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));

    CHK_RET(ActiveSlaveStreams(param.stream));

    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem.range(baseOffset, inputMemSize * outerRankSize);
    CHK_SMART_PTR_NULL(currentOutputMem);

    std::unique_ptr<ExecutorBase> outerExecutor;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        outerExecutor.reset(
            new (std::nothrow) AllGatherMeshAtomic(dispatcher_, streamInfo_.ringStreams,
            streamInfo_.ringSignal, streamInfo_.ringSignalAux, commIndex, outerRankSize,
            topoAttr_.userRank));
    } else {
        outerExecutor.reset(
            new (std::nothrow) AllGatherMesh(dispatcher_, streamInfo_.ringStreams, streamInfo_.ringSignal,
                streamInfo_.ringSignalAux, commIndex, outerRankSize, topoAttr_.userRank));
    }
    CHK_SMART_PTR_NULL(outerExecutor);
    CHK_RET(outerExecutor->Prepare(currentOutputMem, currentOutputMem, execMem.inputMem,
        execMem.count * outerRankSize, param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED,
        OUTER_BRIDGE_RANK_ID, dataSegsSlice, baseOffset));
    u32 rankSize = outerRankSize;
    CHK_RET(outerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(outerExecutor, outerCommInfo));
    HCCL_INFO("all gather mesh HD outer run success");

    //  第三步， AI server 间 recursive halving doubling all gather
    u64 hdSize = inputMemSize * outerRankSize;
    u64 hdCount = hdSize / perDataSize;

    std::unique_ptr<ExecutorBase> innerExecutor;
    if (UseInterServerRingAlgo(algType_) || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
        // 1-单server-SDMA
        innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        HCCL_INFO("allgather mesh: using ring algo inter-server.");
    } else if (UseInterServerNHRAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
        HCCL_INFO("allgather mesh: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllGatherNHRV1(dispatcher_));
        HCCL_INFO("allgather mesh: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
        HCCL_INFO("allgather mesh: using nonuniform-bruck algo inter-server.");
    } else {
        innerExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
        HCCL_INFO("allgather mesh: using halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
    CHK_RET(innerExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, hdCount,
        param.DataDes.dataType, param.stream, HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
        std::vector<Slice>(COMM_INDEX_0), 0));

    rankSize = innerCommInfo.localRankSize;
    CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + serverIndex,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

    HCCL_INFO("all gather mesh HD inner run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherMeshExecutor", AllGatherMesh, CollAllGatherMeshExecutor);
} // namespace hccl
