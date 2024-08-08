/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mesh_executor.h"

namespace hccl {

CollAllReduceMeshExecutor::CollAllReduceMeshExecutor(const HcclDispatcher dispatcher,
                                                     std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

void CollAllReduceMeshExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    bool isInlineReduce = IsSupportSDMAReduce(param.inputPtr, param.outputPtr,
        param.DataDes.dataType, param.reduceType);
    meshSinglePlane_ = (topoAttr_.deviceType == DevType::DEV_TYPE_910B) &&
        !topoMatcher_->GetExternalInputHcclDeterministic() &&
        isInlineReduce && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

HcclResult CollAllReduceMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllReduceMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllReduceMeshExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = meshSinglePlane_;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollAllReduceMeshExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollAllReduceMeshExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    bool smallData = IsAllReduceSmallData(curSize);
    return smallData;
}

HcclResult CollAllReduceMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    std::vector<Slice> dataSegsSlicePerDie; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    std::unique_ptr<ExecutorBase> innerExecutor;
    std::unique_ptr<ExecutorBase> outer2Executor;

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 sliceNum = outerCommInfo.localRankSize;
    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    // mesh算法stream数量为server内rank数减1

    CHK_RET(ActiveSlaveStreams(param.stream));

    if (!topoMatcher_->GetExternalInputHcclDeterministic() && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64) &&
        ((topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD) ||
        (IsSupportHighPerf() && param.reduceType == HCCL_REDUCE_SUM))) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
            param.DataDes.dataType, param.reduceType, dataSegsSlice, const_cast<Stream&>(param.stream), COMM_LEVEL0));
    } else {
        CHK_RET(ExecutorBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
            param.DataDes.dataType, param.reduceType, multiStreamSlice,
            const_cast<Stream&>(param.stream), COMM_LEVEL0));
    }

    HCCL_INFO("allreduce meshhd stage0 run success.");

    /* 内层topo:all_reduce */
    /* 外层所有rank均参与内层的allReduce计算，所以此处对rank不作限制，但是每个rank需找到自己所在的内层通信域 */
    u32 commIndex = outerCommInfo.localRank;
    CHK_PRT_RET(commIndex >= dataSegsSlice.size(),
        HCCL_ERROR("[CollAllReduceMeshExecutor][Run]commIndex[%u] >= dataSegsSlice size[%llu]", commIndex,
        dataSegsSlice.size()), HCCL_E_INTERNAL);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceInput);
    DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceOutput);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    if (UseInterServerRingAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce mesh: using ring algo inter-server.");
    } else if (UseInterServerNHRAlgo(algType_)) {
        u64 curSize = execMem.count * perDataSize; // 单位 byte
        HCCL_DEBUG("allreduce mesh: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
            curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
        if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
            innerExecutor.reset(new (std::nothrow) AllReduceNHROneshot(dispatcher_, reduceAttr));
        } else {
            innerExecutor.reset(new (std::nothrow) AllReduceNHR(dispatcher_, reduceAttr));
        }
        HCCL_INFO("allreduce mesh: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllReduceNHRV1(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce mesh: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllReduceNB(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce mesh: using nb algo inter-server.");
    } else {
        innerExecutor.reset(new (std::nothrow) AllReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce mesh: using Recursive halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    u32 rankSize = innerCommInfo.localRankSize;
    // 节点间的hd 使用环0来记录

    u64 hdCount = dataSegsSlice[commIndex].size / perDataSize;
    CHK_RET(innerExecutor->Prepare(allreduceInput, allreduceOutput, allreduceOutput, hdCount,
        param.DataDes.dataType, param.stream, param.reduceType,
        OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

    CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        innerCommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

    HCCL_INFO("allreduce meshhd stage1 run success.");

    /* 外层topo:all_gather */

    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        outer2Executor.reset(
            new (std::nothrow) AllGatherMeshAtomic(dispatcher_, streamInfo_.ringStreams,
            streamInfo_.ringSignal, streamInfo_.ringSignalAux, outerCommInfo.localRank, outerCommInfo.localRankSize,
            topoAttr_.userRank));
    } else {
        outer2Executor.reset(
            new (std::nothrow) AllGatherMesh(dispatcher_, streamInfo_.ringStreams,
            streamInfo_.ringSignal, streamInfo_.ringSignalAux, outerCommInfo.localRank, outerCommInfo.localRankSize,
            topoAttr_.userRank));
    }

    CHK_SMART_PTR_NULL(outer2Executor);

    /* 节点内执行器 stage2 */
    {
        u32 rankSize = outerCommInfo.localRankSize;
        CHK_RET(outer2Executor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType,
            OUTER_BRIDGE_RANK_ID, dataSegsSlice, 0));

        CHK_RET(outer2Executor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(outer2Executor, outerCommInfo));
    }

    HCCL_INFO("allreduce meshhd stage2 run success");
    return HCCL_SUCCESS;
}

bool CollAllReduceMeshExecutor::IsSupportHighPerf()
{
    return ((topoMatcher_->GetExternalInputHcclHighPerfEnable() != 0) &&
            (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
}

REGISTER_EXEC("AllReduceMeshExecutor", AllReduceMesh, CollAllReduceMeshExecutor);

} // namespace hccl