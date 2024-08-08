/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_reduce_mesh_executor.h"

namespace hccl {

CollReduceMeshExecutor::CollReduceMeshExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CalcLevel0CommInfo(inputType, outputType, opTransport);
    CalcLevel1CommInfo(inputType, outputType, opTransport);
    return HCCL_SUCCESS;
}

HcclResult CollReduceMeshExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollReduceMeshExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceMeshExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = meshSinglePlane_;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
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

    ActiveSlaveStreams(param.stream);

    if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_CONFIG_DISABLE && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64) &&
        (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(tag_, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
            dataSegsSlice, const_cast<Stream&>(param.stream), COMM_LEVEL0));
    } else {
        std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(ExecutorBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(tag_, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType, multiStreamSlice,
            const_cast<Stream&>(param.stream), COMM_LEVEL0));
    }
    HCCL_INFO("reduce mesh stage0 run success");

    // step2: 节点间的reduce
    u32 commIndex = outerCommInfo.localRank;
    CHK_PRT_RET(commIndex >= dataSegsSlice.size(), HCCL_ERROR("[CollReduceMeshExecutor][Run]commIndex[%u] >= dataSegsSlice size[%llu]",
        commIndex, dataSegsSlice.size()), HCCL_E_INTERNAL);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, tag_.c_str(),
        innerCommInfo.links.size());

    DeviceMem reduceInput = execMem.inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(reduceInput);
    DeviceMem reduceOutput = execMem.outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(reduceOutput);

    u32 rankSize = innerCommInfo.localRankSize;
    if (rankSize > 1) {
        u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, param.DataDes.dataType, param.reduceType);

        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        
        CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[ReduceOperator][ReduceMeshExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                subUserrankRoot, topoAttr_.userRank, param.root), HCCL_E_INTERNAL);
        
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));

        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
        }
        CHK_SMART_PTR_NULL(innerExecutor);
        // 节点间的hd 使用环0来记录
        u64 hdCount = dataSegsSlice[commIndex].size / perDataSize;

        CHK_RET(innerExecutor->Prepare(reduceInput, reduceOutput, reduceOutput, hdCount, param.DataDes.dataType, param.stream,
            param.reduceType, planeRoot, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

        CHK_RET(innerExecutor->RegisterProfiler((innerCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
    } else {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, reduceOutput, reduceInput, const_cast<Stream&>(param.stream)));
    }

    HCCL_INFO("reduce mesh stage1 run success");

    SingleSubCommTransport &outerTransportInfo =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_LEVEL0][COMM_INDEX_0]);

    if (outerTransportInfo.userRank2subCommRank.find(param.root) !=
        outerTransportInfo.userRank2subCommRank.end()) {
        const u32 rootRank = outerTransportInfo.userRank2subCommRank[param.root];

        std::unique_ptr<ExecutorBase> outerExecutor;
        outerExecutor.reset(new (std::nothrow) GatherMesh(dispatcher_, streamInfo_.ringStreams,
                streamInfo_.ringSignal, streamInfo_.ringSignalAux, topoAttr_.userRank));
        CHK_SMART_PTR_NULL(outerExecutor);
        CHK_RET(outerExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            param.DataDes.dataType, const_cast<Stream &>(param.stream), param.reduceType, rootRank, dataSegsSlice));

        u32 rankSize = outerCommInfo.localRankSize;
        CHK_RET(outerExecutor->RegisterProfiler((0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(outerExecutor, outerCommInfo));
    }
    HCCL_INFO("reduce mesh stage2 run success");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceMeshExecutor", ReduceMesh, CollReduceMeshExecutor);

} // namespace hccl