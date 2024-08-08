/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mesh_small_count_executor.h"

namespace hccl {

CollAllReduceMeshSmallCountExecutor::CollAllReduceMeshSmallCountExecutor(const HcclDispatcher dispatcher,
                                                                         std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

void CollAllReduceMeshSmallCountExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    totalSize_ = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

bool CollAllReduceMeshSmallCountExecutor::CalcScratchMemFlag(const u64 totalSize)
{
    return GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        topoAttr_.deviceType == DevType::DEV_TYPE_910B &&
        topoMatcher_->GetExternalInputHcclDeterministic() &&
        topoAttr_.deviceNumPerAggregation > DEVICE_TWO &&
        topoAttr_.deviceNumPerAggregation < DEVICE_EIGHT &&
        totalSize <= HCCL_SMALL_COUNT_GRAPH_64_KB;
}

HcclResult CollAllReduceMeshSmallCountExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (CalcScratchMemFlag(totalSize_) == true) {
        scratchMemSize = totalSize_ * (topoAttr_.userRankSize - 1);
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO("[CollAllReduceMeshSmallCountExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshSmallCountExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        totalStreamNum = topoAttr_.deviceNumPerAggregation - 1U;
    } else {
        totalStreamNum = topoAttr_.deviceNumPerAggregation;
    }
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllReduceMeshSmallCountExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshSmallCountExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshSmallCountExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        if (CalcScratchMemFlag(totalSize_) == true) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::PARAM_OUTPUT;
        }
    }
    HCCL_INFO("[CollAllReduceMeshSmallCountExecutor][CalcTransportMemType]" \
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshSmallCountExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshSmallCountExecutor::Orchestrate(const OpParam& param,
    const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    ParseParam(param);
    algResResp_ = &algRes;
    GetStreamInfo(algRes);
    auto rtStream = param.stream.ptr();
    HCCL_PROFILER_ADD_STREAM(rtStream, param.tag, 0, algType_);
    CHK_RET(AddSubStreamToProfiling());

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, GetWorkflowMode());
        HCCL_PROFILER_ADD_OPDATA(param.tag, param.DataDes.count, param.inputPtr, param.outputPtr, \
            param.DataDes.dataType, param.root, algoAttr_.identifier);
        HCCL_PROFILER_ADD_GROUPRANK(algoAttr_.identifier, topoAttr_.userRankSize, topoAttr_.userRank);
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLREDUCE, param.tag,
            param.DataDes.count, param.DataDes.dataType, param.reduceType, algRes.cclInputMem.size(),
            algRes.cclOutputMem.size()));
    }

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
    } else {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
    }
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceMeshSmallCountExecutor][Orchestrate]errNo[0x%016llx]excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(RankConsistent::GetInstance().DelOpPara(param.tag));
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(param.tag);
        HCCL_PROFILER_DEL_OPDATA(param.tag);
        HCCL_PROFILER_DEL_GROUPRANK(param.tag);
    }
    HCCL_INFO("tag[%s], AllReduce executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshSmallCountExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    if (!CalcScratchMemFlag(totalSize_)) {
        execMem.scratchMem = execMem.outputMem;
    }

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
        (execMem.count * SIZE_TABLE[param.DataDes.dataType]) <= HCCL_SMALL_COUNT_256_KB, 1);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    CHK_RET(ActiveSlaveStreams(param.stream));

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };

    std::unique_ptr<ExecutorBase> outer2Executor;
    if (!topoMatcher_->GetExternalInputHcclDeterministic()) {
        outer2Executor.reset(new (std::nothrow) AllReduceReduceBcast(dispatcher_,
            reduceAttr, streamInfo_.ringStreams, streamInfo_.ringSignal, streamInfo_.ringSignalAux,
            outerCommInfo.localRank, outerCommInfo.localRankSize, topoAttr_.userRank, &opInfo));
    } else if (topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT) {
        if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            outer2Executor.reset(new (std::nothrow) AllReduceDoubling(dispatcher_, reduceAttr));
        } else {
            outer2Executor.reset(new (std::nothrow) AllReduceDoublingDirect(dispatcher_, reduceAttr, &opInfo));
        }
    } else {
        outer2Executor.reset(new (std::nothrow) AllReduceLocalReduceBcast(dispatcher_,
            reduceAttr, streamInfo_.ringStreams, streamInfo_.ringSignal, streamInfo_.ringSignalAux,
            outerCommInfo.localRank, outerCommInfo.localRankSize, topoAttr_.userRank, &opInfo));
    }
    CHK_SMART_PTR_NULL(outer2Executor);
    CHK_RET(outer2Executor->Prepare(execMem.inputMem, execMem.scratchMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, dataSegsSlice, 0));

    CHK_RET(
        outer2Executor->RegisterProfiler(
            (outerCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(outer2Executor, outerCommInfo));
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    }
    HCCL_INFO("all reduce small count executor run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMeshSmallCountExecutor", AllReduceMeshSmallCount, CollAllReduceMeshSmallCountExecutor);

} // namespace hccl