/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_deter_executor.h"

namespace hccl {

CollReduceScatterDeterExecutor::CollReduceScatterDeterExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

void CollReduceScatterDeterExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    meshSinglePlane_ = false;

    // 是否需要scratch memory 选中确定性计算Executor，其他条件必定满足，只需区分是否为图模式
    scratchMemFlag_ = (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

HcclResult CollReduceScatterDeterExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) { // 确定性计算只有图模式需要scratch memory
        scratchMemSize = totalSize_ + CCE_REDUCE_ALIGN_FACTOR * CCE_REDUCE_ALIGN_SIZE;
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO("[CollReduceScatterDeterExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%u]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        totalStreamNum = topoAttr_.deviceNumPerAggregation - 1U;
    } else {
        totalStreamNum = topoAttr_.deviceNumPerAggregation;
    }
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterDeterExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::CCL_OUTPUT;
        }
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::PARAM_OUTPUT;
        }
    }
    HCCL_INFO("[CollReduceScatterDeterExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDeterExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = meshSinglePlane_;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterDeterExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    u64 maxCountPerLoop = (inCCLbufferSize_ - HCCL_MIN_SLICE_ALIGN_910B * topoAttr_.deviceNumPerAggregation) /
        unitSize / (topoAttr_.deviceNumPerAggregation - 1);
    maxCountPerLoop = maxCountPerLoop / HCCL_MIN_SLICE_ALIGN_910B;
    maxCountPerLoop = maxCountPerLoop * HCCL_MIN_SLICE_ALIGN_910B;
    return maxCountPerLoop;
}

bool CollReduceScatterDeterExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollReduceScatterDeterExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    bool smallData = curSize <= HCCL_SMALL_COUNT_32_KB;
    return smallData;
}

HcclResult CollReduceScatterDeterExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::unique_ptr<ExecutorBase> outerExecutor;

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    CHK_RET(ActiveSlaveStreams(param.stream));

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType};

    if ((param.DataDes.count * unitSize > HCCL_SMALL_COUNT_32_KB) ||
        (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ||
        ((topoAttr_.deviceNumPerAggregation != DEVICE_EIGHT) && (topoAttr_.deviceNumPerAggregation != DEVICE_FOUR))) {
        outerExecutor.reset(new (std::nothrow) ReduceScatterLocalReduce(dispatcher_, reduceAttr,
            streamInfo_.ringStreams, streamInfo_.ringSignal, streamInfo_.ringSignalAux, topoAttr_.userRank, &opInfo));
    } else {
        outerExecutor.reset(new (std::nothrow) ReduceScatterHDStage(dispatcher_, reduceAttr, streamInfo_.ringStreams,
            streamInfo_.ringSignal, streamInfo_.ringSignalAux, topoAttr_.userRank, &opInfo));
    }

    CHK_SMART_PTR_NULL(outerExecutor);
    CHK_RET(outerExecutor->Prepare(execMem.inputMem, execMem.scratchMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, dataSegsSlice, 0));

    CHK_RET(outerExecutor->RegisterProfiler(
        (outerCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(outerExecutor, outerCommInfo));
    HCCL_INFO("reducescatter mesh deter run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterDeterExecutor", ReduceScatterDeter, CollReduceScatterDeterExecutor);
}