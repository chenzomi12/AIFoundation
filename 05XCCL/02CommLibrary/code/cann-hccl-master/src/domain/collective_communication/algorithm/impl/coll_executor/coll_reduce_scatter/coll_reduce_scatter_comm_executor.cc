/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_comm_executor.h"

namespace hccl {

CollReduceScatterCommExecutor::CollReduceScatterCommExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

void CollReduceScatterCommExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        (topoAttr_.deviceType == DevType::DEV_TYPE_910B || topoAttr_.deviceType == DevType::DEV_TYPE_910_73) &&
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
        IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

HcclResult CollReduceScatterCommExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = inCCLbufferSize_ + CCE_REDUCE_ALIGN_FACTOR * CCE_REDUCE_ALIGN_SIZE;
        } else {
            scratchMemSize = totalSize_ + CCE_REDUCE_ALIGN_FACTOR * CCE_REDUCE_ALIGN_SIZE;
        }
    } else {
        scratchMemSize = 0U;
    }

    HCCL_INFO("[CollReduceScatterCommExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%u]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterCommExecutor::CalcTransportMemType(TransportMemType &inputType,
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
    HCCL_INFO("[CollReduceScatterCommExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterCommExecutor::CalcCombinedCommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_COMBINE, CommType::COMM_TAG_MAX);
    if (UseInterServerNHRAlgo(algType_)) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    } else if (UseInterServerNHRV1Algo(algType_)) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
    } else if (UseInterServerNBAlgo(algType_)) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else {
        commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE], inputType, outputType));

    return HCCL_SUCCESS;
}

u64 CollReduceScatterCommExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterCommExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                    (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

HcclResult CollReduceScatterCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_COMBINE, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(COMM_COMBINE, COMM_INDEX_0);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    // 构造ring algorithm对应的reduce-scatter实例
    std::unique_ptr<ExecutorBase> executor;
    if (UseInterServerNHRAlgo(algType_)) {
        executor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter comm: using nhr algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType));
    } else if (UseInterServerNHRV1Algo(algType_)) {
        executor.reset(new (std::nothrow) ReduceScatterNHRV1(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter comm: using nhr_v1 algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType));
        CHK_RET(RunTemplate(executor, combinedCommInfo));
    } else if (UseInterServerNBAlgo(algType_)) {
        executor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter comm: using nonuniform-bruck algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType));
        CHK_RET(RunTemplate(executor, combinedCommInfo));
    } else {
        executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter comm: using ring algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType));
        CHK_RET(RunTemplate(executor, combinedCommInfo));
        // 将cclInBuffer中与userRank_对应的部分拷贝至cclOutBuffer
        u64 dataSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        DeviceMem srcMem = execMem.inputMem.range(dataSize * topoAttr_.userRank, dataSize);
        DeviceMem dstMem = execMem.outputMem.range(0, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterComm", ReduceScatterComm, CollReduceScatterCommExecutor);
}