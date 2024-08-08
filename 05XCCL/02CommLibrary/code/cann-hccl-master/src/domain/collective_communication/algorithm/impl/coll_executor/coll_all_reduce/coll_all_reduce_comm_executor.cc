/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_comm_executor.h"

namespace hccl {

CollAllReduceCommExecutor::CollAllReduceCommExecutor(const HcclDispatcher dispatcher,
                                                     std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllReduceCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceCommExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllReduceCommExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceCommExecutor::CalcCombinedCommInfo(TransportMemType inputType,
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

bool CollAllReduceCommExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollAllReduceCommExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    bool smallData = IsAllReduceSmallData(curSize);
    return smallData;
}

HcclResult CollAllReduceCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_COMBINE, 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(COMM_COMBINE, 0);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<ExecutorBase> executor;
    if (UseInterServerNHRAlgo(algType_)) {
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType]; // 单位 byte
        if (curSize <= NHR_ALLREDUCE_SMALL_SIZE) {
            executor.reset(new (std::nothrow) AllReduceNHROneshot(dispatcher_, reduceAttr));
        } else {
            executor.reset(new (std::nothrow) AllReduceNHR(dispatcher_, reduceAttr));
        }
        HCCL_INFO("allreduce comm: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        executor.reset(new (std::nothrow) AllReduceNHRV1(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce comm: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        executor.reset(new (std::nothrow) AllReduceNB(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce comm: using nonuniform-bruck algo inter-server.");
    } else {
        executor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce comm: using ring algo inter-server.");
    }
    CHK_SMART_PTR_NULL(executor);

    u32 rankSize = combinedCommInfo.localRankSize;
    CHK_RET(executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType,
        OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

    CHK_RET(executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        combinedCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(executor, combinedCommInfo));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceComm", AllReduceComm, CollAllReduceCommExecutor);

} // namespace hccl