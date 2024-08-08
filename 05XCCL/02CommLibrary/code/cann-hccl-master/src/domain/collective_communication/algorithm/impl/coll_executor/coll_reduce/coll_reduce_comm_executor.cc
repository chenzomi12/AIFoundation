/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_reduce_comm_executor.h"

namespace hccl {

CollReduceCommExecutor::CollReduceCommExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CalcCombinedCommInfo(inputType, outputType, opTransport);
    return HCCL_SUCCESS;
}

HcclResult CollReduceCommExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollReduceCommExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceCommExecutor::CalcCombinedCommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_COMBINE, CommType::COMM_TAG_MAX);
    commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE], inputType, outputType));

    return HCCL_SUCCESS;
}

HcclResult CollReduceCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_COMBINE, 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(COMM_COMBINE, 0);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<ExecutorBase> executor;
    executor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
    HCCL_INFO("Reduce comm: using ring algo inter-server.");
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

REGISTER_EXEC("ReduceComm", ReduceComm, CollReduceCommExecutor);

} // namespace hccl