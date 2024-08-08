/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_comm_executor.h"

namespace hccl {
CollAllGatherCommExecutor::CollAllGatherCommExecutor(const HcclDispatcher dispatcher,
                                                     std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllGatherCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherCommExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherCommExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherCommExecutor::CalcCombinedCommInfo(TransportMemType inputType, TransportMemType outputType,
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

HcclResult CollAllGatherCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_COMBINE, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(COMM_COMBINE, COMM_INDEX_0);

    // 构造ring algorithm对应的all_gather实例
    std::unique_ptr<ExecutorBase> executor;
    if (UseInterServerNHRAlgo(algType_)) {
        executor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
        HCCL_INFO("algather comm: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        executor.reset(new (std::nothrow) AllGatherNHRV1(dispatcher_));
        HCCL_INFO("algather comm: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        executor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
        HCCL_INFO("algather comm: using nonuniform-bruck algo inter-server.");
    } else {
        executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        HCCL_INFO("algather comm: ring algo inter-server.");
    }
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID));

    CHK_RET(RunTemplate(executor, combinedCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherComm", AllGatherComm, CollAllGatherCommExecutor);

} // namespace hccl