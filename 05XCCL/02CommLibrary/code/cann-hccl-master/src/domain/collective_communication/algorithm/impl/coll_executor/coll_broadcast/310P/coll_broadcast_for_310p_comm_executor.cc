/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_broadcast_for_310p_comm_executor.h"

namespace hccl {
CollBroadcastFor310PCommExecutor::CollBroadcastFor310PCommExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollBroadcastFor310PCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}


HcclResult CollBroadcastFor310PCommExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollBroadcastFor310PCommExecutor][CalcOuterCommInfo]tag[%s ]start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollBroadcastFor310PCommExecutor][CalcOuterCommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastFor310PCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    std::unique_ptr<ExecutorBase> executor;

    executor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
    CHK_SMART_PTR_NULL(executor);
    // 获取root
    u32 rootRank = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRank));

    CHK_RET(executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
                              param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, param.root));

    u32 rankSize = outerCommInfo.localRankSize;
    CHK_RET(executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(executor, outerCommInfo));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadCastCommFor310P", BroadcastFor310PComm, CollBroadcastFor310PCommExecutor);
} // namespace hccl