/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_comm_executor.h"

namespace hccl {

CollBroadcastCommExecutor::CollBroadcastCommExecutor(const HcclDispatcher dispatcher,
                                std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
}


HcclResult CollBroadcastCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastCommExecutor::CalcCombinedCommInfo(TransportMemType inputType,
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

HcclResult CollBroadcastCommExecutor::CalcStreamNum(u32& streamNum)
{
    // 只传递从流数量
    streamNum = 0;
    HCCL_INFO("[CollBroadcastCommExecutor][CalcStreamNum]tag[%s] streamNum_ is [%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_COMBINE, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(COMM_COMBINE, 0);

    std::unique_ptr<ExecutorBase> executor;
    u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    if (UseInterServerNHRAlgo(algType_)) {
        if (curSize <= NHR_BCAST_SMALL_SIZE) {
            executor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
        } else {
            executor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
        }
        HCCL_INFO("broadcast comm: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        executor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
        HCCL_INFO("broadcast comm: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        if (ShouldUseBinaryBroadcastOfNB(curSize, combinedCommInfo.localRankSize, topoAttr_.userRankSize,
                topoAttr_.deviceNumPerAggregation)) {
            executor.reset(new (std::nothrow) BroadcastNBBinary(dispatcher_));
        } else {
            executor.reset(new (std::nothrow) BroadcastNB(dispatcher_));
        }
        HCCL_INFO("broadcast comm: using nonuniform-bruck algo inter-server.");
    } else {
        executor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
        HCCL_INFO("broadcast comm: using ring algo inter-server.");
    }
    CHK_SMART_PTR_NULL(executor);

    // 获取root
    u32 rootRank = 0;
    CHK_RET(GetRankByUserRank(COMM_COMBINE, COMM_INDEX_0, param.root, rootRank));

    CHK_RET(executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
                param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, rootRank));
    CHK_RET(RunTemplate(executor, combinedCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadCastComm", BroadcastComm, CollBroadcastCommExecutor);

} // namespace hccl