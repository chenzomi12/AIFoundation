/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BATCH_SEND_RECV_EXECUTOR_H
#define COLL_BATCH_SEND_RECV_EXECUTOR_H

#include "coll_comm_executor.h"

namespace hccl {
class CollBatchSendRecvExecutor : public CollCommExecutor {
public:
    CollBatchSendRecvExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollBatchSendRecvExecutor() = default;
    HcclResult Orchestrate(const OpParam& param, const AlgResourceResponse& algRes) override;
    // 增量建链资源计算接口
    HcclResult CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest& resourceRequest) override;
private:
    /* *************** 资源计算 *************** */
    void ParseParam(const OpParam& param) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */
    u64 CalcSendLoopMaxCount(DeviceMem& inCCLBuffer, const u32 unitSize);
    u64 CalcRecvLoopMaxCount(DeviceMem& outCCLBuffer, const u32 unitSize);
    HcclResult GetSendRecvInfo(HcclSendRecvItem* itemPtr);
    HcclResult RunLoop(const OpParam &param, const AlgResourceResponse &algRes, u32 index);
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;

private:
    std::set<u32> commTargetUserRankSet_;
    u32 remoteUserRank_;
    HcclSendRecvType sendRecvType_;
};
} // namespace hccl

#endif