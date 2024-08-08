/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLREDUCE_COMM_EXECUTOR_H
#define COLL_ALLREDUCE_COMM_EXECUTOR_H

#include "coll_comm_executor.h"

namespace hccl {
class CollAllReduceExecutor : public CollCommExecutor {
public:
    CollAllReduceExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllReduceExecutor() = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceResponse& algRes) override;
protected:
    /* *************** 算法编排 *************** */
    // AllReduce Loop Executor公共接口
    virtual u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize);
    virtual bool IsHugeData(const u64 curSize);
    virtual bool IsSmallData(const u64 totalSize, const u64 curSize);
    HcclResult RunLoop(const OpParam &param, const AlgResourceResponse &algRes);
    HcclResult AvoidSubgraphLoop(const OpParam &param, const AlgResourceResponse &algRes);

    // 工具类
    u64 GetSliceNum(const u64 totalSize);
    bool IsAllReduceSmallData(u64 size);
    HcclResult PrepareSliceDataWithAlignSize(u64 totalSize, u32 sliceNum,
        u64 piplineOffset, std::vector<Slice>& dataSlice, u64 alignSize);
    HcclResult PrepareAivBuffers(u32 rankSize, u32 rankId, u32 rankOffset,
        DeviceMem &inputMem, DeviceMem &outputMem, std::vector<LINK> &links, void **dataBuffers, void **flagBuffers,
        UserMemType dataMemType, UserMemType flagMemType, u32 dataMemOffset, u32 flagMemOffset);

    bool CCLMemSlice_{true};    // 每次Loop是否需要对CCLMem进行切片
    bool DMAReduceFlag_{false}; // 是否DMA消减
private:
    HcclResult RunLoopInner(const OpParam &param, const ReduceType &reduceType, ExecMem &execMem);
};

} // namespace hccl

#endif