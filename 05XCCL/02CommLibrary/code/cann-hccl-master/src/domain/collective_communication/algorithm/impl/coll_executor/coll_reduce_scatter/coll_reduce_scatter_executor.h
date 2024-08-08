/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_EXECUTOR_H
#define COLL_REDUCESCATTER_EXECUTOR_H
#include "coll_comm_executor.h"

namespace hccl {

constexpr u64 CCE_REDUCE_ALIGN_FACTOR = 2; // cce reduce数据大小32字节对齐  2是指前后各有

class CollReduceScatterExecutor : public CollCommExecutor {
public:
    explicit CollReduceScatterExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollReduceScatterExecutor() = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceResponse& algRes) override;
protected:
    // ReduceScatter Loop Executor公共接口
    virtual u64 CalcLoopMaxCount(const u32 unitSize);
    virtual bool IsHugeData(const u64 curSize);
    virtual bool IsSmallData(const u64 totalSize, const u64 curSize);
    virtual u32 CalcDataSplit(const u64 curSize);
    virtual HcclResult RunLoop(const OpParam &param, const AlgResourceResponse &algRes);

    // 工具类
    std::vector<std::vector<Slice>> ReduceScatterRingSlicePrepare(u32 ringNum, u32 sliceNum,
        bool useInlineReduce, DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice, const std::string &tag);

    bool CCLMemSlice_{true};     // 每次Loop是否需要对CCLMem进行切片
    bool DMAReduceFlag_{false};  // 是否DMA消减
    bool scratchMemFlag_{false}; // 是否需要申请scratch memory，不需要申请则传入outputmem为scratchmem
    u64 totalSize_{0};           // 总数据量

private:
    HcclResult RunLoopInner(const OpParam &param, const ReduceType &reduceType, ExecMem &execMem);
};

} // namespace hccl

#endif