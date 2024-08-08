/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHER_EXECUTOR_H
#define COLL_ALLGATHER_EXECUTOR_H
#include "coll_comm_executor.h"
namespace hccl {
class CollAllGatherExecutor : public CollCommExecutor {

public:
    explicit CollAllGatherExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllGatherExecutor() = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceResponse& algRes) override;
protected:
    // AllGather Loop Executor公共接口
    virtual u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize);
    virtual bool IsHugeData(const u64 curSize);
    virtual u32 IsDataSplit(const u64 curSize);
    HcclResult RunLoop(const OpParam &param, const AlgResourceResponse &algRes);

    // 工具类
    HcclResult PrepareAllgatherSlice(u32 sliceNum, u64 inputMemSize, std::vector<Slice> &dataSegsSlice) const;

    HcclResult CalculateLevel1AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
        std::vector<std::vector<Slice>> multRingsSliceZero, std::vector<std::vector<Slice>> &multRingsSlice) const;

    HcclResult CalculateLevel2AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
        u32 level2RankSize, std::vector<Slice> dataSegsSlice, std::vector<Slice> &level0DataSlice) const;

    HcclResult AllGatherLevel2(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    bool DMAReduceFlag_{false}; // 是否DMA消减的标志
private:
    HcclResult RunLoopInner(const OpParam &param, ExecMem &execMem);
};

} // namespace hccl

#endif