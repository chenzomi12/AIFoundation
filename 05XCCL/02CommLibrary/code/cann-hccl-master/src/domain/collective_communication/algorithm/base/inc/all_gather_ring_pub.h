/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_RING_PUB_H
#define ALL_GATHER_RING_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class AllGatherRing : public ExecutorBase {
public:
    explicit AllGatherRing(const HcclDispatcher dispatcher);

    ~AllGatherRing() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    // 获取向该rank往前的第i个rank
    inline u32 ForwordRank(u32 rank, u32 rankSize, u32 preNum) const
    {
        return (rank + rankSize - preNum) % rankSize;
    }
    HcclResult RunAllGather(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices);

    HcclResult TxVector(const LINK &link, const std::vector<Slice> &txSlices);
    HcclResult RxVector(const LINK &link, const std::vector<Slice> &rxSlices);
    HcclResult Tx(const LINK &link, const Slice &txSlice);
    HcclResult Rx(const LINK &link, const Slice &rxSlice);

    // allgather ring chunk实现相关函数
    HcclResult RunAllGatherChunk(const u32 rank, const u32 rankSize, const std::vector<Slice> &outputSlices);
    HcclResult HeadAllGatherChunk(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices);
    HcclResult MidAllGatherChunk(u32 rank, u32 rankSize, u32 sliceIdx, const std::vector<Slice> &outputSlices);
    HcclResult TailAllGatherChunk(u32 rank, u32 rankSize, u32 sliceIdx, const std::vector<Slice> &outputSlices);
    HcclResult AllGatherSlicesPrep(u32 rankSize, u32 nicSize);

    // 迭代6新增加
    std::shared_ptr<Transport> linkLeft_;
    std::shared_ptr<Transport> linkRight_;
};
}  // namespace hccl

#endif /* ALL_GATHER_RING_PUB_H */