/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_RING_PUB_H
#define REDUCE_SCATTER_RING_PUB_H

#include "executor_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterRing : public ExecutorBase {
public:
    explicit ReduceScatterRing(const HcclDispatcher dispatcher,
        const u64 reduceAttrBitMap);
    ~ReduceScatterRing() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult RunReduceScatter(const u32 rank, const u32 rankSize, const std::vector<Slice> &inputSlices,
                                    const std::vector<Slice> &outputSlices);
    HcclResult RunVectorSourceReducer(const LINK &link, const std::vector<Slice> &txSlices,
                                      const std::vector<Slice> &txSlicetemp);
    HcclResult RunVectorDestRducer(const LINK &link, const std::vector<Slice> &rxSlices,
                                   const std::vector<Slice> &rxSlicetemp);
    HcclResult RunSourceReducer(const LINK &link, const Slice &txSlice, const Slice &txSlicetemp);
    HcclResult RunDestRducer(const LINK &link, const Slice &rxSlice, const Slice &rxSlicetemp);

    LINK linkLeft_;
    LINK linkRight_;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;

    const u64 reduceAttr_; /* 0x1:表示data_type + reduce_type支持inlinereduce  */

    // reduce scatter ring chunk实现相关函数
    HcclResult RunReduceScatterChunk(const u32 rank, const u32 rankSize,
                                       const std::vector<Slice> &inputSlices, const std::vector<Slice> &outputSlices);
    HcclResult HeadReduceScatterChunk(u32 rank, u32 rankSize, const std::vector<Slice> &inputSlices,
                                        const std::vector<Slice> &outputSlices);
    HcclResult MidReduceScatterChunk(u32 rank, u32 rankSize, u32 sliceIdx, const std::vector<Slice> &inputSlices,
                                       const std::vector<Slice> &outputSlices);
    HcclResult TailReduceScatterChunk(u32 rank, u32 rankSize, u32 sliceIdx, const std::vector<Slice> &inputSlices,
                                        const std::vector<Slice> &outputSlices);
    HcclResult ReduceScatterSlicesPrep(u32 rankSize, u32 nicSize);
};
}  // namespace hccl

#endif /* REDUCE_SCATTER_RING_PUB_H */
