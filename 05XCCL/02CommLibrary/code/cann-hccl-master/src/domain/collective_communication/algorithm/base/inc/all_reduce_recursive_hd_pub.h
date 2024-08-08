/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_RECURSIVE_HD_PUB_H
#define ALL_REDUCE_RECURSIVE_HD_PUB_H

#include "recursive_halvingdoubling_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class AllReduceRecursiveHalvingDoubling : public RecursiveHalvingDoublingBase {
public:
    explicit AllReduceRecursiveHalvingDoubling(const HcclDispatcher dispatcher,
        const u64 reduceAttrBitMap);
    ~AllReduceRecursiveHalvingDoubling() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
    HcclResult RunAsyncStaged(const u32 rank, const u32 rankSize, const std::vector<LINK> &links,
        RunStage stage) override;

protected:
    HcclResult PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
private:
    HcclResult ReduceInPartOne(u32 rank, const std::vector<LINK> &links);

    HcclResult ReduceScatterInBlock(u32 rank, u32 rankSize, const std::vector<LINK> &links);

    HcclResult AllGatherInBlock(u32 rank, u32 rankSize, const std::vector<LINK> &links);

    HcclResult GatherInPartOne(u32 rank, const std::vector<LINK> &links);

    const u64 reduceAttr;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
}  // namespace hccl

#endif /* __ALL_REDUCE_RECURSIVE_HALVINGDOUBLING_PUB_H__ */
