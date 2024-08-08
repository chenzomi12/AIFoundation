/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_RECURSIVE_HD_PUB_H
#define REDUCE_RECURSIVE_HD_PUB_H

#include "recursive_halvingdoubling_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceRecursiveHalvingDoubling : public RecursiveHalvingDoublingBase {
public:
    explicit ReduceRecursiveHalvingDoubling(const HcclDispatcher dispatcher,
        const u64 reduceAttrBitMap);
    ~ReduceRecursiveHalvingDoubling() override;

    HcclResult RunAsync(
        const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport> > &links) override;

protected:
private:
    HcclResult ReduceInPartOne(u32 rank, const std::vector<LINK> &links);

    HcclResult ReduceScatterInBlock(u32 rank, u32 rankSize, const std::vector<LINK> &links);

    HcclResult GatherInBlock(u32 rank, u32 rankSize, const std::vector<LINK> &links);

    HcclResult CalculateStepSlices(const std::vector<Slice> &inputSlices, u32 stepNum, u32 rank, SliceType type,
                                std::vector<Slice> &sliceOut);
    HcclResult BuildRootSubLinks(const std::vector<LINK> &links, std::vector<LINK> &subLinks,
                                 u32 rankSize) const;
    std::vector<Slice> txSlices_;
    std::vector<Slice> rxSlices_;
    const u64 reduceAttr;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
}  // namespace hccl

#endif /* __REDUCE_RECURSIVE_HALVINGDOUBLING_PUB_H__ */
