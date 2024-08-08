/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_DOUBLING_DIRECT_PUB_H
#define ALL_REDUCE_DOUBLING_DIRECT_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class AllReduceDoublingDirect : public ExecutorBase {
public:
    explicit AllReduceDoublingDirect(
        const HcclDispatcher dispatcher, const u64 reduceAttrBitMap, const HcomCollOpInfo *opInfo);
    ~AllReduceDoublingDirect() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    const u64 reduceAttr_;

    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunAllReduce(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunInlineReduce(hccl::DeviceMem &dst, const hccl::DeviceMem &src, const LINK &link);
    const HcomCollOpInfo *opInfo_;
};
}  // namespace hccl

#endif /* ALL_REDUCE_DOUBLING_DIRECT_PUB_H */