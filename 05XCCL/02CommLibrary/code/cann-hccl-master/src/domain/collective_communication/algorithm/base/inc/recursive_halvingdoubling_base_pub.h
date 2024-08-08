/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RECURSIVE_HALVINGDOUBLING_BASE_PUB_H
#define RECURSIVE_HALVINGDOUBLING_BASE_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class RecursiveHalvingDoublingBase : public ExecutorBase {
public:
    explicit RecursiveHalvingDoublingBase(const HcclDispatcher dispatcher);
    ~RecursiveHalvingDoublingBase() override;

protected:
    HcclResult CalcPartOneSizeAndBlockSize(const u32 rankSize);
    HcclResult BuildSubLinks(const std::vector<LINK> &links, std::vector<LINK> &subLinks,
                               u32 rankSize) const;
    HcclResult CalculateSlices(u64 dataBytes) const;

    u32 blockSize_;
    u32 part1Size_;
    u32 round_;

private:
};
}  // hccl

#endif  /* RECURSIVE_HALVINGDOUBLING_BASE_PUB_H */
