/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NONUNIFORM_HIERARCHICAL_RING_V1_BASE_PUB_H
#define NONUNIFORM_HIERARCHICAL_RING_V1_BASE_PUB_H

#include <cmath>
#include "executor_base_pub.h"

namespace hccl {

class RingInfo {
public:
    explicit RingInfo(u32 rankSize);
    ~RingInfo();

    u32 GetRankSize() const;
    u32 GetRankOffset() const;
    u32 GetSqrtRankSize() const;
    u32 GetRowSize() const;
    u32 GetColSize() const;
    u32 GetVIndex(u32 rank) const;
    u32 GetHIndex(u32 rank) const;
    u32 GetVSizeByRank(u32 rank) const;
    u32 GetVSizeByHIndex(u32 hIndex) const;
    u32 GetHSizeByRank(u32 rank) const;
    u32 GetHSizeByVIndex(u32 vIndex) const;
    u32 GetRank(u32 vIndex, u32 hIndex) const;

    u32 threshold = 5;
private:
    u32 rankSize_{};
    u32 sqrtRankSize_{};
    u32 extraColSize_{};
    u32 extraRowSize_{};
    u32 rankOffset_{};
    u32 colSize_{};
    u32 rowSize_{};
};

class NHRV1Base : public ExecutorBase {
public:
    explicit NHRV1Base(const HcclDispatcher dispatcher);
    ~NHRV1Base() override;

    static RingInfo GetRingInfo(u32 rankSize);

private:
};
}  // hccl

#endif  /* NONUNIFORM_HIERARCHICAL_RING_V1_BASE_PUB_H */
