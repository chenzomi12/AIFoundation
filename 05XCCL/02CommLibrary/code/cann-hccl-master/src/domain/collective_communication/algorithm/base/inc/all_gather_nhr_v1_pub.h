/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALL_GATHER_NHR_V1_PUB_H
#define ALL_GATHER_NHR_V1_PUB_H

#include "nonuniform_hierarchical_ring_v1_base_pub.h"

#include "all_gather_ring_pub.h"

namespace hccl {
class AllGatherNHRV1 : public NHRV1Base {
public:
    explicit AllGatherNHRV1(const HcclDispatcher dispatcher);

    ~AllGatherNHRV1() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
    HcclResult RunAllGatherOnHorizontal(u32 rank, const std::vector<LINK> &links, const RingInfo &info);
    HcclResult RunAllGatherOnVertical(u32 rank, const std::vector<LINK> &links, const RingInfo &info);

private:
};
}  // namespace hccl
#endif /* ALL_GATHER_NHR_V1_PUB_H */