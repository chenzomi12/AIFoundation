/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_NHR_V1_PUB_H
#define BROADCAST_NHR_V1_PUB_H

#include "nonuniform_hierarchical_ring_v1_base_pub.h"

namespace hccl {

class BroadcastNHRV1 : public NHRV1Base {
public:
    explicit BroadcastNHRV1(const HcclDispatcher dispatcher);
    ~BroadcastNHRV1() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

private:
    DeviceMem scratch_; /* * 临时deviceMem */

    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult SetDefaultSlices(const u32 rank, const RingInfo &info);
    HcclResult CheckSlices(const u32 rank, const RingInfo &info);
    HcclResult RunScatterOnHorizontal(const u32 rank, const std::vector<LINK> &links, const RingInfo &info);
    HcclResult RunBroadcastOnVertical(const u32 rank, const std::vector<LINK> &links, const RingInfo &info);
    HcclResult RunAllGatherOnHorizontal(const u32 rank, const std::vector<LINK> &links, const RingInfo &info);
};
}  // hccl

#endif  /* BROADCAST_NHR_V1_PUB_H */
