/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_NHR_V1_PUB_H
#define ALL_REDUCE_NHR_V1_PUB_H

#include "nonuniform_hierarchical_ring_v1_base_pub.h"
#include "all_gather_ring_pub.h"
#include "all_reduce_ring_pub.h"
#include "reduce_scatter_ring_pub.h"


namespace hccl {
class AllReduceNHRV1 : public NHRV1Base {
public:
    explicit AllReduceNHRV1(const HcclDispatcher dispatcher, const u64 reduceAttrBitMap);

    ~AllReduceNHRV1() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
    HcclResult RunAsyncStaged(const u32 rank, const u32 rankSize, const std::vector<LINK> &links,
        RunStage stage) override;

protected:
    HcclResult PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
private:
    HcclResult RunReduceScatterOnHorizontal(u32 rank, const std::vector<LINK> &links, const RingInfo &info);
    HcclResult RunAllReduceOnVertical(u32 rank, const std::vector<LINK> &links, const RingInfo &info);
    HcclResult RunAllGatherOnHorizontal(u32 rank, const std::vector<LINK> &links, const RingInfo &info);
    HcclResult CalcHSlicesAndLinks(const u32 rank, const std::vector<LINK> &links, const RingInfo &info,
        std::vector<LINK> &hLinks, std::vector<Slice> &hSlices);
    HcclResult CalcVSlicesAndLinks(const u32 rank, const std::vector<LINK> &links, const RingInfo &info,
        std::vector<LINK> &vLinks, std::vector<Slice> &vSlices);
    HcclResult RunReduceScatterBrokenRing(const u32 rank, const std::vector<LINK> &links,
        const std::vector<Slice> &slices);
    HcclResult RunAllGatherBrokenRing(const u32 rank, const std::vector<LINK> &links,
        const std::vector<Slice> &slices);
    const u64 reduceAttr_;
};
}  // namespace hccl
#endif /* ALL_REDUCE_NHR_V1_PUB_H */