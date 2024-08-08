/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALL_GATHER_NHR_PUB_H
#define ALL_GATHER_NHR_PUB_H

#include "nonuniform_hierarchical_ring_base_pub.h"
#include "comm_base_pub.h"

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

namespace hccl {
class AllGatherNHR : public NHRBase {
public:
    explicit AllGatherNHR(const HcclDispatcher dispatcher);

    ~AllGatherNHR() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

    HcclResult RunAsyncWithReorder(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);

protected:
private:
    HcclResult RunAllGather(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices,
        const std::vector<LINK> &links);

    HcclResult Tx(const LINK &link, std::vector<Slice> &txSlices);
    HcclResult Rx(const LINK &link, std::vector<Slice> &rxSlices);

    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep &stepInfo) override;
};
} // namespace hccl

#endif /* ALL_GATHER_NHR_PUB_H */