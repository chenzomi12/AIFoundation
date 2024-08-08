/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_NHR_PUB_H
#define BROADCAST_NHR_PUB_H

#include "nonuniform_hierarchical_ring_base_pub.h"

namespace hccl {
class BroadcastNHR : public NHRBase {
public:
    explicit BroadcastNHR(const HcclDispatcher dispatcher);

    ~BroadcastNHR() override;

    HcclResult RunAsync(
        const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport> > &links) override;
    
protected:
private:
    HcclResult PrepareSlice(const u32 rank, const u32 rankSize);
};
}  // namespace hccl

#endif /* BROADCAST_NHR_PUB_H */
