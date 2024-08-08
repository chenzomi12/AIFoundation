/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_RING_PUB_H
#define BROADCAST_RING_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class BroadcastRing : public ExecutorBase {
public:
    explicit BroadcastRing(const HcclDispatcher dispatcher);
    ~BroadcastRing() override;

    HcclResult RunAsync(
        const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport> > &links) override;

protected:
private:
    std::shared_ptr<Transport> linkLeft_;
    std::shared_ptr<Transport> linkRight_;

    DeviceMem scratch_; /* * 临时deviceMem */
};
}  // namespace hccl

#endif /* BROADCAST_RING_PUB_H */
