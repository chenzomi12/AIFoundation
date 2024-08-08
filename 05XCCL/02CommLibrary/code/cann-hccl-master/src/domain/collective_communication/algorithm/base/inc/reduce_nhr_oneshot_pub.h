/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_NHR_ONESHOT_PUB_H
#define REDUCE_NHR_ONESHOT_PUB_H

#include <cmath>

#include "nonuniform_hierarchical_ring_base_pub.h"
#include "executor_base_pub.h"

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

#include "reducer_pub.h"
#include "sender_pub.h"
#include "comm_base_pub.h"

namespace hccl {
class ReduceNHROneshot : public NHRBase {
public:
    explicit ReduceNHROneshot(const HcclDispatcher dispatcher, const u64 reduceAttrBitMap);
    ~ReduceNHROneshot() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);

    HcclResult RunReduceNHROneshot(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);

    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep &stepInfo) override;

    const u64 reduceAttr_; /* 0x1:表示data_type + reduce_type支持inlinereduce  */

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
} // hccl

#endif /* REDUCE_NHR_ONESHOT_PUB_H */