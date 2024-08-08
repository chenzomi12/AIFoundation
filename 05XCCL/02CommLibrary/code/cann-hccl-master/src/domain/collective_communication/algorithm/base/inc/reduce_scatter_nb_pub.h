/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_NB_PUB_H
#define REDUCE_SCATTER_NB_PUB_H

#include <cmath>

#include "nonuniform_bruck_base_pub.h"

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

#include "reducer_pub.h"
#include "sender_pub.h"
#include "comm_base_pub.h"

namespace hccl {
class ReduceScatterNB : public NBBase {
public:
    explicit ReduceScatterNB(const HcclDispatcher dispatcher, const u64 reduceAttrBitMap);
    ~ReduceScatterNB() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

private:
    const u64 reduceAttr_; /* 0x1:表示data_type + reduce_type支持inlinereduce  */

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;

    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult CheckSlices(const std::vector<Slice> &checkSlices, const u32 rankSize);

    HcclResult RunSrcReducerNB(const u32 step, const u32 nSlices,
                               u32 txSliceIdx, const u32 deltaSliceIndex,
                               const LINK linkRight, const u32 rank,
                               const u32 rankSize, const std::vector<Slice> &inputSlices,
                               const std::vector<Slice> &outputSlices);

    HcclResult RunDestReducerNB(const u32 step, const u32 nSteps,
                                const u32 nSlices, u32 rxSliceIdx,
                                const u32 deltaSliceIndex, const LINK linkLeft,
                                const u32 rank, const u32 rankSize,
                                const std::vector<Slice> &inputSlices,
                                const std::vector<Slice> &outputSlices);

    HcclResult RunReduceScatterNB(const u32 rank, const u32 rankSize, const std::vector<LINK> &links,
        const std::vector<Slice> &inputSlices, const std::vector<Slice> &outputSlices);
    HcclResult RunDestReducer(const LINK &link, const std::vector<Slice> &rxSlices,
        const std::vector<Slice> &rxSlicestemp);
    HcclResult RunSourceReducer(const LINK &link, const std::vector<Slice> &txSlices,
        const std::vector<Slice> &txSlicestemp);
};
} // hccl

#endif /* REDUCE_SCATTER_NB_PUB_H */