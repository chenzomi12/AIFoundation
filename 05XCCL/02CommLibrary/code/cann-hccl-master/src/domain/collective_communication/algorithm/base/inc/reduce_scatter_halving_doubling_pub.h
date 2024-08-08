/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_HALVING_DOUBLING_PUB_H
#define REDUCE_SCATTER_HALVING_DOUBLING_PUB_H

#include "executor_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceScatterHalvingDoubling : public ExecutorBase {
public:
    explicit ReduceScatterHalvingDoubling(const u32 blockSize, const HcclDispatcher dispatcher,
        const u64 reduceAttrBitMap, const UserMemType hdInputMemType = UserMemType::OUTPUT_MEM,
        const UserMemType hdOutputMemType = UserMemType::INPUT_MEM);

    ~ReduceScatterHalvingDoubling() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult RunReduceScatter(const u32 rank, const u32 stepNum,
                                    const HcclDispatcher dispatcher,
                                    const std::vector<LINK> &links);
    HcclResult RunSourceReducer(const LINK &link, const Slice &txSlice);
    HcclResult RunDestRducer(const LINK &link, const HcclDispatcher dispatcher,
                                 const Slice &rxSlice, const DstMemType reduceDst);

    u32 GetBlockStep(u32 blocksize) const;
    HcclResult CalculateSlices(const u64 size, const u32 sliceNum, std::vector<Slice> &slicesOut);
    HcclResult CalcStepSlices(const std::vector<Slice> &inputSlices,
        const u32 stepNum, const u32 rank, const SliceType type, std::vector<Slice> &slicesOut);
    u32 blockSize_;
    std::vector<Slice> txSlices_; // 下标为step, 标识每个step的tx_size
    std::vector<Slice> rxSlices_; // 下标为step, 标识每个step的tx_size
    const u64 reduceAttr_;               /* 0x1:表示data_type + reduce_type支持inlinereduce  */

    UserMemType hdInputMemType_;   // 算法使用的input mem对应用户mem的类型
    UserMemType hdOutputMemType_;  // 算法使用的output mem对应用户mem的类型

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
};
}  // namespace hccl

#endif /* REDUCE_SCATTER_HALVING_DOUBLING_PUB_H */
