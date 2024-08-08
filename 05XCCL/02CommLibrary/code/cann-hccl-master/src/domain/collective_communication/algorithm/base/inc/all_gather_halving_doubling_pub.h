/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_HALVING_DOUBLING_PUB_H
#define ALL_GATHER_HALVING_DOUBLING_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class AllGatherHalvingDoubling : public ExecutorBase {
public:
    explicit AllGatherHalvingDoubling(u32 blockSize, const HcclDispatcher dispatcher,
                                      UserMemType hdInputMemType = UserMemType::OUTPUT_MEM,
                                      UserMemType hdOutputMemType = UserMemType::INPUT_MEM);
    ~AllGatherHalvingDoubling() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult RunAllGather(u32 rank, u32 stepNum,
                                const std::vector<LINK> &links);
    HcclResult Tx(const LINK &link, const Slice &txSlice);
    HcclResult Rx(const LINK &link, const Slice &rxSlice);

    u32 Log2(u32 antilogarithm) const;
    HcclResult CalculateSlices(u64 size, u32 sliceNum, std::vector<Slice> &sliceOut) const;
    HcclResult CalculateSlices(const std::vector<Slice> &inputSlices, u32 stepNum, u32 rank, SliceType type,
                                  std::vector<Slice> &sliceOut);

    u32 blockSize_;                      // binary block的size, 等于rank_size
    std::vector<Slice> txSlices_;        // 下标为step, 标识每个step的tx_size
    std::vector<Slice> rxSlices_;        // 下标为step, 标识每个step的rx_size
    u32 interRank_;
    u32 interRankSize_;
    UserMemType hdInputMemType_;   // 算法使用的input mem对应用户mem的类型
    UserMemType hdOutputMemType_;  // 算法使用的output mem对应用户mem的类型
};
}  // namespace hccl

#endif /* ALL_GATHER_HALVING_DOUBLING_PUB_H */
