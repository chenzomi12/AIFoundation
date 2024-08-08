/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BCAST_HALVINGDOUBLING_PUB_H
#define BCAST_HALVINGDOUBLING_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class BcastHalvingDoubling : public ExecutorBase {
public:
    explicit BcastHalvingDoubling(const HcclDispatcher dispatcher);
    ~BcastHalvingDoubling() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport> > &links) override;

protected:
private:
    // 根据rank号和ranksize计算HD算法需要的相关参数
    void PrepareBlockParams(const u32 rank, const u32 rankSize);
    // 计算与root相关的参数
    void PrepareRootBlockParam(const u32 rootRank, const u32 rankSize);

    HcclResult BcastInRootBlock(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult GatherBetweenBlocksRootBlock(const u32 rank, const u32 rankSize,
                                                  const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult GatherBetweenBlocksHigherBlock(const u32 rank, const u32 rankSize,
                                                    const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult GatherBetweenBlocksLowerBlock(const u32 rank, const u32 rankSize,
                                                   const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult ReceiveData(const std::shared_ptr<Transport> &link, u64 dataCount);
    HcclResult SendData(const std::shared_ptr<Transport> &link, u64 dataCount);

    inline u64 DataSize(u64 dataCount) const
    {
        return dataCount * DataUnitSize(dataType_);
    }

    u32 rootBlockOffset_;  //  root rank 所在的block起始位置
    u32 rootBlockSize_;    // root所在block的大小
    u32 stepsInBlock_;     // rank 所在block需要执行操作的步骤(所在block组的幂数)
    u32 myBlockSize_;      // 本rank所在的block内总rank数
    u32 rankInMyBlock_;   // 本rank所在的block内的排序
    u32 myBlockOffset_;    // 本rank所在block的起始rank在总体rank中的偏移(所有高阶blocksize的和)
    u32 higherBlockSize_;  // 高阶block组的rank数，如果没有，则为0
    u32 lowerBlockSize_;   // 如果没有则为0
    u32 highestStep_;       // 所有rank的最高阶
};
}  // namespace hccl

#endif /* __BCAST_HAVLINGDOUBLING_PUB_H__ */
