/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BCAST_RECURSIVE_HALVINGDOUBLING_PUB_H
#define BCAST_RECURSIVE_HALVINGDOUBLING_PUB_H

#include "recursive_halvingdoubling_base_pub.h"

namespace hccl {
class BcastRecursiveHalvingDoubling : public RecursiveHalvingDoublingBase {
public:
    explicit BcastRecursiveHalvingDoubling(const HcclDispatcher dispatcher);
    ~BcastRecursiveHalvingDoubling() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport> > &links) override;

protected:
private:
    HcclResult SendData(const u32 destRank, const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult ReceiveData(const u32 destRank, const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult BroadcastInBlock(const u32 rank, const std::vector<std::shared_ptr<Transport> > &links);
    u32 GetRankIndexInBlock(const u32 rank) const;
    u32 GetRankIndexReal(const u32 rankInBlock) const;
    HcclResult OddNumberRankProcess(const u32 rank, const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult EvenNumberRankProcess(const u32 rank, const std::vector<std::shared_ptr<Transport> > &links);

    bool hasData_; // 表示该rank在运算步骤中是否有数据需要发送，该值可变
};
}  // namespace hccl

#endif /* BCAST_RECURSIVE_HALVINGDOUBLING_PUB_H */
