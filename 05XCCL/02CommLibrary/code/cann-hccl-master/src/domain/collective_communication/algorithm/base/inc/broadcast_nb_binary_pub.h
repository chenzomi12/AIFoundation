/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef BROADCAST_NB_BINARY_PUB_H
#define BROADCAST_NB_BINARY_PUB_H
 
#include "nonuniform_bruck_base_pub.h"
 
namespace hccl {
class BroadcastNBBinary : public NBBase {
public:
    explicit BroadcastNBBinary(const HcclDispatcher dispatcher);
 
    ~BroadcastNBBinary() override;
 
    HcclResult RunAsync(
        const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport> > &links) override;
 
protected:
private:
    HcclResult RunBroadcastNBBinary(
        const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult Tx(const LINK &link, const std::vector<Slice> &txSlices);
    HcclResult Rx(const LINK &link, const std::vector<Slice> &rxSlices);
 
    bool hasData_ = false;
};
 
bool ShouldUseBinaryBroadcastOfNB(const u64 dataSize, const u32 rankSize, const u32 userRankSize,
    const float bandwidth);
}  // namespace hccl
 
#endif /* BROADCAST_NB_BINARY_PUB_H */