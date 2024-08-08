/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GATHER_STAR_PUB_H
#define GATHER_STAR_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class GatherStar : public ExecutorBase {
public:
    explicit GatherStar(const HcclDispatcher dispatcher, u32 userRank);

    ~GatherStar() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport>> &links) override;
protected:
private:
    HcclResult RunRecvGather(const u32 srcRank, const Slice &slice,
        const std::vector<std::shared_ptr<Transport>> &links);
    HcclResult RunSendGather(const u32 dstRank, const Slice &slice,
        const std::vector<std::shared_ptr<Transport>> &links);
    void PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const;
    HcclResult ExecuteBarrierSrcRank(std::shared_ptr<Transport> link, Stream &stream) const;

    const std::vector<Stream> Streams_; /** 多steam**/
    u32 userRank_;
};
} // namespace hccl

#endif /* GATHER_STAR_PUB_H */
