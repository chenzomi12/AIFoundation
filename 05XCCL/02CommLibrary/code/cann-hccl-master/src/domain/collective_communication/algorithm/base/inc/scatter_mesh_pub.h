/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_MESH_PUB_H
#define SCATTER_MESH_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class ScatterMesh : public ExecutorBase {
public:
    explicit ScatterMesh(const HcclDispatcher dispatcher,
                         const u32 interRank, const u32 interRankSize);

    ~ScatterMesh() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize,
                                   const std::vector<std::shared_ptr<Transport> > &links) override;

protected:
private:
    HcclResult RunSendScatter(const u32 dstRank, const Slice &slice, const std::vector<LINK> &links);
    HcclResult RunRecvScatter(const u32 srcRank, const Slice &slice, const std::vector<LINK> &links);
    void PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const;

    u32 interRank_;       // comm内的rank排序
    u32 interRankSize_; // 本comm内ranksize总数
};
}  // namespace hccl

#endif /* * SCATTER_MESH_PUB_H */
