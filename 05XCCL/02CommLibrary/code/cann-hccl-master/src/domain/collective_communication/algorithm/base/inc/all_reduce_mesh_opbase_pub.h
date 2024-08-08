/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_MESH_OPBASE_PUB_H
#define ALL_REDUCE_MESH_OPBASE_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class  AllReduceMeshDirect : public ExecutorBase {
public:
    explicit  AllReduceMeshDirect(const HcclDispatcher dispatcher, const u64 reduceAttrBitMap,
        std::vector<Stream> &meshStreams, const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 interRank, u32 interRankSize, u32 userRank,
        const HcomCollOpInfo *opInfo);
    ~ AllReduceMeshDirect() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
    HcclResult PrepareSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice);
    HcclResult PrepareAllreduceSliceData();
    HcclResult RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK> &links);
    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }
    const u64 reduceAttr_;
    u32 localRank_;
    u32 localRankSize_;
    u32 userRank_;
    std::vector<Stream> meshStreams_;               /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignal_;    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux_; /* 从stream wait，主steam record */
    const HcomCollOpInfo *opInfo_;
};
} // namespace hccl
#endif /* ALL_REDUCE_MESH_OPBASE_H */