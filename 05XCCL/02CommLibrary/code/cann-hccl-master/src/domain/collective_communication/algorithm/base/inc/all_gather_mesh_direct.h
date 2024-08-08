/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MESH_DIRECT_H
#define ALL_GATHER_MESH_DIRECT_H

#include "executor_base_pub.h"

namespace hccl {
class AllgatherMeshDirect : public ExecutorBase {
public:
    explicit AllgatherMeshDirect(const HcclDispatcher dispatcher,
        std::vector<Stream> &meshStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux,
        u32 interRank,       // 所有环里面的id
        u32 interRankSize,
        u32 userRank,
        HcomCollOpInfo *opInfo = nullptr); // 所有大环的rank个数，commcombine提供接口

    ~AllgatherMeshDirect() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
    // 获取向该rank往前的第i个rank
    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }

    inline u32 ForwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + step) % rankSize;
    }
    std::vector<Stream> meshStreams_; /** 多steam**/

    const std::vector<std::shared_ptr<LocalNotify>> &meshSignal_;  /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux_; /* 从stream wait，主steam record */
    u32 interRank_;       // 在所有rank环上的rankid
    u32 interRankSize_;
    u32 userRank_;
    HcomCollOpInfo *opInfo_;

private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
};
}  // namespace hccl

#endif /* ALL_GATHER_MESH_PUB_H */
