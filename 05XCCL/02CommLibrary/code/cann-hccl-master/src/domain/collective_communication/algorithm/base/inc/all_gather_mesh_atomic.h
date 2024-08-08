/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MESH_ATOMIC_H
#define ALL_GATHER_MESH_ATOMIC_H

#include "all_gather_mesh_pub.h"

namespace hccl {
class AllGatherMeshAtomic : public AllGatherMesh {
public:
    explicit AllGatherMeshAtomic(const HcclDispatcher dispatcher,
        std::vector<Stream> &meshStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux,
        u32 interRank,       // 所有环里面的id
        u32 interRankSize,
        u32 userRank); // 所有大环的rank个数，commcombine提供接口

    ~AllGatherMeshAtomic() override;

protected:
    HcclResult RunAllGather(const std::vector<LINK> &links,
                                const std::vector<Slice> &outputSlices,
                                const std::vector<Slice> &inputSlices) override;
};
}  // namespace hccl

#endif /* ALL_GATHER_MESH_ATOMIC_H */
