/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_STAGED_MESH_PUB_H
#define ALLTOALL_V_STAGED_MESH_PUB_H

#include "alltoallv_staged_base_pub.h"

namespace hccl {
class AlltoAllVStagedMesh : public AlltoAllVStagedBase {
public:
    explicit AlltoAllVStagedMesh(const HcclDispatcher dispatcher, Stream &stream,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain,
        u32 userRank, std::vector<Stream> &subStreams);
    ~AlltoAllVStagedMesh() override;
    HcclResult Prepare(DeviceMem &sendMem, DeviceMem &recvMem, StageAlltoAllVAddrInfo &sendAddrInfo,
        StageAlltoAllVAddrInfo &recvAddrInfo, bool isAlltoAllZCopyMode,
        const std::vector<Stream> &subStreams = std::vector<Stream>()) override;
    HcclResult Prepare(DeviceMem &sendMem, DeviceMem &recvMem, DeviceMem &scratchInputMem,
        DeviceMem &scratchOutputMem, StageAlltoAllVAddrInfo &sendAddrInfo, StageAlltoAllVAddrInfo &recvAddrInfo,
        bool isAlltoAllZCopyMode, const std::vector<Stream> &subStreams = std::vector<Stream>()) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult RunZCopyMode(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult LoadTask(std::shared_ptr<Transport> destTransport, Stream &currentStream,
        std::vector<TxMemoryInfo> &txMems, std::vector<RxMemoryInfo> &rxMems) const;
    void BuildSendRecvMemoryInfo(std::vector<TxMemoryInfo>& txMems, std::vector<RxMemoryInfo>& rxMems, u32 destRank);
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub_;
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain_;
    u32 userRank_;
    std::vector<Stream> &subStreams_;
};
} // namespace hccl
#endif /* ALLTOALL_V_STAGED_MESH_PUB_H */