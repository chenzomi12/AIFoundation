/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_STAGED_PAIRWISE_PUB_H
#define ALLTOALL_V_STAGED_PAIRWISE_PUB_H

#include "alltoallv_staged_base_pub.h"

namespace hccl {
class AlltoAllVStagedPairwise : public AlltoAllVStagedBase {
public:
    explicit AlltoAllVStagedPairwise(const HcclDispatcher dispatcher, Stream &stream);

    ~AlltoAllVStagedPairwise() override;
    HcclResult Prepare(DeviceMem &sendMem, DeviceMem &recvMem, StageAlltoAllVAddrInfo &sendAddrInfo,
        StageAlltoAllVAddrInfo &recvAddrInfo, bool isAlltoAllZCopyMode,
        const std::vector<Stream> &subStreams = std::vector<Stream>()) override;
    HcclResult Prepare(DeviceMem &sendMem, DeviceMem &recvMem, DeviceMem &scratchInputMem,
        DeviceMem &scratchOutputMem, StageAlltoAllVAddrInfo &sendAddrInfo, StageAlltoAllVAddrInfo &recvAddrInfo,
        bool isAlltoAllZCopyMode = false, const std::vector<Stream> &subStreams = std::vector<Stream>()) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult RunZCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunBCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);

    // 单算子执行相关接口 start
    void CalcSendRecvTimes(u64 &sendTimes, u64 &recvTimes, const u32 prevRank, const u32 nextRank);
    void LoadPolicies(const u32 rank, StageAlltoAllVAddrInfo &addrInfos,
        std::vector<std::list<OneSendRecvAddrInfo>> &policies);
    HcclResult CheckPolicies(const u64 times, const std::vector<std::list<OneSendRecvAddrInfo>> &policies) const;
    void SplitSendRecvAddrInfo(OneSendRecvAddrInfo &curLastInfo, OneSendRecvAddrInfo &addrInfo,
        const u64 &curCCLBufSize) const;
    HcclResult SendRecv(const u64 curSendTime, const std::vector<std::list<OneSendRecvAddrInfo>> &sendPolicies,
        const u64 curRecvTime, const std::vector<std::list<OneSendRecvAddrInfo>> &recvPolicies,
        std::shared_ptr<Transport> prevTransport, std::shared_ptr<Transport> nextTransport);
    // 单算子执行相关接口 end

    HcclResult ExecuteBarrier(std::shared_ptr<Transport> preLink, std::shared_ptr<Transport> aftLink);
    HcclResult ExecuteBarrier(bool hasSend, bool hasRecv, std::shared_ptr<Transport> preLink,
        std::shared_ptr<Transport> aftLink);

    // 单算子CCLbuf
    DeviceMem scratchInputMem_;
    DeviceMem scratchOutputMem_;
    u64 scratchMemSize_; // 单算子CCLbuf大小
};
} // namespace hccl
#endif /* ALLTOALL_V_STAGED_PAIRWISE_PUB_H */