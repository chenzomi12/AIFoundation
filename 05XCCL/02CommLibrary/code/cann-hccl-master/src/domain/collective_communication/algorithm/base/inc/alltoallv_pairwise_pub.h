/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_PAIRWISE_PUB_H
#define ALLTOALL_V_PAIRWISE_PUB_H

#include "executor_base_pub.h"

namespace hccl {
struct AlltoAllVBufferInfo {
    DeviceMem mem;
    u64* counts = nullptr;
    u64* displs = nullptr;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;

    AlltoAllVBufferInfo& operator=(const AlltoAllVBufferInfo& that) noexcept
    {
        if (&that != this) {
            mem = that.mem;
            counts = that.counts;
            displs = that.displs;
            dataType = that.dataType;
        }
        return *this;
    }

    AlltoAllVBufferInfo& operator=(const AlltoAllVBufferInfo&& that) noexcept
    {
        if (&that != this) {
            mem = that.mem;
            counts = that.counts;
            displs = that.displs;
            dataType = that.dataType;
        }
        return *this;
    }
};

class AlltoAllVPairWise {
public:
    explicit AlltoAllVPairWise(const HcclDispatcher dispatcher,
        const std::map<u32, std::vector<u64>> &rankSendDisplsMap = {},
        const std::map<u32, std::vector<u64>> &rankRecvDisplsMap = {},
        HcclWorkflowMode workMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    virtual ~AlltoAllVPairWise();

    /* 图模式使用该prepare */
    HcclResult Prepare(AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer,
                       bool isAlltoAllZCopyMode, const Stream &stream);

    /* 单算子使用该prepare */
    HcclResult Prepare(AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer,
                       DeviceMem& scratchInputMem, DeviceMem& scratchOutputMem,
                       bool isAlltoAllZCopyMode, const Stream &stream);
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);

protected:
private:
    HcclResult LocalCopy(const u32 rank);
    // 单算子模式使用该函数
    HcclResult RunBCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    // 图模式使用该函数
    HcclResult RunZCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    HcclResult CalcSendRecvCounts(u32 times, u32 curTime, u64 totalBytes, u64 &curBytes) const;

    // 单算子模式使用该SendRecv
    HcclResult SendRecv(u64 curSendBytes, u64 curRecvBytes, u8* sendAddr, u8* recvAddr,
        std::shared_ptr<Transport> prevTransport, std::shared_ptr<Transport> nextTransport);
    // 图模式使用该SendRecv
    HcclResult SendRecv(TxMemoryInfo txMemoryInfo, RxMemoryInfo rxMemoryInfo,
        std::shared_ptr<Transport> prevTransport, std::shared_ptr<Transport> nextTransport);

    const HcclDispatcher dispatcher_;
    AlltoAllVBufferInfo sendBuffer_;
    AlltoAllVBufferInfo recvBuffer_;
    // 约束： scratchInputMem scratchOutputMem 用于transport中转，
    // 这两块内存需要在transport注册，两块内存的大小需一致
    DeviceMem scratchInputMem_;
    DeviceMem scratchOutputMem_;
    Stream stream_;
    u64 scratchMemSize_;
    u32 sendDataUnitBytes_;
    u32 recvDataUnitBytes_;
    const std::map<u32, std::vector<u64>> &rankSendDisplsMap_;
    const std::map<u32, std::vector<u64>> &rankRecvDisplsMap_;
    HcclWorkflowMode workMode_;
    bool isAlltoAllZCopyMode_;
};
}  // namespace hccl

#endif /* ALLTOALL_V_PAIRWISE_PUB_H */