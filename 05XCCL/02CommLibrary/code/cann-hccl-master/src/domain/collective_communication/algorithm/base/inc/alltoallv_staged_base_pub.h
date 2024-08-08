/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_STAGED_BASE_PUB_H
#define ALLTOALL_V_STAGED_BASE_PUB_H

#include "alltoallv_staged_calculator_pub.h"

namespace hccl {
class AlltoAllVStagedBase {
public:
    explicit AlltoAllVStagedBase(const HcclDispatcher dispatcher, Stream &stream);
    virtual ~AlltoAllVStagedBase() = default;

    virtual HcclResult Prepare(DeviceMem &sendMem, DeviceMem &recvMem, StageAlltoAllVAddrInfo& sendAddrInfo,
        StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode,
        const std::vector<Stream> &subStreams = std::vector<Stream>()) = 0;
    virtual HcclResult Prepare(DeviceMem &sendMem, DeviceMem &recvMem, DeviceMem &scratchInputMem,
        DeviceMem &scratchOutputMem, StageAlltoAllVAddrInfo &sendAddrInfo, StageAlltoAllVAddrInfo &recvAddrInfo,
        bool isAlltoAllZCopyMode, const std::vector<Stream> &subStreams = std::vector<Stream>()) = 0;
    virtual HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) = 0;

protected:
    HcclResult LocalCopy(u32 rank);

    const HcclDispatcher dispatcher_;

    DeviceMem sendMem_;
    DeviceMem recvMem_;

    StageAlltoAllVAddrInfo sendAddrInfo_;
    StageAlltoAllVAddrInfo recvAddrInfo_;
    Stream &mainStream_;
    bool isAlltoAllZCopyMode_;
};
} // namespace hccl
#endif /* ALLTOALL_V_STAGED_BASE_PUB_H */