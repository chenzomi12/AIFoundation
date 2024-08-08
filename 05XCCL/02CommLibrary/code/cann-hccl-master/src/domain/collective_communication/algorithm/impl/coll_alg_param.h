/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_COMM_H
#define COLL_ALG_COMM_H

#include <string>
#include <set>
#include <unordered_set>

#include "hccl_common.h"
#include "hccl_types.h"
#include "transport_pub.h"
#include "stream_pub.h"
#include "local_notify.h"
#include "hccl_opbase_atrace_info_pub.h"
#include "common.h"

namespace hccl {
using RankId = u32;

enum TransportMemType {
    CCL_INPUT = 0,
    CCL_OUTPUT,
    SCRATCH,
    PARAM_INPUT,
    PARAM_OUTPUT,
    AIV_INPUT,
    AIV_OUTPUT,
    RESERVED
};

enum OpMode {
    OPBASE = 0,
    OFFLOAD = 1
};

enum DeviceMode {
    HOST = 0,
    AICPU = 1
};

struct TransportRequest {
    bool isValid = false;
    RankId localUserRank = 0;
    RankId remoteUserRank = 0;
    TransportMemType inputMemType = TransportMemType::RESERVED;
    TransportMemType outputMemType = TransportMemType::RESERVED;
};

struct SingleSubCommTransport {
    std::vector<TransportRequest> transportRequests;
    std::vector<LINK> links;
    bool isUsedRdma = false;
    u64 taskNum = 0;
    std::map<u32, u32> userRank2subCommRank;
    std::map<u32, u32> subCommRank2UserRank;
    bool supportDataReceivedAck = false;
    LinkMode linkMode = LinkMode::LINK_DUPLEX_MODE;
    bool enableUseOneDoorbell = false;
    bool needVirtualLink =false; // for alltoall 多线程性能提升使用
    std::vector<LINK> virtualLinks; // for alltoall 多线程性能提升使用
};

using LevelNSubCommTransport = std::vector<SingleSubCommTransport>;
using OpCommTransport = std::vector<LevelNSubCommTransport>;

struct AlgResourceRequest {
    u64 scratchMemSize = 0;
    u32 streamNum;
    u32 notifyNum;
    bool needAivBuffer;
    DeviceMode mode = DeviceMode::HOST;     // 用于区分是host模式，还是aicpu模式
    OpCommTransport opTransport;
    void Describe()
    {
        HCCL_DEBUG("[AlgResourceRequest], scratchMemSize[%u], streamNum[%u], notifyNum[%u], needAivBuffer[%u], "
            "DeviceMode[%d].", scratchMemSize, streamNum, notifyNum, needAivBuffer, mode);
    };
};

struct AlgResourceResponse {
    DeviceMem cclInputMem;
    DeviceMem cclOutputMem;
    DeviceMem paramInputMem;
    DeviceMem paramOutputMem;
    DeviceMem scratchMem;
    DeviceMem aivInputMem;
    DeviceMem aivOutputMem;
    std::vector<Stream> streams;
    std::vector<std::shared_ptr<LocalNotify> > notifies;  // 大小为streams的两倍
    OpCommTransport opTransportResponse;
};

struct OpParam {
    std::string tag;
    Stream stream;
    void* inputPtr;
    u64 inputSize;
    void* outputPtr;
    u64 outputSize;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    RankId root = INVALID_VALUE_RANKID;
    RankId dstRank;
    RankId srcRank;
    HcclOpBaseAtraceInfo* opBaseAtraceInfo = nullptr;
    union {
        struct {
            u64 count;
            HcclDataType dataType;
        } DataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            void* sendCounts;
            void* recvCounts;
            void* sdispls;
            void* rdispls;
            void* sendCountMatrix;
        } All2AllDataDes;
        struct {
            HcclSendRecvItem** orderedList;
            u32 itemNum;
        } BatchSendRecvDataDes;
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
};

struct SubCommInfo {
    u32 localRank;
    u32 localRankSize;
    std::vector<LINK> links;
    std::vector<LINK> virtualLinks; // for alltoall 多线程性能提升使用
};

}   // namespace hccl
#endif