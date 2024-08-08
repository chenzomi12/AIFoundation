/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALG_H
#define HCCL_ALG_H

#include "hccl_common.h"
#include "mem_device_pub.h"
#include "dispatcher.h"
#include "parallel_task_loader.h"
#include "comm_factory_pub.h"
#include "ccl_buffer_manager.h"
#include "workspace_resource.h"
#include "hccl_impl_pub.h"
#include "hccl_opbase_atrace_info_pub.h"
#include "resource_manager/queue_notify_manager.h"
#include "topo_matcher.h"
#include "coll_alg_operator.h"

namespace hccl {
class hcclImpl;
class HcclAlg {
public:
    explicit HcclAlg();
    virtual ~HcclAlg();
    HcclResult Init(const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
        std::unique_ptr<WorkspaceResource> &workSpaceRes, CCLBufferManager &cclBufferManager,
        const HcclDispatcher dispatcher, const HcclDispatcher vDispatcher,
        const std::unique_ptr<NotifyPool> &notifyPool,
        std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
        const std::unique_ptr<QueueNotifyManager> &queueNotifyManager,
        HcclAlgoAttr &algoAttr, HcclTopoAttr &topoAttr, bool isHeterogComm = false);
    HcclResult ReleaseCommInfos();
    HcclResult AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, Stream stream, HcomCollOpInfo *opInfo = nullptr);
    HcclResult AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);

    HcclResult Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        Stream stream);
    HcclResult BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
    HcclResult Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, Stream stream);
    HcclResult ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, Stream stream,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
    HcclResult Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream);
    HcclResult ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
    HcclResult Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, Stream stream);
    HcclResult SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, Stream stream);
    HcclResult Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, Stream stream);
    HcclResult ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, Stream stream);
    HcclResult Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
        HcclDataType dataType, Stream stream);
    HcclResult GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
        u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize);
    HcclResult GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        u64 &memSize);
    HcclResult GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize);
    HcclResult ClearOpResource(const std::string &tag);
    bool IsExistCommRes(const std::string &tag);
    HcclResult CreateMutiStreamRes(const std::string &tag, Stream &stream, innerStreamInfo_t &streamInfo,
        AlgType algType, bool isAicpuModeEn = false);
    HcclResult CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
        std::unique_ptr<CommInfo> &commInfo, u32 root = INVALID_VALUE_RANKID, bool isP2p = false,
        bool isAicpuModeEn = false);
    HcclResult CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
        u32 root = INVALID_VALUE_RANKID, bool isP2p = false);
    HcclResult CreateP2PCommQuerry(const std::string &tag, u32& status);
    HcclResult CreateP2PCommAsync(const std::string &tag, DeviceMem &mem, u32 peerRank, u32& status);
    void CancelCommRes(const std::string &tag);
    void Break();
    HcclResult SetAlgType(AlgType algType, HcclCMDType opType);
    HcclResult GetAlgType(AlgType &algType, HcclCMDType opType);
    static std::string AlgTypeToStr(const AlgType algType);
    HcclResult SupportDeterministicOptim(bool &isDeterministicOptim);
    HcclResult SetHDCModeInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
        std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort);

    u8 GetDeterministicConfig() const;  // 获取确定性计算配置
    HcclResult SetDeterministicConfig(const u8 deterministic);  // 设置确定性计算配置

    std::unique_ptr<CollAlgOperator> GetAlgOperator(const HcclCMDType &opType);

    HcclResult GetAlltoAllStatus(DeviceMem &tinySendRecvMem, bool &isAlltoAllZCopyMode);
private:
    std::unique_ptr<hcclImpl> pimpl_;
    HcclResult InitExternalEnable();
    HcclResult InitTopoInfoPartOne(HcclTopoAttr &topoAttr);
    HcclResult InitTopoInfoPartTwo();
    HcclResult InitAlgoInfo(HcclAlgoAttr &algoAttr);
    HcclTopoInfo topoInfo_;
    HcclAlgoInfo algoInfo_;
    HcclExternalEnable externalEnable_;
    std::unique_ptr<TopoMatcher> topoMatcher_;
};
}  // namespace hccl

#endif  // HCCL_ALG_H
