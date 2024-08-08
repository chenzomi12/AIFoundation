/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLTOALL_COMM_EXECUTOR_H
#define COLL_ALLTOALL_COMM_EXECUTOR_H
#include "coll_comm_executor.h"
namespace hccl {
constexpr u64 MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH = 16;
constexpr u64 TINY_MEM_SIZE = 2 * 1024 * 1024; // tinyMem size
constexpr u32 MINORS_NUM_TWO = 2;

class CollAlltoAllExecutor : public CollNativeExecutorBase {
public:
    CollAlltoAllExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAlltoAllExecutor() = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceResponse& algRes) override;
    HcclResult SetExcutorExtraInfo(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo) override;
    HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest &resourceRequest) override;
    virtual HcclResult CheckNeedCreateVirtualLinks(AlgResourceRequest &resourceRequest);
protected:
    /* *************** 算法编排 *************** */
    // 公共接口
    HcclOpMetaInfo GetOpMeta(HcclCMDType opType, const u64 size);
    void UpdateAlltoAllZCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    void CalcIntraMeshAggregationAlltoAllMemInfo(const AlltoAllUserRankInfo &userRankInfo,
        const std::vector<SendRecvInfo> &allSendRecvInfo,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosIntra, u32 meshAggregationRankSize,
        const bool &isSingleMesh);
    void CalcIntraMeshAggregationSendInfo(const AlltoAllUserRankInfo &userRankInfo,
        const SendRecvInfo &mySendRecvInfo, const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo,
        u32 rankInMeshAggregation, u32 infoIndex, OneSendRecvAddrInfo &curSendInfo, u32 meshAggregationRankSize,
        const bool &isSingleMesh);
    void CalcIntraMeshAggregationRecvInfo(const AlltoAllUserRankInfo &userRankInfo,
        const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo, u32 infoIndex, OneSendRecvAddrInfo &curRecvInfo,
        u32 meshAggregationRankSize, const bool &isSingleMesh);
    void CalcIntraMeshAggregationRecvInfoInMeshAggregation(u32 rankIndex, u32 infoIndex,
        const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo, u64 &localOffset, u32 &offsetCounter,
        u64 &localLength, u64 &remoteOffset, u32 meshAggregationRankSize);
    u64 CalAlltoAllVScratchMemSize(u64 &workSpaceMemSize);
    bool NAFullmeshSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize);
    bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize);
    bool HasMassTasks(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);

    OpParam AlltoAllVParam_;
    AlgResourceResponse algRes_;
    bool DMAReduceFlag_{false}; // 是否DMA消减
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo_;
    bool isAlltoAllZCopyMode_ = false;
};

} // namespace hccl

#endif