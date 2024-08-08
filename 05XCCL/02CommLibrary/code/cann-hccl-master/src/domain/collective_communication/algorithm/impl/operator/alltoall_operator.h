/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_OPERATOR_H
#define ALLTOALL_OPERATOR_H

#include "coll_alg_operator.h"
namespace hccl {
class AlltoAllOperator : public CollAlgOperator {
public:
    AlltoAllOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~AlltoAllOperator();

    HcclResult GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
        u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize);
    HcclResult GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        u64 &memSize);

    HcclResult CheckSendRecvParams(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    HcclResult GetAlltoAllvSendRecvInfo(const OpParam& param, const HostMem &alltoallAddrInfoGathered);
    HcclResult GetAlltoAllvcSendRecvInfo(const void *sendCountMatrix, HcclDataType sendType, HcclDataType recvType);
    void UpdateAlltoAllCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, std::string& copyMode);
    HcclResult SelectAlgforAlltoAll(const OpParam& param, std::string& algName, std::string& copyMode);
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag);

    HcclResult GetAlltoAllvAllAddrInfo(u64 *sendLength, u64 *sendOffset, u64 *recvLength, u64 *recvOffset,
        Stream &stream, std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);
    HcclResult PrepareAlltoAllAddrInfo(const void *sendCounts, const void *sdispls, HcclDataType sendType,
        const void *recvCounts, const void *rdispls, HcclDataType recvType, Stream &stream,
        std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);
    HcclResult PreparePreOpParam(OpParam& preProcessOpParam, const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
        Stream &preProcessStream);
    bool JudgeIfNeedPreProcessAndGetParam(const OpParam& param, std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);
    void SetPreProcessResult(HostMem hostCollectBuffer);
    HcclResult SetExcutorExtraInfo(const std::string& algName);

private:
    std::vector<u64> GenerateSendCountMatrix(u64 count, u32 rankSize);

    bool IsSatisfyAlltoallPipelineCondition();
    u64 GetAlltoall2LevelPipelineScratchSize910B(
        u32 rank,
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    u64 GetAlltoall2LevelPipelineMaxScratchSize910B(
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    HcclResult RunAlltoAllVTwoLevelPipeline(DeviceMem &sendBuf, DeviceMem &recvBuf,
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, Stream &stream, const std::string &tag);
    HcclResult RunAlltoAllVFullMesh(DeviceMem &sendBuf, HcclDataType sendType, DeviceMem &recvBuf,
        HcclDataType recvType, std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        Stream &stream, const std::string &tag);
    HcclResult RunAlltoAllVStaged(DeviceMem &sendBuf, HcclDataType sendType, DeviceMem &recvBuf, HcclDataType recvType,
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, Stream &stream, const std::string &tag);
    HcclResult PrepareAlltoAllVStaged1(DeviceMem &sendBuf, DeviceMem &recvBuf, DeviceMem &scratchMem,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosIntra,
        Stream &stream, const std::string &tag, std::unique_ptr<AlltoAllVStagedBase> &alltoallOuter);

    HcclResult PrepareAlltoAllVStaged2(DeviceMem &recvBuf, DeviceMem &scratchMem,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosInter,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosInter,
        Stream &stream, const std::string &tag, std::unique_ptr<AlltoAllVStagedBase> &alltoallInner);

    HcclResult GetAlltoAllvAllSendRecvInfo(u64 *sendLength, u64 *sendOffset,
        u64 *recvLength, u64 *recvOffset, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo);
    // 重载方法适配单算子alltoallv
    HcclResult GetAlltoAllvAllSendRecvInfo(u64 *sendLength, u64 *sendOffset,
        u64 *recvLength, u64 *recvOffset, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, Stream &stream);
    HcclResult FormatAllMeshAggregationSendRecvInfo(HostMem &alltoallAddrInfoGathered,
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);
    // 更新AlltoAll BCopy/ZCopy模式
    void UpdateAlltoAllZCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, const std::string &tag);
    // 获取参与AlltoAll的所有rank的收发参数
    HcclResult GetAllMeshAggregationSendRecvInfo(const void *sendCounts, const void *sdispls, HcclDataType sendType,
        const void *recvCounts, const void *rdispls, HcclDataType recvType,
        std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, Stream &stream);
    // 解析获取alltoallvc的所有rank收发参数
    HcclResult GetAlltoAllvcAllSendRecvInfo(const void *sendCountMatrix, HcclDataType sendType, HcclDataType recvType,
        std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo);
    HcclResult ExchangeSendRecvInfoFromAllGather(const std::string &tag, void *inputPtr, void *outputPtr,
        u64 inputCount, HcclDataType dataType, Stream stream);
    bool HasMassTasks(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo);

    bool isAlltoAllZCopyMode_ = false;
    std::map<std::string, bool> isAlltoAllZCopyModeMap_;
    DeviceMem tinySendRecvMem_; // 在sendCount/recvCount全0时, 使用tinySendRecvMem_, 避免使用空deviceMem
    HostMem hostCollectBuffer_;
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo_;
};
}

#endif /** __ALLTOALL_OPERATOR_H_ */