/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_factory.h"
#include <sstream>
#include <algorithm>
#include "p2p_mgmt_pub.h"
#include "adapter_pub.h"
#include "device_capacity.h"
#include "nonuniform_hierarchical_ring_v1_base_pub.h"
#include "search_path.h"
#include "calc_p2p_transport_req.h"
namespace hccl {
// 模组设备数量
constexpr u32 SERVER_RANK_SIZE = 8;
bool Ascending(const RankInfo &first, const RankInfo &second)
{
    if (first.serverIdx != second.serverIdx) {
        return first.serverIdx < second.serverIdx;
    } else {
        return first.devicePhyId < second.devicePhyId;
    }
}
bool CompareWithUserRankAscend(const RankInfo &left, const RankInfo &right)
{
    return left.userRank < right.userRank;
}

CommFactory::CommFactory(const std::string &identifier, const u32 userRank, const u32 userRankSize,
    const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool,
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
    const bool isUsedRdmaOuter, const TopoType topoFlag, const DevType deviceType,
    const std::vector<RankInfo> rankVector, const NICDeployment nicDeploymentInner, bool isHeterogComm,
    bool isDiffAggregation, const void *transportResourceInfoAddr, size_t transportResourceInfoSize,
    u32 meshAggregationRankSize, bool isHaveCpuRank, bool isUsedInterHccsMode, bool useSdidForDeviceId)
    : identifier_(identifier),
      userRank_(userRank),
      userRankSize_(userRankSize),
      topoFlag_(topoFlag),
      deviceType_(deviceType),
      dispatcher_(dispatcher),
      notifyPool_(notifyPool),
      netDevCtxMap_(netDevCtxMap),
      isUsedRdmaOuter_(isUsedRdmaOuter),
      CommPlaneVector_(COMM_LEVEL_RESERVED),
      isBridgeVector_(),
      serverToRank_(),
      superPodToRank_(),
      multiOuterOrder_(),
      rankVector_(rankVector),
      nicDeployInner_(nicDeploymentInner),
      isHeterogComm_(isHeterogComm),
      isDiffAggregation_(isDiffAggregation),
      transportResourceInfoAddr_(transportResourceInfoAddr),
      transportResourceInfoSize_(transportResourceInfoSize),
      meshAggregationRankSize_(meshAggregationRankSize),
      serverNum_(0),
      isHaveCpuRank_(isHaveCpuRank),
      reusedSocketManager_(),
      deviceLogicId_(0),
      isUsedInterHccsMode_(isUsedInterHccsMode),
      useSdidForDeviceId_(useSdidForDeviceId)
{
}

CommFactory::~CommFactory()
{
    // 销毁资源
    multiOuterOrder_.clear();
    CommPlaneVector_.clear();
    isBridgeVector_.clear();
    superPodToRank_.clear();
    serverToRank_.clear();
    deviceLinkTypeMap_.clear();
    rankVector_.clear();
}

HcclResult CommFactory::Init()
{
    HCCL_INFO(
        "factory init:collective id[%s], user rank[%u], user rank size[%u], topo type[%d], device Type[%d], "\
        "meshAggregationRankSize[%u]",
        identifier_.c_str(), userRank_, userRankSize_, topoFlag_, deviceType_, meshAggregationRankSize_);

    // 参数有效性校验
    CHK_RET(CheckInitInfo());
    s32 deviceLogicID = 0;
    if (!isHeterogComm_ && rankVector_[userRank_].devicePhyId != HOST_DEVICE_ID) {
        CHK_RET(hrtGetDevice(&deviceLogicID));
        deviceLogicId_ = deviceLogicID;
    }

    reusedSocketManager_.reset(new (std::nothrow) HcclSocketManager(nicDeployInner_, deviceLogicId_,
        rankVector_[userRank_].devicePhyId, userRank_));
    CHK_PTR_NULL(reusedSocketManager_);
    CHK_RET(reusedSocketManager_->Init());

    // 填充必要数据结构
    CHK_RET(SetRankInfo());

    if (IsGeneralServer() && GetRemoteIsHdc()) {
        HCCL_INFO("heterog ES ps factory init no need set topoInfo");
    } else {
        // 设置拓扑信息
        CHK_RET(SetTopologyInfo());
        // 根据拓扑类型以及芯片类型，校验两层拓扑(外层/内层)、单层拓扑的平面个数合法性
        CHK_RET(CheckPlaneInfo());
    }

    CHK_RET(SetRankMap());
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CheckCommPara(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo)
{
    CHK_PRT_RET(commParaInfo.commPlane >= COMM_LEVEL_RESERVED,
        HCCL_ERROR("[Check][CommPara]tag[%s], commPlane[%d] is invalid, is out of range [0, %d]",
            tag.c_str(), commParaInfo.commPlane, COMM_LEVEL_RESERVED - 1), HCCL_E_PARA);

    // 判断commPlane和commType的组合是否支持
    bool isSupport = true;
    switch (commParaInfo.commType) {
        case CommType::COMM_TAG_RING_INNER:
        case CommType::COMM_TAG_HALVING_DOUBLING: {
            isSupport = (commParaInfo.commPlane == COMM_LEVEL0) ||
                        (commParaInfo.commPlane == COMM_LEVEL1) ||
                        (commParaInfo.commPlane == COMM_LEVEL2);
            break;
        }
        case CommType::COMM_TAG_MESH: {
            isSupport = (commParaInfo.commPlane == COMM_COMBINE) ||
                        (commParaInfo.commPlane == COMM_LEVEL0) ||
                        (commParaInfo.commPlane == COMM_MESH_L0) ||
                        (commParaInfo.commPlane == COMM_MESH_L1) ||
                        (commParaInfo.commPlane == COMM_LEVEL2) ||
                        (commParaInfo.commPlane == COMM_COMBINE_ORDER);
            break;
        }
        case CommType::COMM_TAG_RING_COMBINED:
        case CommType::COMM_TAG_MESH_COMBINED:
        case CommType::COMM_TAG_P2P: {
            isSupport = commParaInfo.commPlane == COMM_COMBINE;
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING:
        case CommType::COMM_TAG_NONUNIFORM_BRUCK: {
            isSupport = (commParaInfo.commPlane == COMM_LEVEL1);
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1: {
            isSupport = (commParaInfo.commPlane == COMM_LEVEL1 && deviceType_ != DevType::DEV_TYPE_910_73);
            break;
        }
        case CommType::COMM_TAG_STAR:
        case CommType::COMM_TAG_WHOLE_NHR:
        case CommType::COMM_TAG_WHOLE_NHR_V1:
        case CommType::COMM_TAG_WHOLE_NB: {
            isSupport = (deviceType_ != DevType::DEV_TYPE_910_73);
            break;
        }
        default: {
            HCCL_ERROR("[Check][CommPara]commType[%d] is invalid", commParaInfo.commType);
            return HCCL_E_PARA;
        }
    }

    CHK_PRT_RET(isSupport == false,
        HCCL_ERROR("[Check][CommPara]tag[%s], deviceType[%d], commPlane[%d] and commType[%d] is not support",
            tag.c_str(), deviceType_, commParaInfo.commPlane, commParaInfo.commType), HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetIsUsedRdma(const CommParaInfo &commParaInfo, bool &isUsedRdma)
{
    std::vector<std::vector<RankInfo> > commP2PPlaneVec;
    if (commParaInfo.commType == CommType::COMM_TAG_P2P) {
        // P2P只需要判断两张卡之间的连接关系
        bool invalidcheck = (rankVector_.size() <= userRank_) || (rankVector_.size() <= commParaInfo.peerUserRank);
        CHK_PRT_RET(invalidcheck, HCCL_ERROR("[GetIsUsedRdma]dstUserRank[%u] or userRank[%u] is bigger than "\
            "rankVector size[%u]", commParaInfo.peerUserRank, userRank_, rankVector_.size()), HCCL_E_PARA);

        std::vector<RankInfo> commP2PRankVec;
        commP2PRankVec.push_back(rankVector_[userRank_]);
        commP2PRankVec.push_back(rankVector_[commParaInfo.peerUserRank]);
        commP2PPlaneVec.push_back(commP2PRankVec);
    }

    std::vector<std::vector<RankInfo> > &commPlaneVec = (commParaInfo.commType == CommType::COMM_TAG_P2P) ?
        commP2PPlaneVec : CommPlaneVector_[commParaInfo.commPlane];

    bool isInterSuperPod = false;
    bool isInterServer = false;
    bool isConnectedWithPcie = false;
    for (const std::vector<RankInfo> &commPlane : commPlaneVec) {
        for (const RankInfo &dstRank : commPlane) {
            if (rankData_.superPodId != dstRank.superPodId) { // 跨超节点场景
                isInterSuperPod = true;
            } else if (rankData_.serverIdx != dstRank.serverIdx) { // 不跨超节点, 跨server场景
                isInterServer = true;
            } else { // 同server, PCIE互连场景
                auto it = deviceLinkTypeMap_.find(dstRank.devicePhyId);
                CHK_PRT_RET(it == deviceLinkTypeMap_.end(),
                    HCCL_ERROR("can't find devicePhyId[%d] in deviceLinkTypeMap_", dstRank.devicePhyId),
                    HCCL_E_NOT_FOUND);
                isConnectedWithPcie |= (it->second == LinkTypeInServer::PXI_TYPE);
            }
        }
    }
    // 使能RDMA的场景: 1.跨超节点  2.跨server且不使能HCCS  3.PCIE连接且使能RDMA开关
    isUsedRdma = (isInterSuperPod) ||
                 (isInterServer && !isUsedInterHccsMode_) || (isConnectedWithPcie && isUsedRdmaOuter_);
    HCCL_INFO("[GetIsUsedRdma]isUsedRdma[%d], isInterSuperPod[%d], isInterServer[%d], isUsedInterHccsMode_[%d], "\
        "isConnectedWithPcie[%d], isUsedRdmaOuter_[%d]", isUsedRdma, isInterSuperPod, isInterServer,
        isUsedInterHccsMode_, isConnectedWithPcie, isUsedRdmaOuter_);
    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetIsUsedRdmaMap(std::unordered_map<u32, bool> &isUsedRdmaMap)
{
    for (const RankInfo &dstRank : rankVector_) {
        bool isInterSuperPod = false;
        bool isInterServer = false;
        bool isConnectedWithPcie = false;
        if (rankData_.superPodId != dstRank.superPodId) { // 跨超节点场景
            isInterSuperPod = true;
        } else if (rankData_.serverIdx != dstRank.serverIdx) { // 不跨超节点, 跨server场景
            isInterServer = true;
        } else { // 同server, PCIE互连场景
            auto it = deviceLinkTypeMap_.find(dstRank.devicePhyId);
            CHK_PRT_RET(it == deviceLinkTypeMap_.end(),
                HCCL_ERROR("can't find devicePhyId[%d] in deviceLinkTypeMap_", dstRank.devicePhyId),
                HCCL_E_NOT_FOUND);
            isConnectedWithPcie |= (it->second == LinkTypeInServer::PXI_TYPE);
        }
        // 使能RDMA的场景: 1.跨超节点  2.跨server且不使能HCCS  3.PCIE连接且使能RDMA开关
        bool isUsedRdma = (isInterSuperPod) ||
                (isInterServer && !isUsedInterHccsMode_) || (isConnectedWithPcie && isUsedRdmaOuter_);
        isUsedRdmaMap[dstRank.userRank] = isUsedRdma;
        HCCL_DEBUG("[GetIsUsedRdma]isUsedRdma[%d], isInterSuperPod[%d], isInterServer[%d], isUsedInterHccsMode_[%d], "\
            "isConnectedWithPcie[%d], isUsedRdmaOuter_[%d], dstRank[%d]", isUsedRdma, isInterSuperPod, isInterServer,
            isUsedInterHccsMode_, isConnectedWithPcie, isUsedRdmaOuter_, dstRank.userRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetRankVecInfo(std::vector<std::vector<std::vector<u32>>> &serverAndsuperPodToRank)
{
    std::vector<std::vector<u32>> serverToRank;
    std::vector<std::vector<u32>> superPodToRank;
    serverToRank.clear();
    superPodToRank.clear();
    u32 firstIdx = 0;

    serverToRank.resize(serverToRank_.size());
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        serverToRank[firstIdx].resize((iterMap->second).size());
        if (!(iterMap->second).empty()) {
            for (u32 i = 0; i < (iterMap->second).size(); i++) {
                serverToRank[firstIdx][i] = (iterMap->second)[i].userRank;
            }
        }
        firstIdx++;
    }

    u32 podFirstIdx = 0;
    superPodToRank.resize(superPodToRank_.size());
    for (auto iterMap = superPodToRank_.begin(); iterMap != superPodToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            superPodToRank[podFirstIdx].resize((iterMap->second).size());
            for (u32 i = 0; i < (iterMap->second).size(); i++) {
                superPodToRank[podFirstIdx][i] = (iterMap->second)[i].userRank;
            }
        }
        podFirstIdx++;
    }
    serverAndsuperPodToRank.push_back(serverToRank);
    serverAndsuperPodToRank.push_back(superPodToRank);
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommPlane(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, std::vector<std::unique_ptr<CommBase> > &commVec)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO("[Create][CommPlane]tag[%s], identifier[%s], commPlane[%d], commType[%d]",
        tag.c_str(), identifier_.c_str(), commParaInfo.commPlane, commParaInfo.commType);

    CHK_RET(CheckCommPara(tag, inputMem, outputMem, commParaInfo));
    bool isUsedRdma = false;
    CHK_RET(GetIsUsedRdma(commParaInfo, isUsedRdma));
    if (GetExternalInputEnableRdmaSdmaConcurrent() && deviceType_ == DevType::DEV_TYPE_910_73) {
        isUsedRdma = commParaInfo.forceRdma;
    }

    switch (commParaInfo.commType) {
        case CommType::COMM_TAG_RING_INNER:
        case CommType::COMM_TAG_RING_COMBINED: {
            ret = CreateCommRing(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
                isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_HALVING_DOUBLING: {
            ret = CreateCommHD(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
                isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_STAR: {
            std::vector<std::vector<RankInfo> > commPlaneVec;
            std::vector<RankInfo> linkParas;
            CreateStarLinkPara(linkParas);
            commPlaneVec.push_back(linkParas);
            ret = CreateCommStar(tag, inputMem, outputMem, commParaInfo, commPlaneVec, isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING: {
            ret = CreateCommNHR(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
                isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_WHOLE_NHR: {
            ret = CreateCommNHR(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
                isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1: {
            ret = CreateCommNHRV1(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
                isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_WHOLE_NHR_V1: {
            ret = CreateCommNHRV1(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
                isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_BRUCK: {
            ret = CreateCommNB(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
                isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_WHOLE_NB: {
            ret = CreateCommNB(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
                isUsedRdma, commVec);
            break;
        }
        case CommType::COMM_TAG_MESH: {
            if (commParaInfo.meshSinglePlane == true) {
                // 910B非确定性计算场景，server内MESH组网只需要创建一个commbase平面
                std::vector<std::vector<RankInfo> > commPlaneVec;
                commPlaneVec.push_back(CommPlaneVector_[commParaInfo.commPlane][0]);
                ret = CreateCommMesh(tag, inputMem, outputMem, commParaInfo, commPlaneVec, isUsedRdma, commVec);
            } else {
                ret = CreateCommMesh(tag, inputMem, outputMem, commParaInfo,  CommPlaneVector_[commParaInfo.commPlane],
                    isUsedRdma, commVec);
            }
            break;
        }
        case CommType::COMM_TAG_P2P: {
            ret = CreateCommP2P(tag, inputMem, outputMem, commParaInfo,
                CommPlaneVector_[commParaInfo.commPlane], isUsedRdma, commVec);
            break;
        }
        default: {
            HCCL_ERROR("[Create][CommPlane]commType[%d] is invalid", commParaInfo.commType);
            return HCCL_E_PARA;
        }
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Create][CommPlane]failed, tag[%s], commPlane[%d], commType[%d]",
        tag.c_str(), commParaInfo.commPlane, commParaInfo.commType), ret);

    HCCL_INFO("complete commPlane[%d] commType[%d] creation, Time:%lld us",
        commParaInfo.commPlane, commParaInfo.commType, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommRing(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
    bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        // 只有在当前环是bridge rank才需要创建comm实例
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork {};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO("[Create][CommRing]comm is used %s. userRank = %u, rank = %u",
            isUsedRdma ? "rdma" : "sdma", userRank_, rank);

        commVec[ringIndex].reset(new (std::nothrow) CommRing(identifier_, userRank_, userRankSize_,
            rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma,
            transportResourceInfoAddr_, transportResourceInfoSize_, tag, nicDeployInner_,
            false, false, isHaveCpuRank_, useSdidForDeviceId_));

        CHK_PRT_RET(!commVec[ringIndex], HCCL_ERROR("[Create][CommRing]comm array[%u] reset failed",
            ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommRing]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommHD(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
    bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    u32 subUserRankRoot = INVALID_VALUE_RANKID;
    if (commParaInfo.root != INVALID_VALUE_RANKID) {
        subUserRankRoot = GetSubRootUserRank(userRank_, commParaInfo.root);
        if (subUserRankRoot == INVALID_VALUE_RANKID) {
            HCCL_ERROR("[create][CommHD]get sub root userrank value[%u] invalid.", subUserRankRoot);
            return HCCL_E_PARA;
        }
    }

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        // 只有在当前环是bridge rank才需要创建comm实例
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork {};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO("[create][CommHD]comm is used %s. userRank = %u, rank = %u",
            isUsedRdma ? "rdma" : "sdma", userRank_, rank);

        commVec[ringIndex].reset(new (std::nothrow) CommHalvingDoubling(identifier_, userRank_, userRankSize_,
            rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma,
            transportResourceInfoAddr_, transportResourceInfoSize_,
            tag, nicDeployInner_, subUserRankRoot, HalvingDoublingType::RECURSIVE_HALVING_DOUBLING,
            isHaveCpuRank_, useSdidForDeviceId_));

        CHK_PRT_RET(!commVec[ringIndex], HCCL_ERROR("[create][CommHD]comm array[%u] reset failed",
            ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[create][CommHD]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

void CommFactory::CreateStarLinkPara(std::vector<RankInfo> &linkParas)
{
    linkParas = rankVector_;
}

HcclResult CommFactory::CreateCommStar(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
    bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec)
{
    HCCL_INFO("create comm star start");
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        IntraExchanger exchangerNetwork {};
        HCCL_INFO("[CreateCommStar] CommStar is used %s. userRank = %u", isUsedRdma ? "rdma" : "sdma", userRank_);

        commVec[ringIndex].reset(new (std::nothrow) CommStar(identifier_, userRank_, userRankSize_, userRank_,
            commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, transportResourceInfoAddr_,
            transportResourceInfoSize_, tag, nicDeployInner_, commParaInfo.root));

        CHK_PRT_RET(!commVec[ringIndex], HCCL_ERROR("[create][CommStar]comm array[%u] reset failed",
            ringIndex), HCCL_E_PARA);
        commVec[ringIndex]->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPort_, isSetHDCModeInfo_, isUseRankPort_);

        if (JudgmentSetHeterogP2p(userRank_)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[create][CommStar]comm array[%u] star rank[%u] init failed", ringIndex, userRank_);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommNHR(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
    bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        // 只有在当前环是bridge rank才需要创建comm实例
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork {};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO("[Create][CommNHR]comm is used %s. userRank = %u, rank = %u",
            isUsedRdma ? "rdma" : "sdma", userRank_, rank);

        commVec[ringIndex].reset(new (std::nothrow) CommNHR(identifier_, userRank_, userRankSize_,
            rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, transportResourceInfoAddr_,
            transportResourceInfoSize_, tag, nicDeployInner_));

        CHK_PRT_RET(!commVec[ringIndex], HCCL_ERROR("[Create][CommNHR]comm array[%u] reset failed",
            ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommNHR]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommNHRV1(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
    bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        // 只有在当前环是bridge rank才需要创建comm实例
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork {};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO("[Create][CommNHRV1]comm is used %s. userRank = %u, rank = %u",
            isUsedRdma ? "rdma" : "sdma", userRank_, rank);

        commVec[ringIndex].reset(new (std::nothrow) CommNHRV1(identifier_, userRank_, userRankSize_,
            rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, transportResourceInfoAddr_,
            transportResourceInfoSize_, tag, nicDeployInner_));

        CHK_PRT_RET(!commVec[ringIndex], HCCL_ERROR("[Create][CommNHRV1]comm array[%u] reset failed",
            ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommNHRV1]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommNB(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
    bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        // 只有在当前环是bridge rank才需要创建comm实例
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork {};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO("[Create][CommNB]comm is used %s. userRank = %u, rank = %u",
            isUsedRdma ? "rdma" : "sdma", userRank_, rank);

        commVec[ringIndex].reset(new (std::nothrow) CommNB(identifier_, userRank_, userRankSize_,
            rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, transportResourceInfoAddr_,
            transportResourceInfoSize_, tag, nicDeployInner_));

        CHK_PRT_RET(!commVec[ringIndex], HCCL_ERROR("[Create][CommNB]comm array[%u] reset failed",
            ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommNB]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommMesh(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec, bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase> > &commVec)
{
    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        CHK_PRT_RET(rank == INVALID_VALUE_RANKID, HCCL_ERROR("[Create][CommMesh] invalid rank info."), HCCL_E_PARA);

        IntraExchanger exchangerNetwork {};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO("[Create][CommMesh]comm is used %s. userRank = %u, rank = %u",
            isUsedRdma ? "rdma" : "sdma", userRank_, rank);

        commVec[ringIndex].reset(new (std::nothrow) CommMesh(identifier_, userRank_, userRankSize_,
            rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, transportResourceInfoAddr_,
            transportResourceInfoSize_, tag, false, nicDeployInner_, false, commParaInfo.isAicpuModeEn,
            isHaveCpuRank_, useSdidForDeviceId_));

        CHK_PRT_RET(!commVec[ringIndex], HCCL_ERROR("[Create][CommMesh]comm array[%u] reset failed",
            ringIndex), HCCL_E_PARA);

        if (JudgmentSetHeterogP2p(rank)) {
            commVec[ringIndex]->SetHeterogP2PType();
        }
        commVec[ringIndex]->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPort_, isSetHDCModeInfo_, isUseRankPort_);
        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommMesh]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommP2P(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec, bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase> > &commVec)
{
    bool invalidcheck = (rankVector_.size() <= userRank_) || (rankVector_.size() <= commParaInfo.peerUserRank);
    CHK_PRT_RET(invalidcheck,
        HCCL_ERROR("[Create][CommP2P]dstUserRank[%u] or userRank[%u] is bigger than rank vector size[%u].",
            commParaInfo.peerUserRank, userRank_, rankVector_.size()), HCCL_E_PARA);

    bool heterogP2P = ((rankVector_[userRank_].devicePhyId == HOST_DEVICE_ID) &&
                        (rankVector_[commParaInfo.peerUserRank].devicePhyId != HOST_DEVICE_ID)) ||
                        ((rankVector_[userRank_].devicePhyId != HOST_DEVICE_ID) &&
                        (rankVector_[commParaInfo.peerUserRank].devicePhyId == HOST_DEVICE_ID));

    if (heterogP2P) {
        return CreateCommP2PSync(tag, inputMem, outputMem, commParaInfo, CommPlaneVector_[commParaInfo.commPlane],
            isUsedRdma, commVec);
    }

    u32 ringSize = commPlaneVec.size();
    commVec.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ++ringIndex) {
        u32 rank = GetSubCollectiveRank(commPlaneVec[ringIndex]);
        CHK_PRT_RET(rank == INVALID_VALUE_RANKID, HCCL_ERROR("[Create][CommP2P] invalid rank info."), HCCL_E_PARA);

        IntraExchanger exchangerNetwork {};
        exchangerNetwork.socketManager = reusedSocketManager_;

        HCCL_INFO("[Create][CommP2P]comm is used %s. userRank = %u, rank = %u",
            isUsedRdma ? "rdma" : "sdma", userRank_, rank);

        commVec[ringIndex].reset(new (std::nothrow) CommP2P(identifier_, userRank_, userRankSize_,
            rank, commPlaneVec[ringIndex].size(), topoFlag_, dispatcher_, notifyPool_, netDevCtxMap_, exchangerNetwork,
            commPlaneVec[ringIndex], inputMem, outputMem, isUsedRdma, transportResourceInfoAddr_,
            transportResourceInfoSize_, tag, commParaInfo.peerUserRank, nicDeployInner_,
            isHaveCpuRank_, useSdidForDeviceId_));

        CHK_PRT_RET(!commVec[ringIndex], HCCL_ERROR("[Create][CommP2P]comm array[%u] reset failed",
            ringIndex), HCCL_E_PARA);

        if (commVec[ringIndex]->Init() != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommP2P]comm array[%u] init failed", ringIndex);
            commVec[ringIndex].reset(nullptr);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CreateCommP2PSync(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
    const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec, bool isUsedRdma,
    std::vector<std::unique_ptr<CommBase> > &commVec)
{
    u32 status;
    commVec = CreateCommP2PAsync(tag, inputMem, outputMem, commParaInfo.peerUserRank, status);
    for (u32 index = 0; index < commVec.size(); index++) {
        CHK_PRT_RET(!commVec[index],
            HCCL_ERROR("[Create][CommP2PSync]errNo[0x%016llx] tag[%s], created p2pComm[%u] is null.",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str(), index), HCCL_E_NOT_FOUND);
    }

    if (status == 0) {
        return HCCL_SUCCESS;
    }
    do {
        HcclResult ret = CreateCommP2PQuerry(commVec, status);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommP2P]comm p2p init failed");
            return ret;
        }
        SaluSleep(COMM_P2P_QUERRY_WAIT_TIME);
    } while (status == 1);
    return HCCL_SUCCESS;
}

std::vector<std::unique_ptr<CommBase> > CommFactory::CreateCommP2PAsync(const std::string &tag,
    const DeviceMem& inputMem, const DeviceMem& outputMem, const u32 dstUserRank, u32& status)
{
    u32 ringSize = CommPlaneVector_[COMM_COMBINE].size();
    std::vector<std::unique_ptr<CommBase> > commP2PArray(0); // 复用CommBase来实现P2P拓扑功能

    bool memFlag = !inputMem || !outputMem;
    CHK_PRT_RET(memFlag, HCCL_ERROR("[Create][CommP2P]inputMem is null or outputMem is null."), commP2PArray);

    commP2PArray.resize(ringSize); // ring_size即为网络平面数, 比如能组几条环

    bool invalidcheck = (rankVector_.size() <= userRank_) || (rankVector_.size() <= dstUserRank);
    CHK_PRT_RET(invalidcheck, HCCL_ERROR("[Create][CommP2P]dstUserRank[%u] or userRank[%u] is bigger than rank vector.",
        dstUserRank, userRank_), commP2PArray);

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        u32 rank = GetSubCollectiveRank(CommPlaneVector_[COMM_COMBINE][ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        IntraExchanger exchangerNetwork {};
        HCCL_INFO("[CreateCommP2PAsync] CommP2P is used %s. userRank = %u, rank = %u",
            isUsedRdmaOuter_ ? "rdma" : "sdma", userRank_, rank);
        commP2PArray[ringIndex].reset(new (std::nothrow) CommP2P(identifier_, userRank_, userRankSize_,
            rank, CommPlaneVector_[COMM_COMBINE][ringIndex].size(), TopoType::TOPO_TYPE_COMMON, dispatcher_,
            notifyPool_, netDevCtxMap_, exchangerNetwork, CommPlaneVector_[COMM_COMBINE][ringIndex], inputMem,
            outputMem, isUsedRdmaOuter_, transportResourceInfoAddr_, transportResourceInfoSize_, tag, dstUserRank,
            nicDeployInner_));

        CHK_PRT_RET(!commP2PArray[ringIndex], HCCL_ERROR("[Create][CommP2P]comm p2p array[%u] reset failed.",
            ringIndex), commP2PArray);
        if (commP2PArray[ringIndex]->BuildAsync(status) != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][CommP2P]comm p2p array[%u] init failed", ringIndex);
            commP2PArray[ringIndex].reset(nullptr);
            return commP2PArray;
        }
        HCCL_DEBUG("BuildAsync %u", status);
    }
    return commP2PArray;
}

HcclResult CommFactory::CreateCommP2PQuerry(std::vector<std::unique_ptr<CommBase> >& comm, u32& status)
{
    HcclResult ret;
    std::vector<u32> commStatus(comm.size());
    for (u32 index = 0; index < comm.size(); index++) {
        CHK_SMART_PTR_NULL(comm[index]);
        ret = comm[index]->BuildQuerry(commStatus[index]);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Querry][CommP2P]comm p2p array[%u] init failed", index);
            comm[index].reset(nullptr);
            return ret;
        }
    }
    status = (static_cast<int>(comm.size()) == std::count(commStatus.begin(), commStatus.end(), 0)) ? 0 : 1;
    HCCL_DEBUG("CreateCommP2PQuerry %u", status);
    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetServerIdx(const RankInfo &rankInfo, u32 &serverIdx) const
{
    // 通过ranktable指定集群信息场景，可以调整server在ranktable的排序(serverIdx）来指定server间通信的topo，从优化通信拓扑
    // rootInfo初始化场景，会自动收集集群信息，外部无法指定server的排序，可以无视serverIdx，使用serverID来代替
    // PS:返回的serverIdx，会影响rankMap_中server的排序，从而影响bridgeRank的选择，优化通信拓扑
    CHK_PRT_RET((rankInfo.serverIdx == INVALID_UINT), HCCL_ERROR("server idx is invalid."), HCCL_E_INTERNAL);
    serverIdx = rankInfo.serverIdx;
    return HCCL_SUCCESS;
}

// 集群中存在910B A+X时，0-7卡: moduleIdx = 2 * serverIdx; 8-15卡: moduleIdx = 2 * serverIdx + 1
// 集群中不存在910B A+X时，moduleIdx = serverIdx
HcclResult CommFactory::GetModuleIdx(const RankInfo &rankInfo, u32 &moduleIdx)
{
    // 获取moduleIdx，在16P同时使用左右两个module时，moduleIdx标识当前rank所在的module，其他场景下moduleIdx等同于serverIdx
    u32 serverIdx = 0;
    CHK_RET(GetServerIdx(rankInfo, serverIdx));
    if (IsDiffDeviceModuleInServer()) {
        moduleIdx = serverIdx * FACTOR_NUM_TWO + rankInfo.devicePhyId / DEVICE_PER_MODULE;
    } else {
        moduleIdx = serverIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetRankInfo()
{
    for (u32 index = 0; index < rankVector_.size(); index++) {
        if (userRank_ == rankVector_[index].userRank) {
            rankData_ = rankVector_[index];
            HCCL_INFO("[SetRankInfo]rankData_: userRank[%u], devicePhyId[%d], serverIdx[%u], superPodId[%s]",
                rankData_.userRank, rankData_.devicePhyId, rankData_.serverIdx, rankData_.superPodId.c_str());
            break;
        }
    }

    std::set<u32> serverIdxs;
    std::set<u32> moduleIdxs;
    for (u32 index = 0; index < rankVector_.size(); index++) {
        // 填充superPodRankMap_, 记录superPodId -> rankInfo
        auto itSuperPod = superPodToRank_.find(rankVector_[index].superPodId);
        if (itSuperPod != superPodToRank_.end()) {
            itSuperPod->second.push_back(rankVector_[index]);
        } else {
            std::vector<RankInfo> rankVecTmp;
            rankVecTmp.push_back(rankVector_[index]);
            superPodToRank_.insert(std::make_pair(rankVector_[index].superPodId, rankVecTmp));
        }

        u32 moduleIdx = 0;
        CHK_RET(GetModuleIdx(rankVector_[index], moduleIdx));
        moduleIdxs.insert(moduleIdx);
        // 填充serverRankMap_, 只记录本superPod下的serverIdx -> rankInfo
        if (rankVector_[index].superPodId == rankData_.superPodId) {
            auto itServer = serverToRank_.find(moduleIdx);
            if (itServer != serverToRank_.end()) {  // 存在该服务器内相关rank的对应信息
                itServer->second.push_back(rankVector_[index]);
            } else {  // 不存在则新增一条map记录
                std::vector<RankInfo> rankVecTmp;
                rankVecTmp.push_back(rankVector_[index]);
                serverToRank_.insert(std::make_pair(moduleIdx, rankVecTmp));
            }
        }

        // 同一个server内, 记录本rank和其他rank的链路
        if (rankVector_[index].serverIdx == rankData_.serverIdx) {
            LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
            if (rankData_.devicePhyId != rankVector_[index].devicePhyId &&
                rankData_.devicePhyId != HOST_DEVICE_ID &&
                rankVector_[index].devicePhyId != HOST_DEVICE_ID &&
                topoFlag_ != TopoType::TOPO_TYPE_HETEROG) {
                CHK_RET(hrtGetPairDeviceLinkType(rankData_.devicePhyId, rankVector_[index].devicePhyId, linkType));
            }
            deviceLinkTypeMap_.insert(std::make_pair(rankVector_[index].devicePhyId, linkType));
        }

        u32 serverIdx = 0;
        CHK_RET(GetServerIdx(rankVector_[index], serverIdx));
        serverIdxs.insert(serverIdx);
    }

    serverNum_ = serverIdxs.size();
    u32 rankNumPerAggregation = userRankSize_ / static_cast<u32>(moduleIdxs.size());

    // 调整每个server内的user_rank排序(server内device id从小到大,但不一定连续)
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            std::sort(iterMap->second.begin(), iterMap->second.end(), Ascending);
        }
    }

    // 调整每个superPod内的user_rank排序, 按照serverIdx从小到大、device id从小到大排序
    for (auto iterMap = superPodToRank_.begin(); iterMap != superPodToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            std::sort(iterMap->second.begin(), iterMap->second.end(), Ascending);
        }
    }

    ranksOneNode_ = { 0, 8, 4, 2, 1, 4, rankNumPerAggregation, 0, rankNumPerAggregation, rankNumPerAggregation};
    // 校验每个server内的设备个数与topo类型的组合是否正确
    if (topoFlag_ != TopoType::TOPO_TYPE_COMMON) {
        CHK_RET(CheckServerInfo());
    }
    // 校验每个superPod下的device数量相同
    CHK_RET(CheckSuperPodInfo());

    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetTopologyInfo()
{
    CHK_RET(SetTopoDefaultInfo());

    CHK_RET(SetTopoInfoForLevel0());
    CHK_RET(SetTopoInfoForLevel1());
    CHK_RET(SetTopoInfoForLevel2());

    // 是否支持按mesh划分通信拓扑
    bool isSupportMeshTopo = meshAggregationRankSize_ > 0 &&
                             userRankSize_ % meshAggregationRankSize_ == 0;
    if (isSupportMeshTopo) {
        CHK_RET(SetTopoInfoForMeshL0());
        CHK_RET(SetTopoInfoForMeshL1());
    } else {
        HCCL_INFO("[Set][TopologyInfo]topo is not support Mesh, meshAggregationRankSize_[%u], userRankSize_[%u]",
            meshAggregationRankSize_, userRankSize_);
    }
    CommPlaneVector_[COMM_COMBINE_ORDER].push_back(rankVector_);
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetTopoDefaultInfoFor8P()
{
    // 填充combined_rank_vector_:不区分board_type,只生成default单层拓扑
    std::vector<RankInfo> tmpCombinedVector;
    // 服务器内排序固定为0, 2, 3, 1, 5, 7, 6, 4，挑选8P多环中适用于combined的一组服务器内排序
    std::vector<u32> devOrder = { 0, 2, 3, 1, 5, 7, 6, 4 };
    // 维护topo输出的信息
    std::string outLogInfo = "userRank/devicePhyId: ";

    // 填充combined_rank_vector_的内层vector:combined场景只有一条固定的环
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        /* 服务器内8P满配单环特殊适配逻辑 */
        for (u32 index = 0; index < devOrder.size(); index++) {
            u32 devIndex = devOrder[index];
            u32 combinedUserRank = (iterMap->second)[devIndex].userRank;

            bool checkError = (rankVector_.size() <= combinedUserRank);
            CHK_PRT_RET(checkError,
                HCCL_ERROR("[Set][TopoDefaultInfoFor8P]combined userRank[%u] is bigger than rank vector",
                    combinedUserRank), HCCL_E_INTERNAL);

            RankInfo tmpCombinedPara = rankVector_[combinedUserRank];
            outLogInfo.append(std::to_string(tmpCombinedPara.userRank));
            outLogInfo.append("/");
            outLogInfo.append(std::to_string(tmpCombinedPara.devicePhyId));
            outLogInfo.append("; ");
            tmpCombinedVector.push_back(tmpCombinedPara);
        }
    }

    CommPlaneVector_[COMM_COMBINE].push_back(tmpCombinedVector);
    HCCL_RUN_INFO("combineTopo: identifier[%s], userRank[%u], userRankSize[%u] topoRankInfo[%s]",
        identifier_.c_str(), userRank_, userRankSize_, outLogInfo.c_str());
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetTopoDefaultInfo()
{
    // 填充combined_rank_vector_:不区分board_type,只生成default单层拓扑
    std::vector<RankInfo> tmpCombinedVector;

    bool incrementFlag = true; // 节点间建链的两个deviceID必须相同(同一个网段平面)，server间需要特殊处理
    // 维护topo输出的信息
    std::string outLogInfo = "userRank/devicePhyId: ";
    RankInfo tempRankData;

    // 填充combined_rank_vector_的内层vector:combined场景只有一条固定的环
    for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            // TOPO_TYPE_COMMON为单环拓扑场景，不需要执行如下判断设置流程
            if (topoFlag_ != TopoType::TOPO_TYPE_COMMON) {
                if (((iterMap->second).size() == SERVER_RANK_SIZE) && !IsDiffDeviceModuleInServer()
                    && (topoFlag_ != TopoType::TOPO_TYPE_HETEROG) && !Is310PDevice()) {
                    CHK_RET(SetTopoDefaultInfoFor8P()); // 服务器内dev个数相同已在hcom层做过校验
                    return HCCL_SUCCESS;
                }
            }

            if (incrementFlag) {
                for (u32 incrementIndex = 0; incrementIndex < (iterMap->second).size(); incrementIndex++) {
                    u32 combinedUserRank = (iterMap->second)[incrementIndex].userRank;
                    bool checkError = (rankVector_.size() <= combinedUserRank);
                    CHK_PRT_RET(checkError, HCCL_ERROR("[Set][TopoDefaultInfo]combined userRank[%u] is bigger than "\
                        "rank vector", combinedUserRank), HCCL_E_INTERNAL);
                    tempRankData = rankVector_[combinedUserRank];
                    outLogInfo.append(std::to_string(tempRankData.userRank));
                    outLogInfo.append("/");
                    outLogInfo.append(std::to_string(tempRankData.devicePhyId));
                    outLogInfo.append("; ");
                    tmpCombinedVector.push_back(tempRankData);
                }

                incrementFlag = false;
            } else {
                for (u32 decrementIndex = (iterMap->second).size(); decrementIndex > 0; decrementIndex--) {
                    u32 combinedUserRank = (iterMap->second)[decrementIndex - 1].userRank;
                    bool checkError = (rankVector_.size() <= combinedUserRank);
                    CHK_PRT_RET(checkError, HCCL_ERROR("[Set][TopoDefaultInfo]combined userRank[%u] is bigger than "\
                        "rank vector", combinedUserRank), HCCL_E_INTERNAL);
                    tempRankData = rankVector_[combinedUserRank];
                    outLogInfo.append(std::to_string(tempRankData.userRank));
                    outLogInfo.append("/");
                    outLogInfo.append(std::to_string(tempRankData.devicePhyId));
                    outLogInfo.append("; ");
                    tmpCombinedVector.push_back(tempRankData);
                }

                incrementFlag = true;
            }
        }
    }
    if (topoFlag_ == TopoType::TOPO_TYPE_COMMON) {
        std::sort(tmpCombinedVector.begin(), tmpCombinedVector.end(), CompareWithUserRankAscend);
    }

    CommPlaneVector_[COMM_COMBINE].push_back(tmpCombinedVector);
    HCCL_RUN_INFO("combineTopo: identifier[%s], userRank[%u], userRankSize[%u] topoRankInfo[%s]",
        identifier_.c_str(), userRank_, userRankSize_, outLogInfo.c_str());
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetSingleOuterFor8P()
{
    // 该函数处理场景:8P满配、非8P_RING算法(8P满配下走4PMESH)
    std::vector<RankInfo> tmpOuterVector;
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Set][SingleOuterFor8P]can't find serverId[%s] in rank map",
        rankData_.serverId.c_str()), HCCL_E_NOT_FOUND);

    u32 startIndex = (rankData_.devicePhyId < static_cast<s32>(meshAggregationRankSize_)) ?
        0 : meshAggregationRankSize_;
    u32 devcount = 0;
    // 维护topo输出的信息
    std::string outLogInfo = "userRank/devicePhyId: ";
    RankInfo tempRankData;
    while (devcount < meshAggregationRankSize_) {
        u32 outerStartRank = (iterRank->second)[startIndex].userRank;
        bool checkError = (rankVector_.size() <= outerStartRank);
        CHK_PRT_RET(checkError, HCCL_ERROR("[Set][SingleOuterFor8P]outer userRank[%u] is bigger than rank vector",
            outerStartRank), HCCL_E_INTERNAL);
        tempRankData = rankVector_[outerStartRank];

        outLogInfo.append(std::to_string(tempRankData.userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(tempRankData.devicePhyId));
        outLogInfo.append("; ");
        tmpOuterVector.push_back(tempRankData);
        startIndex++;
        devcount++;
    }

    // 4PMESH场景下，外层拓扑3个平面
    u32 outerSize = ranksOneNode_[static_cast<u32>(topoFlag_)] -1;
    for (u32 index = 0; index < outerSize; index++) {
        CommPlaneVector_[COMM_LEVEL0].push_back(tmpOuterVector);
    }
    HCCL_RUN_INFO("SetTopoInfoForLevel0: identifier[%s], userRank[%u], userRankSize[%u], topoRankInfo[%s]",
        identifier_.c_str(), userRank_, userRankSize_, outLogInfo.c_str());
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetSingleOuter()
{
    // 填充outer_rank_vector_，该函数处理场景非8P_RING算法
    std::vector<RankInfo> tmpOuterVector;
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Set][SingleOuter]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    std::vector<s32> devicePhyIdVector;
    for (u32 i = 0; i < rankVector_.size(); i++) {
        devicePhyIdVector.push_back(rankVector_[i].devicePhyId);
    }
    s32 maxPhyId = *max_element(devicePhyIdVector.begin(), devicePhyIdVector.end());
    // 8P满配场景:4PMESH算法 + 8Pfullmesh + 16P仅使用左边module
    if (((iterRank->second).size() == DEVICE_PER_MODULE &&
        maxPhyId < DEVICE_PER_MODULE) &&
        (topoFlag_ == TopoType::TOPO_TYPE_4P_MESH || topoFlag_ == TopoType::TOPO_TYPE_NP_MESH)) {
        return SetSingleOuterFor8P(); // 服务器内dev个数相同已在hcom层做过校验
    }

    // 维护topo输出的信息
    std::string outLogInfo = "userRank/devicePhyId: ";
    RankInfo tempRankData;
    // 其他场景 + 16P使用右边module
    for (u32 startIndex = 0; startIndex < (iterRank->second).size(); startIndex++) {
        u32 outerStartRank = (iterRank->second)[startIndex].userRank;
        bool checkError = (rankVector_.size() <= outerStartRank);
        CHK_PRT_RET(checkError, HCCL_ERROR("[Set][SingleOuter]outer userRank[%u] is bigger than rank vector",
            outerStartRank), HCCL_E_INTERNAL);
        tempRankData = rankVector_[outerStartRank];

        outLogInfo.append(std::to_string(tempRankData.userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(tempRankData.devicePhyId));
        outLogInfo.append("; ");
        tmpOuterVector.push_back(tempRankData);
    }

    // NPmesh或4Pmesh场景下，外层拓扑平面为device数量-1
    u32 outerSize = (topoFlag_ == TopoType::TOPO_TYPE_4P_MESH || topoFlag_ == TopoType::TOPO_TYPE_NP_MESH) ?
        (ranksOneNode_[static_cast<u32>(topoFlag_)] - 1) : 1;

    for (u32 index = 0; index < outerSize; index++) {
        CommPlaneVector_[COMM_LEVEL0].push_back(tmpOuterVector);
    }
    HCCL_RUN_INFO("SetTopoInfoForLevel0: identifier[%s], userRank[%u], userRankSize[%u], topoRankInfo[%s]",
        identifier_.c_str(), userRank_, userRankSize_, outLogInfo.c_str());
    return HCCL_SUCCESS;
}

// 适配ROH平面网段隔离，奇数rank互通，偶数rank互通，奇偶不通
bool CheckSdmaWithRohTopo(const std::vector<u32> &nicList, std::vector<u32> &topoList)
{
    std::vector<u32> tmpNicList(nicList);
    std::sort(tmpNicList.begin(), tmpNicList.end());
    SearchPath searchPath;
    topoList = searchPath.Search(tmpNicList);
    if (topoList.empty()) {
        return false;
    }
    return true;
}

std::vector<std::vector<u32>> GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType, std::vector<u32> &nicList)
{
    std::vector<std::vector<u32>> multiRingOrder;
    if (topoType == TopoType::TOPO_TYPE_8P_RING) { // 4 ring 场景
        // 每个环的排序是按照设备物理ID进行的
        std::vector<u32> tmpOuter0 = { 0, 1, 2, 6, 5, 4, 7, 3 }; // 环0
        std::vector<u32> tmpOuter1 = { 0, 3, 7, 4, 5, 6, 2, 1 }; // 环1
        std::vector<u32> tmpOuter2 = { 0, 2, 3, 1, 5, 7, 6, 4 }; // 环2
        std::vector<u32> tmpOuter3 = { 0, 4, 6, 7, 5, 1, 3, 2 }; // 环3

        // 填充8pring 多环的comm outer 四个环的顺序
        multiRingOrder.push_back(tmpOuter0);
        multiRingOrder.push_back(tmpOuter1);
        multiRingOrder.push_back(tmpOuter2);
        multiRingOrder.push_back(tmpOuter3);
    } else if (topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING) { // 2 ring 场景
        std::vector<u32> tmpOuter0;   // 环0
        std::vector<u32> tmpOuter1;  // 环1
        std::vector<u32> rohOuter;
        if (GetExternalInputEnableRdmaSdmaConcurrent() && (CheckSdmaWithRohTopo(nicList, rohOuter))) {
            tmpOuter0 = rohOuter;          // 环0, 8卡 { 0, 1, 3, 2, 4, 5, 7, 6 };
            tmpOuter1.reserve(ranksSize);  // 环1, 8卡 { 0, 6, 7, 5, 4, 2, 3, 1 };
            tmpOuter1.push_back(rohOuter[0]);
            tmpOuter1.insert(tmpOuter1.end(), rohOuter.rbegin(), rohOuter.rend() - 1);
        } else {
            tmpOuter0 = nicList;  // { 0, 1, 2, 3, 4, 5, 6, 7 };
            tmpOuter1.reserve(ranksSize);
            tmpOuter1.push_back(nicList[0]);
            tmpOuter1.insert(tmpOuter1.end(), tmpOuter0.rbegin(), tmpOuter0.rend() - 1);
        }
        HCCL_INFO("[GetRingsOrderByTopoType] TopoType:TOPO_TYPE_NP_DOUBLE_RING");
        // 填充 double ring 两环的comm outer的顺序
        multiRingOrder.push_back(tmpOuter0);
        multiRingOrder.push_back(tmpOuter1);
    } else { // 1 ring 场景
        std::vector<u32> tmpOuter0 = nicList; // 环0

        // 填充 single ring 单环的comm outer的顺序
        multiRingOrder.push_back(tmpOuter0);
    }
    // 打印多个环
    for (size_t i = 0; i < multiRingOrder.size(); i++) {
        auto ring = multiRingOrder[i];
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = ring.begin(); it != ring.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string ringString = stringRepresentation.str();
        const char *charRing = ringString.c_str();
        HCCL_DEBUG("[GetRingsOrderByTopoType] The No.%zu ring: %s", i, charRing);
    }
    return multiRingOrder;
}

HcclResult CommFactory::SetMultiOuter(u32 ringNum)
{
    std::vector<u32> tmpOuterOrder;
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Set][MultiOuter]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    // 维护topo输出的信息
    std::string outLogInfo = "";
    RankInfo tempRankData;
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        tmpOuterOrder = multiOuterOrder_[ringIndex]; // 获取每一个环的设备物理ID排序
        std::vector<RankInfo> tmpOuterVector;
        outLogInfo = "userRank/devicePhyId: ";
        for (u32 startIndex = 0; startIndex < (iterRank->second).size(); startIndex++) {
            u32 devIndex = tmpOuterOrder[startIndex];
            u32 outerRingUserank = (iterRank->second)[devIndex].userRank;
            bool checkError = (rankVector_.size() <= outerRingUserank);
            CHK_PRT_RET(checkError, HCCL_ERROR("[Set][MultiOuter]outer userRank[%u] is bigger than rank vector",
                outerRingUserank), HCCL_E_INTERNAL);
            tempRankData = rankVector_[outerRingUserank];
            outLogInfo.append(std::to_string(tempRankData.userRank));
            outLogInfo.append("/");
            outLogInfo.append(std::to_string(tempRankData.devicePhyId));
            outLogInfo.append("; ");
            tmpOuterVector.push_back(tempRankData);
        }
        HCCL_RUN_INFO("SetTopoInfoForLevel0: identifier[%s], userRank[%u], userRankSize[%u], topoRankInfo[%s]",
            identifier_.c_str(), userRank_, userRankSize_, outLogInfo.c_str());
        CommPlaneVector_[COMM_LEVEL0].push_back(tmpOuterVector);
        // SDMA&RDMA并发特性
        if (GetExternalInputEnableRdmaSdmaConcurrent()) {
            CommPlaneVector_[COMM_LEVEL0_RDMA].push_back(tmpOuterVector);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetTopoInfoForLevel0()
{
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Set][TopoInfoForLevel0]can't find serverId[%s] in rank map",
                                  rankData_.serverId.c_str()), HCCL_E_NOT_FOUND);
    // 查询本rank所在服务器的rank数
    u32 ranksSize = (iterRank->second).size();

    multiOuterOrder_.clear();
    // 生成mockNicList
    std::vector<u32> mockNicList;
    mockNicList.reserve(ranksSize);
    for (u32 startIndex = 0; startIndex < ranksSize; startIndex++) {
        mockNicList.push_back(startIndex);
    }

    multiOuterOrder_ = GetRingsOrderByTopoType(ranksSize, topoFlag_, mockNicList);

    HCCL_DEBUG("[CommFactory] The ring number is %zu, the rank size is %lu.", multiOuterOrder_.size(), ranksSize);
    if (multiOuterOrder_.size() == 1) {
        CHK_RET(SetSingleOuter());
    } else {    // 8p-ring/np ring 环场景
        u32 ringNum = multiOuterOrder_.size();
        CHK_RET(SetMultiOuter(ringNum)); // 8P_RING场景下，外层拓扑中有四个环; 910_73场景中适配双环
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetTopoInfoForLevel1()
{
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Set][TopoInfoForLevel1]can't find serverId[%s] in rank map",
                                  rankData_.serverId.c_str()), HCCL_E_NOT_FOUND);

    u32 ringSize;
    if (topoFlag_ == TopoType::TOPO_TYPE_2P_MESH) { // 2P_MESH在任何情况下，内层拓扑平面始终为2
        ringSize = ranksOneNode_[static_cast<u32>(topoFlag_)];
    } else if (topoFlag_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) { // 标卡内层拓扑的环数
        ringSize = (iterRank->second).size();
    } else { // 其他场景下内层拓扑平面为每个module中的device数量
        ringSize = ranksOneNode_[static_cast<u32>(topoFlag_)];
    }

    // 内层拓扑的每层环
    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        std::vector<RankInfo> tmpBridgeVector;
        bool bridgeRankFlag = false;
        std::string outLogInfo = ""; // 维护topo输出的信息
        outLogInfo.append(" ringIndex: ");
        outLogInfo.append(std::to_string(ringIndex));
        outLogInfo.append(", ");
        outLogInfo.append("userRank/serverId/devicePhyId/nicIp/isBridgeRank: ");

        // 2、填充bridge_rank_vector_的内层vector和is_bridge_vector_
        for (auto iterMap = serverToRank_.begin(); iterMap != serverToRank_.end(); iterMap++) {
            if (!(iterMap->second).empty()) {
                RankInfo tmpBridgePara;
                u32 bridgeUserRank = (iterMap->second)[ringIndex].userRank;
                u32 bridgeDevicePhyId = (iterMap->second)[ringIndex].devicePhyId;
                std::vector<u32> bridgeNic((iterMap->second)[ringIndex].nicIdx);
                bool checkError = (rankVector_.size() <= bridgeUserRank);
                CHK_PRT_RET(checkError, HCCL_ERROR("[Set][TopoInfoForLevel1]bridge userRank[%u] is bigger than rank "\
                    "vector", bridgeUserRank), HCCL_E_INTERNAL);
                CHK_RET(SetBridgeLinkInfo(tmpBridgePara, bridgeUserRank));
                tmpBridgeVector.push_back(tmpBridgePara);
                std::vector<u32>::iterator iterNic = std::find(bridgeNic.begin(), bridgeNic.end(), bridgeDevicePhyId);
                if ((bridgeNic.size() == 0) || (iterNic != bridgeNic.end())) {
                    if (bridgeUserRank == static_cast<u32>(userRank_)) { // 本rank是否为bridge_rank
                        bridgeRankFlag = true;
                    }
                }

                outLogInfo.append(std::to_string(tmpBridgePara.userRank));
                outLogInfo.append("/");
                outLogInfo.append(tmpBridgePara.serverId);
                outLogInfo.append("/");
                outLogInfo.append(std::to_string(tmpBridgePara.devicePhyId));
                outLogInfo.append("/");
                outLogInfo.append(tmpBridgePara.nicIp[0].GetReadableAddress());
                outLogInfo.append("/");
                outLogInfo.append(std::to_string(bridgeRankFlag));
                outLogInfo.append("; ");
            }
        }

        // 3、填充bridge_rank_vector_、isBridgeVector_
        isBridgeVector_.push_back(bridgeRankFlag);
        CommPlaneVector_[COMM_LEVEL1].push_back(tmpBridgeVector);

        if (GetExternalInputEnableRdmaSdmaConcurrent()) {
            CommPlaneVector_[COMM_LEVEL1_RDMA].push_back(tmpBridgeVector);
        }
        HCCL_INFO("SetTopoInfoForLevel1: topoRankInfo[%s]", outLogInfo.c_str());
    }

    HCCL_RUN_INFO("SetTopoInfoForLevel1: identifier[%s], userRank[%u], userRankSize[%u], plane size[%u]",
        identifier_.c_str(), userRank_, userRankSize_, CommPlaneVector_[COMM_LEVEL1].size());
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetTopoInfoForLevel2()
{
    // 找到当前rank在本超节点内部的序号
    auto it = superPodToRank_.find(rankData_.superPodId);
    CHK_PRT_RET(it == superPodToRank_.end(),
        HCCL_ERROR("[Set][TopoInfoForLevel2]superPodId[%s] is not exist in superPodRankMap",
        rankData_.superPodId.c_str()), HCCL_E_INTERNAL);

    u32 index = 0;
    for (; index < it->second.size(); ++index) {
        if (userRank_ == it->second[index].userRank) {
            break;
        }
    }
    CHK_PRT_RET(index >= it->second.size(),
        HCCL_ERROR("[Set][TopoInfoForLevel2]userRank_[%u] superPodId[%s] is not exist in superPodRankMap",
        userRank_, rankData_.superPodId.c_str()), HCCL_E_INTERNAL);

    std::vector<RankInfo> tmpRankVec;
    for (auto iterMap = superPodToRank_.begin(); iterMap != superPodToRank_.end(); iterMap++) {
        CHK_PRT_RET(iterMap->second.size() <= index,
            HCCL_ERROR("[Set][TopoInfoForLevel2]index[%u] is bigger than rank vector size[%u]",
            index, iterMap->second.size()), HCCL_E_INTERNAL);

        RankInfo& tempRankData = iterMap->second[index];
        tmpRankVec.push_back(tempRankData);

        // 维护topo输出的信息
        std::string outLogInfo = "userRank/devicePhyId/serverIdx/superPodId: ";
        outLogInfo.append(std::to_string(tempRankData.userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(tempRankData.devicePhyId));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(tempRankData.serverIdx));
        outLogInfo.append("/");
        outLogInfo.append(tempRankData.superPodId);
        outLogInfo.append("; ");
        HCCL_INFO("SetTopoInfoForLevel2: topoRankInfo[%s]", outLogInfo.c_str());
    }

    CommPlaneVector_[COMM_LEVEL2].push_back(tmpRankVec);
    HCCL_RUN_INFO("SetTopoInfoForLevel2: identifier[%s], userRank[%u], userRankSize[%u], plane size[%u]",
        identifier_.c_str(), userRank_, userRankSize_, CommPlaneVector_[COMM_LEVEL2].size());
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetTopoInfoForMeshL0()
{
    // 以MeshAggregation为粒度、MeshAggregation内各设备的mesh建链
    u32 rankSize = meshAggregationRankSize_;
    u32 userRankIndexBegin = userRank_ / meshAggregationRankSize_ * meshAggregationRankSize_;
    u32 userRankIndexEnd = userRankIndexBegin + meshAggregationRankSize_;
    std::vector<RankInfo> paraVector(rankSize);
    u32 rankIndex = 0;
    std::string outLogInfo = "userRank/devicePhyId: "; // 维护topo输出的信息

    CHK_PRT_RET(rankVector_.size() < userRankIndexEnd,
        HCCL_ERROR("[Set][TopoInfoForMeshL0]rankVector_ size[%u] should be greater than userRankIndexEnd[%u]",
            rankVector_.size(), userRankIndexEnd), HCCL_E_PARA);

    for (u32 i = userRankIndexBegin; i < userRankIndexEnd; i ++) {
        paraVector[rankIndex] = rankVector_[i];
        outLogInfo.append(std::to_string(paraVector[rankIndex].userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(paraVector[rankIndex].devicePhyId));
        outLogInfo.append("; ");
        rankIndex++;
    }
    CommPlaneVector_[COMM_MESH_L0].push_back(paraVector);
    HCCL_RUN_INFO("SetTopoInfoForMeshL0: identifier[%s], userRank[%u], userRankSize[%u], topoRankInfo[%s]",
        identifier_.c_str(), userRank_, userRankSize_, outLogInfo.c_str());
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetTopoInfoForMeshL1()
{
    // 以MeshAggregation为粒度、MeshAggregation间各平面的mesh建链
    u32 rankSize = userRankSize_ /  meshAggregationRankSize_; // 1 = 7 / 4
    u32 planeID = userRank_ % meshAggregationRankSize_; // 0
    std::vector<RankInfo> paraVector(rankSize);

    CHK_PRT_RET(rankVector_.size() < userRankSize_,
        HCCL_ERROR("[Set][TopoInfoForMeshL1]rankVector_ size[%u] should be greater than userRankSize[%u]",
            rankVector_.size(), userRankSize_), HCCL_E_PARA);

    for (u32 i = planeID; i < userRankSize_; i += meshAggregationRankSize_) {
        u32 rankIndex = i / meshAggregationRankSize_;
        paraVector[rankIndex] = rankVector_[i];
        std::string outLogInfo = "userRank/devicePhyId"; // 维护topo输出的信息
        outLogInfo.append(std::to_string(paraVector[rankIndex].userRank));
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(paraVector[rankIndex].devicePhyId));
        outLogInfo.append("; ");
        HCCL_INFO("SetTopoInfoForMeshL1: topoRankInfo[%s]", outLogInfo.c_str());
    }
    CommPlaneVector_[COMM_MESH_L1].push_back(paraVector);
    HCCL_RUN_INFO("SetTopoInfoForMeshL1: identifier[%s], userRank[%u], userRankSize[%u], plane size[%u]",
        identifier_.c_str(), userRank_, userRankSize_, CommPlaneVector_[COMM_MESH_L1].size());
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetBridgeLinkInfo(RankInfo &bridgePara, u32 bridgeUserRank)
{
    bool checkSize = (rankVector_.size() <= bridgeUserRank);
    CHK_PRT_RET(checkSize,
        HCCL_ERROR("[Set][BridgeLinkInfo]bridge UserRank %u is bigger than rank vector", bridgeUserRank),
        HCCL_E_INTERNAL);

    bridgePara = rankVector_[bridgeUserRank];
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CheckInitInfo()
{
    if (rankVector_.size() == 0) {
        HCCL_ERROR("[Check][InitInfo]Not support the scenes: rank_vector size is zero");
        return HCCL_E_PARA;  // 没有rank_table的场景直接报错
    }

    // 构造函数入参有效性检查:user_rank_size与user_rank_
    if (userRankSize_ <= userRank_) {
        HCCL_ERROR("[Check][InitInfo]userRankSize_[%u] or userRank_[%u] is invalid.", userRankSize_, userRank_);
        return HCCL_E_PARA;
    }

    if (userRankSize_ != rankVector_.size()) {
        HCCL_ERROR("[Check][InitInfo]userRankSize_[%u] is not equal to rank_vector size[%llu].", userRankSize_,\
            rankVector_.size());
        return HCCL_E_PARA;
    }

    bool isParaInvalid = ((topoFlag_ == TopoType::TOPO_TYPE_RESERVED) || (deviceType_ >= DevType::DEV_TYPE_COUNT));
    if (isParaInvalid) {
        HCCL_ERROR("[Check][InitInfo]Not support the scenes: TopoType[%d] or deviceType[%d] is invalid.",
            topoFlag_, deviceType_);
        return HCCL_E_PARA;
    }

    // 入参组合有效性检查:不支持4P_RING
    if ((deviceType_ == DevType::DEV_TYPE_910 || deviceType_ == DevType::DEV_TYPE_910B ||
         deviceType_ == DevType::DEV_TYPE_910_73) && (topoFlag_ == TopoType::TOPO_TYPE_4P_RING)) {
        HCCL_ERROR("[Check][InitInfo]Not support the scenes: TopoType[%d] with deviceType[%d] is invalid.", topoFlag_,
            deviceType_);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult CommFactory::CheckServerInfo()
{
    /*
     * HCOM模块：
     * 1、每个AI server之间的芯片个数必须一致，不一致则报错；
     * 2、每个AI server之间的芯片ID（device ID）必须相同（server0里面devID分别是0、1、4、5；server1->server127也必须是相同的）
     *   ，不一致则报错；
     * HCCL API模块：
     * 3、校验rank_table传进来devID，与rt_get_device查询到的devID，是否相同，不一致则报错（针对当前设备）
     * 因此，上层模块已经校验过的不再重复，本函数仅用于校验每个server内的设备个数与topo类型的组合是否正确
     */
    u32 moduleIdx = 0;
    CHK_RET(GetModuleIdx(rankData_, moduleIdx));
    auto iterRank = serverToRank_.find(moduleIdx); // 查询本rank所在服务器
    bool check = (iterRank == serverToRank_.end());
    CHK_PRT_RET(check,
        HCCL_ERROR("[Check][ServerInfo]can't find serverId[%s] in rank map", rankData_.serverId.c_str()),
        HCCL_E_NOT_FOUND);

    HcclResult ret = HCCL_SUCCESS;

    switch (topoFlag_) {
        case TopoType::TOPO_TYPE_NP_MESH:
        case TopoType::TOPO_TYPE_4P_MESH:
        case TopoType::TOPO_TYPE_2P_MESH:
        case TopoType::TOPO_TYPE_1P_MESH: {  // 4p_mesh场景下，支持server(4P+4P)和server(4P)+server(4P)，2p_mesh/1p_mesh同理
            ret = (((iterRank->second).size() == ranksOneNode_[static_cast<u32>(topoFlag_)]) ||
                   ((iterRank->second).size() == 2 * ranksOneNode_[static_cast<u32>(topoFlag_)])) // 2表示8P满配走4PMESH算法
                        ? HCCL_SUCCESS
                        : HCCL_E_UNAVAIL;
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Check][ServerInfo]check server info err:server rank size[%llu], expected "\
                "value[%u], topo type[%d]", (iterRank->second).size(), ranksOneNode_[static_cast<u32>(topoFlag_)],
                topoFlag_), HCCL_E_UNAVAIL);
            break;
        }
        case TopoType::TOPO_TYPE_NP_SINGLE_RING:
            ret = ((iterRank->second).size() == ranksOneNode_[static_cast<u32>(topoFlag_)]) ? HCCL_SUCCESS :
                HCCL_E_UNAVAIL;
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Check][ServerInfo]check server info err:server rank size[%llu], expected "\
                "less than value[%u], topo type[%d]", (iterRank->second).size(),
                ranksOneNode_[static_cast<u32>(topoFlag_)], topoFlag_), HCCL_E_UNAVAIL);
            break;
        case TopoType::TOPO_TYPE_HETEROG:
        case TopoType::TOPO_TYPE_ES_MESH:
            break;
        default: {  // 8P_RING or 4P_RING
            ret = ((iterRank->second).size() == ranksOneNode_[static_cast<u32>(topoFlag_)]) ? HCCL_SUCCESS :
                HCCL_E_UNAVAIL;
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Check][ServerInfo]check server info err:server rank size[%llu], expected "\
                "value[%u], topo type[%d]", (iterRank->second).size(),
                ranksOneNode_[static_cast<u32>(topoFlag_)], topoFlag_), HCCL_E_UNAVAIL);
            break;
        }
    }

    HCCL_INFO("check server info:server rank size[%llu], expected value[%u], topo type[%d]",
              (iterRank->second).size(),
              ranksOneNode_[static_cast<u32>(topoFlag_)],
              topoFlag_);
    return ret;
}

// 校验每个超节点内的dev数量相同
HcclResult CommFactory::CheckSuperPodInfo()
{
    for (auto iter = superPodToRank_.begin(); iter != superPodToRank_.end(); iter++) {
        u32 devNum = superPodToRank_.begin()->second.size();
        u32 curDevNum = iter->second.size();
        CHK_PRT_RET(devNum != curDevNum,
            HCCL_ERROR("[Check][SuperPodInfo]devNum[%u] in superPod[%s] is inconsistent with "\
            "devNum[%u] in superPod[%s].", devNum, superPodToRank_.begin()->first.c_str(),
            curDevNum, iter->first.c_str()), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::CheckPlaneInfo()
{
    bool isTopoComm = (topoFlag_ == TopoType::TOPO_TYPE_COMMON) && (CommPlaneVector_[COMM_COMBINE].size() != 1);
    CHK_PRT_RET(isTopoComm,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d] and combined plane nub[%llu] are not match",
        topoFlag_, CommPlaneVector_[COMM_COMBINE].size()), HCCL_E_INTERNAL);

    bool isTopo8pring = (topoFlag_ == TopoType::TOPO_TYPE_8P_RING) &&
        ((CommPlaneVector_[COMM_LEVEL0].size() != meshAggregationRankSize_) ||
        (ranksOneNode_[static_cast<u32>(topoFlag_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(isTopo8pring,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d], outer plane nub[%llu], inner plane nub[%llu], is not match",
        topoFlag_, CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()), HCCL_E_INTERNAL);

    bool isTopo2pring = (topoFlag_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) &&
        ((CommPlaneVector_[COMM_LEVEL0].size() != 2) || // 2表示一个节点内通信域里面是否只有2个ring
        (ranksOneNode_[static_cast<u32>(topoFlag_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(isTopo2pring,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d], outer plane nub[%llu], inner plane nub[%llu], is not match",
        topoFlag_, CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()), HCCL_E_INTERNAL);

    bool isTopo4pRing = (topoFlag_ == TopoType::TOPO_TYPE_4P_RING) &&
        ((CommPlaneVector_[COMM_LEVEL0].size() != 1) ||    // 1表示一个节点内通信域里面是否只有一个device
        (ranksOneNode_[static_cast<u32>(topoFlag_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(isTopo4pRing,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d], outer plane nub[%llu], inner plane nub[%llu], is not match",
        topoFlag_, CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()), HCCL_E_INTERNAL);

    bool isTopo4pMesh = (topoFlag_ == TopoType::TOPO_TYPE_4P_MESH) &&
        ((CommPlaneVector_[COMM_LEVEL0].size() !=  (ranksOneNode_[static_cast<u32>(topoFlag_)] - 1)) ||
        (ranksOneNode_[static_cast<u32>(topoFlag_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(isTopo4pMesh,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d], outer plane nub[%llu], inner plane nub[%llu], is not match",
        topoFlag_, CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()), HCCL_E_INTERNAL);

    bool isTopoNpMesh = (topoFlag_ == TopoType::TOPO_TYPE_NP_MESH) &&
        ((CommPlaneVector_[COMM_LEVEL0].size() != (ranksOneNode_[static_cast<u32>(topoFlag_)] - 1)) ||
        (ranksOneNode_[static_cast<u32>(topoFlag_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(isTopoNpMesh,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d], outer plane nub[%llu], inner plane nub[%llu], is not match",
        topoFlag_, CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()), HCCL_E_INTERNAL);

    // 1表示一个module里面是否只有一个device
    bool isTopo2pMesh = (topoFlag_ == TopoType::TOPO_TYPE_2P_MESH) &&
        ((CommPlaneVector_[COMM_LEVEL0].size() != 1) ||
        (ranksOneNode_[static_cast<u32>(topoFlag_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(isTopo2pMesh,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d], outer plane nub[%llu], inner plane nub[%llu], is not match",
        topoFlag_, CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()), HCCL_E_INTERNAL);

    bool isTopo1pMesh = (topoFlag_ == TopoType::TOPO_TYPE_1P_MESH) &&
        ((CommPlaneVector_[COMM_LEVEL0].size() != 1) ||    // 1表示一个节点内通信域里面是否只有一个device
        (ranksOneNode_[static_cast<u32>(topoFlag_)] != CommPlaneVector_[COMM_LEVEL1].size()));
    CHK_PRT_RET(isTopo1pMesh,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d], outer plane nub[%llu], inner plane nub[%llu], is not match",
        topoFlag_, CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()), HCCL_E_INTERNAL);

    bool isTopoNpSingleRing = (topoFlag_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) && (!IsDiffDeviceModuleInServer()) &&
        ((CommPlaneVector_[COMM_LEVEL0].size() != 1) ||
        (CommPlaneVector_[COMM_LEVEL1].size() != ranksOneNode_[static_cast<u32>(topoFlag_)]));
    CHK_PRT_RET(isTopoNpSingleRing,
        HCCL_ERROR("[Check][PlaneInfo]topo type[%d], outer plane nub[%llu], inner plane nub[%llu], is not match",
        topoFlag_, CommPlaneVector_[COMM_LEVEL0].size(), CommPlaneVector_[COMM_LEVEL1].size()), HCCL_E_INTERNAL);

    HCCL_RUN_INFO(
        "plane info:topo type[%d], device type[%d], COMM_COMBINE size[%llu], COMM_LEVEL0 size[%llu], COMM_LEVEL1 " \
        "size[%llu], COMM_LEVEL2 size[%llu], COMM_MESH_L0 size[%llu], COMM_MESH_L1 size[%llu]",
        topoFlag_, deviceType_, CommPlaneVector_[COMM_COMBINE].size(), CommPlaneVector_[COMM_LEVEL0].size(),
        CommPlaneVector_[COMM_LEVEL1].size(), CommPlaneVector_[COMM_LEVEL2].size(),
        CommPlaneVector_[COMM_MESH_L0].size(), CommPlaneVector_[COMM_MESH_L1].size());

    return HCCL_SUCCESS;
}

u32 CommFactory::GetSubRootUserRank(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;
    if ((rankVector_.size() > userRank) && (rankVector_.size() > rootUserRank)) {
        u32 moduleIdx = 0;
        CHK_PRT_RET(GetModuleIdx(rankVector_[rootUserRank], moduleIdx) != HCCL_SUCCESS,
            HCCL_ERROR("[Get][SubRootUserRank]get server id failed."), INVALID_VALUE_RANKID);

        auto iterRankRoot = serverToRank_.find(moduleIdx);
        CHK_PRT_RET(iterRankRoot == serverToRank_.end(),
            HCCL_ERROR("[Get][SubRootUserRank]can't find root serverId[%s] in rank map",
                rankVector_[rootUserRank].serverId.c_str()), INVALID_VALUE_RANKID);

        CHK_PRT_RET(GetModuleIdx(rankVector_[userRank], moduleIdx) != HCCL_SUCCESS,
            HCCL_ERROR("[Get][SubRootUserRank]get server id failed."), INVALID_VALUE_RANKID);

        auto iterRankCurr = serverToRank_.find(moduleIdx);
        CHK_PRT_RET(iterRankCurr == serverToRank_.end(),
            HCCL_ERROR("[Get][SubRootUserRank]can't find local serverId[%s] in rank map",
                rankVector_[userRank].serverId.c_str()), INVALID_VALUE_RANKID);

        for (u32 index = 0; index < (iterRankCurr->second).size(); index++) {
            /* 当userRank的server内rank号与rootUserRank所在服务器中某一个server内rank号相同，
            获取出rootUserRank所在服务器内的userrank */
            if (userRank == (iterRankCurr->second)[index].userRank) {
                tmpUserRank = (iterRankRoot->second)[index].userRank;
                break;
            }
        }
    }
    return tmpUserRank;
}

u32 CommFactory::GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;
    if ((rankVector_.size() <= userRank) || (rankVector_.size() <= rootUserRank)) {
        return tmpUserRank;
    }

    std::string rootSuperPod = rankVector_[rootUserRank].superPodId;
    auto iterRankRoot = superPodToRank_.find(rootSuperPod);
    CHK_PRT_RET(iterRankRoot == superPodToRank_.end(),
        HCCL_ERROR("[Get][GetSubRootUserRankWithSuperPod]can't find root rootSuperPod[%s] in rank map",
            rootSuperPod.c_str()), INVALID_VALUE_RANKID);

    std::string userSuperPod = rankVector_[userRank].superPodId;
    auto iterRankCurr = superPodToRank_.find(userSuperPod);
    CHK_PRT_RET(iterRankCurr == superPodToRank_.end(),
        HCCL_ERROR("[Get][GetSubRootUserRankWithSuperPod]can't find local userSuperPod[%s] in rank map",
            userSuperPod.c_str()), INVALID_VALUE_RANKID);

    for (u32 index = 0; index < (iterRankCurr->second).size(); index++) {
        /* 当userRank的superPod内rank号与rootUserRank所在服务器中某一个superPod内rank号相同，
        获取出rootUserRank所在服务器内的userrank */
        if (userRank == (iterRankCurr->second)[index].userRank) {
            tmpUserRank = (iterRankRoot->second)[index].userRank;
            break;
        }
    }

    return tmpUserRank;
}

u32 CommFactory::GetSubRootForScatter(const u32 root)
{
    // 通过root找到ringIndex, 通过userRank找到Inner中的rank
    u32 subRoot = INVALID_VALUE_RANKID;
    u32 planeIdx = INVALID_VALUE_RANKID;
    u32 ringSize = CommPlaneVector_[COMM_LEVEL1].size();

    CHK_PRT_RET(CommPlaneVector_[COMM_LEVEL1].size() == 0,
        HCCL_ERROR("[GET][GetSubRootForScatter]bridgeRankVector size is zero."), HCCL_E_PARA);

    u32 rank = INVALID_VALUE_RANKID;
    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (isBridgeVector_[ringIndex]) {
            rank = GetSubCollectiveRank(CommPlaneVector_[COMM_LEVEL1][ringIndex]);       // 确定userRank在Inner中的rank号
        }
        for (u32 idx = 0; idx < CommPlaneVector_[COMM_LEVEL1][ringIndex].size(); idx++) {
            if (root == CommPlaneVector_[COMM_LEVEL1][ringIndex][idx].userRank) {       // 获取root所在的平面
                planeIdx = ringIndex;
            }
        }
    }
    CHK_PRT_RET(rank == INVALID_VALUE_RANKID,
        HCCL_ERROR("[GET][GetSubRootForScatter]get rankId in inner failed."), HCCL_E_PARA);
    CHK_PRT_RET(planeIdx == INVALID_VALUE_RANKID,
        HCCL_ERROR("[GET][GetSubRootForScatter]get root[%u] planeIdx[%u] failed.", root, planeIdx), HCCL_E_PARA);
    subRoot = CommPlaneVector_[COMM_LEVEL1][planeIdx][rank].userRank;
    HCCL_DEBUG("[GetSubRootForScatter] userRank_:[%u] subRoot:[%u]", userRank_, subRoot);
    return subRoot;
}

const u32 CommFactory::GetSubCollectiveRank(const std::vector<RankInfo> &vecPara) const
{
    // 在vecPara数据中，查询本user rank，查询到的vec下标就是rank值
    u32 tmpRank = INVALID_VALUE_RANKID;

    for (u32 rankIndex = 0; rankIndex < vecPara.size(); rankIndex++) {
        if (userRank_ == vecPara[rankIndex].userRank) {
            tmpRank = rankIndex;
            break;
        }
    }

    return tmpRank;
}

u32 CommFactory::GetInnerCommRank(const u32 ringIdx)
{
    return GetSubCollectiveRank(CommPlaneVector_[COMM_LEVEL1][ringIdx]);
}

bool CommFactory::JudgmentSetHeterogP2p(u32 rank) const
{
    return isHaveCpuRank_;
}

HcclResult CommFactory::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
    std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    rankDevicePhyIdNicInfoMap_ = rankDevicePhyIdNicInfoMap;
    ranksPort_ = ranksPort;
    isSetHDCModeInfo_ = isSetHDCModeInfo;
    isUseRankPort_ = isUseRankPort;
    return HCCL_SUCCESS;
}

/*
 * *********************************************************************************
 * 用来标识集群中是否存在910B A+X形态
 * **********************************************************************************
 */
bool CommFactory::IsDiffDeviceModuleInServer() const
{
    return deviceType_ == DevType::DEV_TYPE_910B && isDiffAggregation_;
}

HcclResult CommFactory::SetIsUsedRdma(const CommParaInfo &commParaInfo,
    std::vector<SingleSubCommTransport> &commTransport, bool isUsedRdma)
{
    u32 ringSize = commTransport.size();

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        SingleSubCommTransport &subCommTransport = commTransport[ringIndex];
        subCommTransport.isUsedRdma = isUsedRdma;
    }
    HCCL_INFO("[CommFactory][SetIsUsedRdma] commPlane[%d] isUsedRdma[%d]", commParaInfo.commPlane, isUsedRdma);
    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetRankMap(const CommParaInfo &commParaInfo, std::vector<SingleSubCommTransport> &commTransport)
{
    u32 ringSize = commTransport.size();

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        SingleSubCommTransport &subCommTransport = commTransport[ringIndex];
        // 有建链诉求，则记录从userRank到subCommRank 和 从subCommRank到userRank的映射
        if (subCommTransport.transportRequests.size() != 0) {
            if (commParaInfo.commType == CommType::COMM_TAG_PARTIAL_MESH_COMBINED) {
                CHK_RET(GetSub2UserRankMap(commParaInfo.commPlane, 0, subCommTransport.subCommRank2UserRank));
                CHK_RET(GetUserRank2SubMap(commParaInfo.commPlane, 0, subCommTransport.userRank2subCommRank));
            } else {
                CHK_RET(GetSub2UserRankMap(commParaInfo.commPlane, ringIndex, subCommTransport.subCommRank2UserRank));
                CHK_RET(GetUserRank2SubMap(commParaInfo.commPlane, ringIndex, subCommTransport.userRank2subCommRank));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::SetRankMap()
{
    // 构建由UserRank到子通信域的映射
    subCommRank2UserRank_.resize(static_cast<u32>(COMM_LEVEL_RESERVED));
    userRank2subCommRank_.resize(static_cast<u32>(COMM_LEVEL_RESERVED));

    for (u32 levelIndex = 0; levelIndex < CommPlaneVector_.size(); levelIndex++) {
        u32 ringSize = CommPlaneVector_[levelIndex].size();
        subCommRank2UserRank_[levelIndex].resize(ringSize);
        userRank2subCommRank_[levelIndex].resize(ringSize);
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            u32 rankSize = CommPlaneVector_[levelIndex][ringIndex].size();
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                u32 userRank = CommPlaneVector_[levelIndex][ringIndex][rankIndex].userRank;
                subCommRank2UserRank_[levelIndex][ringIndex][rankIndex] = userRank;
                userRank2subCommRank_[levelIndex][ringIndex][userRank] = rankIndex;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetSub2UserRankMap(CommPlane commPlane, u32 ringIndex,
    std::map<u32, u32> &subCommRank2UserRank)
{
    subCommRank2UserRank = subCommRank2UserRank_[static_cast<u32>(commPlane)][ringIndex];
    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetUserRank2SubMap(CommPlane commPlane, u32 ringIndex,
    std::map<u32, u32> &userRank2subCommRank)
{
    userRank2subCommRank = userRank2subCommRank_[static_cast<u32>(commPlane)][ringIndex];
    return HCCL_SUCCESS;
}

HcclResult CommFactory::GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>> &CommPlaneRanks)
{
    CommPlaneRanks.resize(CommPlaneVector_.size());
    for (u32 level = 0; level < CommPlaneVector_.size(); level ++) {
        u32 ringSize = CommPlaneVector_[level].size();
        CommPlaneRanks[level].resize(ringSize);
        for (u32 ringIndex = 0 ; ringIndex < ringSize; ringIndex ++) {
            u32 rankSize = CommPlaneVector_[level][ringIndex].size();
            CommPlaneRanks[level][ringIndex].resize(rankSize);
            for (u32 rankIndex = 0 ; rankIndex < rankSize; rankIndex ++) {
                u32 userRank = CommPlaneVector_[level][ringIndex][rankIndex].userRank;
                CommPlaneRanks[level][ringIndex][rankIndex] = userRank;
            }
        }
    }
    return HCCL_SUCCESS;
}
HcclResult CommFactory::GetIsBridgeVector(std::vector<bool> &isBridgeVector)
{
    isBridgeVector = isBridgeVector_;
    return HCCL_SUCCESS;
}
}  // namespace hccl
