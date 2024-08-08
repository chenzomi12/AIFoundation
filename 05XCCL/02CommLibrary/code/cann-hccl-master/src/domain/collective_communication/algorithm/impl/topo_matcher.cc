/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 内部依赖头文件
#include <condition_variable>
#include "dispatcher.h"
#include "comm_base_pub.h"
#include "externalinput_pub.h"
#include "coll_alg_param.h"
#include "topo_matcher.h"
#include "search_path.h"
#include "calc_p2p_transport_req.h"
namespace hccl {

TopoMatcher::TopoMatcher(const std::vector<std::vector<std::vector<u32>>> CommPlaneRanks,
                         std::vector<bool> isBridgeVector,
                         HcclTopoInfo &topoInfo,
                         HcclAlgoInfo &algoInfo,
                         HcclExternalEnable &externalEnable,
                         std::vector<std::vector<std::vector<u32>>> &serverAndsuperPodToRank)
    : CommPlaneVector_(CommPlaneRanks), isBridgeVector_(isBridgeVector),
      topoInfo_(topoInfo), algoInfo_(algoInfo), externalEnable_(externalEnable), userRank_(topoInfo.userRank),
      serverAndsuperPodToRank_(serverAndsuperPodToRank)
{
    SetRankMap();
}

HcclResult TopoMatcher::CalcCommPlaneInfo(const std::string &tag, const CommParaInfo &commParaInfo,
    std::vector<SingleSubCommTransport> &commTransport, TransportMemType inputMemType, TransportMemType outputMemType)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO("[Calc][CommPlane]tag[%s], commPlane[%d], commType[%d]",
        tag.c_str(), commParaInfo.commPlane, commParaInfo.commType);

    u32 subUserRankRoot = INVALID_VALUE_RANKID;
    if (commParaInfo.root != INVALID_VALUE_RANKID) {
        subUserRankRoot = GetSubRootUserRank(userRank_, commParaInfo.root);
        if (subUserRankRoot == INVALID_VALUE_RANKID) {
            HCCL_ERROR("[TopoMatcher][CalcCommPlaneInfo]get sub root userrank value[%u] invalid.", subUserRankRoot);
            return HCCL_E_PARA;
        }
    }

    std::unique_ptr<CalcTransportReqBase> calcTransportReq;
    switch (commParaInfo.commType) {
        case CommType::COMM_TAG_RING_INNER:
        case CommType::COMM_TAG_RING_COMBINED: {
            calcTransportReq.reset(new (std::nothrow) CalcRingTransportReq(CommPlaneVector_[commParaInfo.commPlane],
                isBridgeVector_, userRank_));
            ret = calcTransportReq->CalcTransportRequest(tag, inputMemType, outputMemType, commParaInfo, commTransport);
            break;
        }
        case CommType::COMM_TAG_HALVING_DOUBLING: {
            calcTransportReq.reset(new (std::nothrow) CalcHDTransportReq(CommPlaneVector_[commParaInfo.commPlane],
                isBridgeVector_, userRank_));
            ret = calcTransportReq->CalcTransportRequest(tag, inputMemType, outputMemType, commParaInfo, commTransport,
                subUserRankRoot);
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING:
        case CommType::COMM_TAG_WHOLE_NHR:{
            calcTransportReq.reset(new (std::nothrow) CalcNHRTransportReq(CommPlaneVector_[commParaInfo.commPlane],
                isBridgeVector_, userRank_));
            ret = calcTransportReq->CalcTransportRequest(tag, inputMemType, outputMemType, commParaInfo, commTransport);
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1:
        case CommType::COMM_TAG_WHOLE_NHR_V1: {
            calcTransportReq.reset(new (std::nothrow) CalcNHRV1TransportReq(CommPlaneVector_[commParaInfo.commPlane],
                isBridgeVector_, userRank_));
            ret = calcTransportReq->CalcTransportRequest(tag, inputMemType, outputMemType, commParaInfo, commTransport);
            break;
        }
        case CommType::COMM_TAG_NONUNIFORM_BRUCK:
        case CommType::COMM_TAG_WHOLE_NB: {
            calcTransportReq.reset(new (std::nothrow) CalcNBTransportReq(CommPlaneVector_[commParaInfo.commPlane],
                isBridgeVector_, userRank_));
            ret = calcTransportReq->CalcTransportRequest(tag, inputMemType, outputMemType, commParaInfo, commTransport);
            break;
        }
        case CommType::COMM_TAG_MESH: {
            calcTransportReq.reset(new (std::nothrow) CalcMeshTransportReq(CommPlaneVector_[commParaInfo.commPlane],
                isBridgeVector_, userRank_));
            ret = calcTransportReq->CalcTransportRequest(tag, inputMemType, outputMemType, commParaInfo, commTransport);
            break;
        }
        case CommType::COMM_TAG_PARTIAL_MESH_COMBINED: {
            calcTransportReq.reset(new (std::nothrow) CalcPartialMeshTransportReq
                (CommPlaneVector_[commParaInfo.commPlane], isBridgeVector_, userRank_));
            ret = calcTransportReq->CalcTransportRequest(tag, inputMemType, outputMemType, commParaInfo, commTransport);
            break;
        }
        case CommType::COMM_TAG_P2P: {
            calcTransportReq.reset(new (std::nothrow) CalcP2PTransportReq(CommPlaneVector_[commParaInfo.commPlane],
                isBridgeVector_, userRank_));
            ret = calcTransportReq->CalcTransportRequest(tag, inputMemType, outputMemType, commParaInfo, commTransport);
            break;
        }
        default: {
            HCCL_ERROR("[Calc][CommPlane]commType[%d] is invalid", commParaInfo.commType);
            return HCCL_E_PARA;
        }
    }

    CHK_RET(SetIsUsedRdma(commParaInfo, commTransport));
    CHK_RET(GetRankMap(commParaInfo, commTransport));

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Calc][CommPlane]failed, tag[%s], commPlane[%d], commType[%d]",
        tag.c_str(), commParaInfo.commPlane, commParaInfo.commType), ret);

    HCCL_INFO("complete commPlane[%d] commType[%d] Calculation, Time:%lld us",
        commParaInfo.commPlane, commParaInfo.commType, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetRankMap(const CommParaInfo &commParaInfo, std::vector<SingleSubCommTransport> &commTransport)
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

HcclResult TopoMatcher::SetRankMap()
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
                u32 userRank = CommPlaneVector_[levelIndex][ringIndex][rankIndex];
                subCommRank2UserRank_[levelIndex][ringIndex][rankIndex] = userRank;
                userRank2subCommRank_[levelIndex][ringIndex][userRank] = rankIndex;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetIsUsedRdma(const CommParaInfo &commParaInfo, bool &isUsedRdma)
{
    std::vector<std::vector<u32> > commP2PPlaneVec;
    if (commParaInfo.commType == CommType::COMM_TAG_P2P) {
        // P2P只需要判断两张卡之间的连接关系
        bool invalidcheck = (topoInfo_.isUsedRdmaMap.size() <= topoInfo_.userRank) ||
                            (topoInfo_.isUsedRdmaMap.size() <= commParaInfo.peerUserRank);
        CHK_PRT_RET(invalidcheck, HCCL_ERROR("[GetIsUsedRdma]dstUserRank[%u] or userRank[%u] is bigger than "\
            "rankVector size[%u]", commParaInfo.peerUserRank, topoInfo_.userRank, topoInfo_.isUsedRdmaMap.size()),
            HCCL_E_PARA);

        std::vector<u32> commP2PRankVec;
        commP2PRankVec.push_back(topoInfo_.userRank);
        commP2PRankVec.push_back(commParaInfo.peerUserRank);
        commP2PPlaneVec.push_back(commP2PRankVec);
    }

    std::vector<std::vector<u32> > &commPlaneVec = (commParaInfo.commType == CommType::COMM_TAG_P2P) ?
        commP2PPlaneVec : CommPlaneVector_[commParaInfo.commPlane];

    for (const std::vector<u32> &commPlane : commPlaneVec) {
        for (const u32 dstRank : commPlane) {
            if (topoInfo_.isUsedRdmaMap[dstRank]) {
                isUsedRdma = true;
                return HCCL_SUCCESS;
            }
        }
    }
    isUsedRdma = false;
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::SetIsUsedRdma(const CommParaInfo &commParaInfo,
    std::vector<SingleSubCommTransport> &commTransport)
{
    bool isUsedRdma = false;
    CHK_RET(GetIsUsedRdma(commParaInfo, isUsedRdma));
    isUsedRdma = (GetExternalInputEnableRdmaSdmaConcurrent() && topoInfo_.deviceType == DevType::DEV_TYPE_910_73) ?
        commParaInfo.forceRdma : isUsedRdma;
    u32 ringSize = commTransport.size();

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        SingleSubCommTransport &subCommTransport = commTransport[ringIndex];
        subCommTransport.isUsedRdma = isUsedRdma;
    }
    HCCL_INFO("[TopoMatcher][SetIsUsedRdma] commPlane[%d] isUsedRdma[%d]", commParaInfo.commPlane, isUsedRdma);
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetSub2UserRankMap(CommPlane commPlane, u32 ringIndex,
    std::map<u32, u32> &subCommRank2UserRank)
{
    subCommRank2UserRank = subCommRank2UserRank_[static_cast<u32>(commPlane)][ringIndex];
    return HCCL_SUCCESS;
}

HcclResult TopoMatcher::GetUserRank2SubMap(CommPlane commPlane, u32 ringIndex,
    std::map<u32, u32> &userRank2subCommRank)
{
    userRank2subCommRank = userRank2subCommRank_[static_cast<u32>(commPlane)][ringIndex];
    return HCCL_SUCCESS;
}

HcclTopoInfo TopoMatcher::GetTopoInfo()
{
    return topoInfo_;
}

HcclAlgoInfo TopoMatcher::GetAlgoInfo()
{
    return algoInfo_;
}

u32 TopoMatcher::GetExternalInputEnableRdmaSdmaConcurrent()
{
    return externalEnable_.enableRdmaSdmaConcurrent;
}

u32 TopoMatcher::GetExternalInputHcclEnableFfts()
{
    return externalEnable_.enableFfts;
}

u32 TopoMatcher::GetExternalInputHcclDeterministic()
{
    return externalEnable_.deterministic;
}

u32 TopoMatcher::GetExternalInputHcclHighPerfEnable()
{
    return externalEnable_.highPerfEnable;
}

u32 TopoMatcher::GetExternalInputIntraRoceSwitch()
{
    return externalEnable_.intraRoceSwitch;
}

u32 TopoMatcher::GetExternalInputHcclDumpDebug()
{
    return externalEnable_.dumpDebug;
}

bool CheckRankNeighbors(const std::vector<u32> &nicList)
{
    // 组成ROH环路必须偶数个,且2节点不能组成双环？
    if (nicList.size() % 2 != 0 || nicList.size() < HCCL_DEVICE_NUM_FOUR) {
        return false;
    }

    std::vector<u32> tmpNicList(nicList);
    std::sort(tmpNicList.begin(), tmpNicList.end());
    u32 halfNum = 2;
    for (u32 i = 0; i < tmpNicList.size() / halfNum; i++) {
        auto nicIndex = i * halfNum;
        // 检查相邻下标的节点，devID是否相邻
        if (tmpNicList[nicIndex] + 1 != tmpNicList[nicIndex + 1]) {
            return false;
        }
    }

    return true;
}

// 适配ROH平面网段隔离，奇数rank互通，偶数rank互通，奇偶不通
bool TopoMatcher::CheckSdmaWithRohTopo(const std::vector<u32> &nicList, std::vector<u32> &topoList)
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

const u32 TopoMatcher::GetSubCollectiveRank(const std::vector<u32> &vecPara) const
{
    // 在vecPara数据中，查询本user rank，查询到的vec下标就是rank值
    u32 tmpRank = INVALID_VALUE_RANKID;

    for (u32 rankIndex = 0; rankIndex < vecPara.size(); rankIndex++) {
        if (userRank_ == vecPara[rankIndex]) {
            tmpRank = rankIndex;
            break;
        }
    }

    return tmpRank;
}

u32 TopoMatcher::GetSubRootForScatter(const u32 root)
{
    // 通过root找到ringIndex, 通过userRank找到Inner中的rank
    u32 subRoot = INVALID_VALUE_RANKID;
    u32 planeIdx = INVALID_VALUE_RANKID;
    u32 ringSize = CommPlaneVector_[COMM_LEVEL1_INDEX].size();

    CHK_PRT_RET(ringSize == 0, HCCL_ERROR("[GET][GetSubRootForScatter]bridgeRankVector size is zero."), HCCL_E_PARA);

    u32 rank = INVALID_VALUE_RANKID;
    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (isBridgeVector_[ringIndex]) {
            rank = GetSubCollectiveRank(CommPlaneVector_[COMM_LEVEL1_INDEX][ringIndex]); // 确定userRank在Inner中的rank号
        }
        for (u32 idx = 0; idx < CommPlaneVector_[COMM_LEVEL1_INDEX][ringIndex].size(); idx++) {
            if (root == CommPlaneVector_[COMM_LEVEL1_INDEX][ringIndex][idx]) {  // 获取root所在的平面
                planeIdx = ringIndex;
            }
        }
    }
    CHK_PRT_RET(rank == INVALID_VALUE_RANKID,
        HCCL_ERROR("[GET][GetSubRootForScatter]get rankId in inner failed."), HCCL_E_PARA);
    CHK_PRT_RET(planeIdx == INVALID_VALUE_RANKID,
        HCCL_ERROR("[GET][GetSubRootForScatter]get root[%u] planeIdx[%u] failed.", root, planeIdx), HCCL_E_PARA);
    subRoot = CommPlaneVector_[COMM_LEVEL1_INDEX][planeIdx][rank];
    HCCL_DEBUG("[GetSubRootForScatter] userRank_:[%u] subRoot:[%u]", userRank_, subRoot);
    return subRoot;
}

u32 TopoMatcher::GetSubRootUserRank(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;

    u32 serverIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[0].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[0][0].size(); j++) {
            if (serverAndsuperPodToRank_[0][i][j] == rootUserRank) {
                serverIdx = i;
                break;
            }
        }
    }
    u32 rankIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[0].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[0][0].size(); j++) {
            if (serverAndsuperPodToRank_[0][i][j] == userRank) {
                rankIdx = j;
                break;
            }
        }
    }

    if (serverIdx != INVALID_VALUE_RANKID && rankIdx != INVALID_VALUE_RANKID) {
        tmpUserRank = serverAndsuperPodToRank_[0][serverIdx][rankIdx];
    }
    return tmpUserRank;
}

u32 TopoMatcher::GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank)
{
    u32 tmpUserRank = INVALID_VALUE_RANKID;

    u32 superPodIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[1].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[1][0].size(); j++) {
            if (serverAndsuperPodToRank_[1][i][j] == rootUserRank) {
                superPodIdx = i;
                break;
            }
        }
    }
    u32 rankIdx = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < serverAndsuperPodToRank_[1].size(); i++) {
        for (u32 j = 0; j < serverAndsuperPodToRank_[1][0].size(); j++) {
            if (serverAndsuperPodToRank_[1][i][j] == userRank) {
                rankIdx = j;
                break;
            }
        }
    }

    if (superPodIdx != INVALID_VALUE_RANKID && rankIdx != INVALID_VALUE_RANKID) {
        tmpUserRank = serverAndsuperPodToRank_[1][superPodIdx][rankIdx];
    }
    return tmpUserRank;
}

HcclResult TopoMatcher::SetDeterministicConfig(const u8 deterministic)
{
    if (deterministic > 1) {
        HCCL_ERROR("[SetDeterministicConfig] deterministic[%d] should be 0 or 1.");
        return HCCL_E_PARA;
    }
    externalEnable_.deterministic = deterministic;
    return HCCL_SUCCESS;
}

u8 TopoMatcher::GetDeterministicConfig() const
{
    return externalEnable_.deterministic;
}

}