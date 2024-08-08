/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_star.h"

constexpr s32 NORMAL_QP_MODE = 0;
constexpr s32 OFFLINE_QP_MODE = 1;
constexpr s32 OPBASE_QP_MODE = 2;
constexpr s32 OFFLINE_QP_MODE_EXT = 3;  // 下沉模式(910B/91073)QP
constexpr s32 OPBASE_QP_MODE_EXT = 4;  // 单算子模式(910B/91073)的QP

namespace hccl {
constexpr s32 MODULE_TYPE_SYSTEM = 0;
constexpr s32 INFO_TYPE_VERSION = 1;
constexpr u32 DEV_TYPE_DIGIT_NUM = 8;
constexpr u32 DEV_TYPE_DIGIT_MASK = 0xff00;

CommStar::CommStar(const std::string &collectiveId, const u32 userRank,
    const u32 userRankSize, const u32 rank, const u32 rankSize, const TopoType topoFlag,
    const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool,
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
    const IntraExchanger &exchanger, const std::vector<RankInfo> paraVector,
    const DeviceMem& inputMem, const DeviceMem& outputMem, const bool isUsedRdmaOuter,
    const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
    const std::string &tag,
    const NICDeployment nicDeployInner, const u32 subUserRankRoot) : CommBase(collectiveId,
        userRank, userRankSize, rank, rankSize, paraVector, topoFlag, dispatcher, notifyPool, netDevCtxMap, exchanger,
        inputMem, outputMem, isUsedRdmaOuter, transportResourceInfoAddr, transportResourceInfoSize, tag,
        nicDeployInner, false, false, false, subUserRankRoot)
{
    IsHostUseDevNic(isHostUseDevNic_);
    HCCL_DEBUG("CommStar isSetHDCModeInfo_[%d] isHostUseDevNic_ is[%d]", isSetHDCModeInfo_, isHostUseDevNic_);
}

CommStar::~CommStar()
{
}

HcclResult CommStar::CalcLink()
{
    if (rank_ == subUserRankRoot_) {
        for (u32 dstRank = 0; dstRank < rankSize_; dstRank++) {
            if (dstRank != rank_) {
                HCCL_INFO("CommStar CalcLink i[%u] am root, dst rank %u", rank_, dstRank);
                HcclResult ret = CalcLinksNum(MachineType::MACHINE_SERVER_TYPE, dstRank);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Calc][Link]comm Star calc links num failed, type[%d], dstRank[%u]",
                    static_cast<int32_t>(MachineType::MACHINE_SERVER_TYPE), dstRank), ret);
            }
        }
    } else {
        HCCL_INFO("CommStar CalcLink i[%u] am not root, dst rank %u", rank_, subUserRankRoot_);
        HcclResult ret = CalcLinksNum(MachineType::MACHINE_CLIENT_TYPE, subUserRankRoot_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Calc][Link]comm Star calc links num failed, type[%d], Root[%u]",
            static_cast<int32_t>(MachineType::MACHINE_CLIENT_TYPE), subUserRankRoot_), ret);
    }

    return HCCL_SUCCESS;
}

HcclResult CommStar::MakeClientInfo(const u32 dstRank, RankInfo &dstRankInfo, bool isInterRdma, bool isInterHccs)
{
    if (isInterRdma && !isInterHccs) {
        HcclRankLinkInfo tempLinkInfo {};
        tempLinkInfo.userRank = dstRankInfo.userRank;

        std::string remoteHostIp(dstRankInfo.nicIp[0].GetReadableAddress());
        std::string LocalHostIp(paraVector_[rank_].nicIp[0].GetReadableAddress());
        if (rankDevicePhyIdNicInfoMap_.find(remoteHostIp) != rankDevicePhyIdNicInfoMap_.end() &&
            remoteHostIp != LocalHostIp && dstRankInfo.devicePhyId == HOST_DEVICE_ID) {
            tempLinkInfo.ip = rankDevicePhyIdNicInfoMap_[remoteHostIp][devicePhyId_];
            tempLinkInfo.devicePhyId = devicePhyId_;
        } else {
            tempLinkInfo.ip = dstRankInfo.nicIp[0];
            tempLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
        }

        tempLinkInfo.socketsPerLink = GetSocketsPerLink();

        tempLinkInfo.port = GetInterRemotePort(tempLinkInfo.devicePhyId, dstRankInfo.userRank);

        auto iter = dstInterClientMap_.find(dstRank);
        bool check = (iter != dstInterClientMap_.end());
        CHK_PRT_RET(check, HCCL_ERROR("[Make][ClientInfo]dstRank[%u] already exists in dst inter client map. ",
            dstRank), HCCL_E_PARA);
        dstInterClientMap_.insert(std::make_pair(dstRank, tempLinkInfo));
    } else {
        dstIntraClientVec_.push_back(dstRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CommStar::MakeServerInfo(const u32 dstRank, RankInfo &dstRankInfo, bool isInterRdma, bool isInterHccs)
{
    // 节点间或者是节点内采用RDMA通信的，放至dst_inter_client_map_,采用rdma建链
    if (isInterRdma && !isInterHccs) {
        HcclRankLinkInfo tempLinkInfo {};
        tempLinkInfo.userRank = dstRankInfo.userRank;

        std::string remoteHostIp(dstRankInfo.nicIp[0].GetReadableAddress());
        std::string LocalHostIp(paraVector_[rank_].nicIp[0].GetReadableAddress());
        if (rankDevicePhyIdNicInfoMap_.find(remoteHostIp) != rankDevicePhyIdNicInfoMap_.end() &&
            remoteHostIp != LocalHostIp && dstRankInfo.devicePhyId == HOST_DEVICE_ID) {
            tempLinkInfo.ip = rankDevicePhyIdNicInfoMap_[remoteHostIp][devicePhyId_];
            tempLinkInfo.devicePhyId = devicePhyId_;
        } else {
            tempLinkInfo.ip = dstRankInfo.nicIp[0];
            tempLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
        }

        tempLinkInfo.socketsPerLink = GetSocketsPerLink();

        tempLinkInfo.port = GetInterRemotePort(tempLinkInfo.devicePhyId, dstRankInfo.userRank);

        auto iter = dstInterServerMap_.find(dstRank);
        bool check = (iter != dstInterServerMap_.end());
        CHK_PRT_RET(check, HCCL_ERROR("[Make][ServerInfo]dstRank[%u] already exists in dst inter server map",
            dstRank), HCCL_E_PARA);
        dstInterServerMap_.insert(std::make_pair(dstRank, tempLinkInfo));
    } else {
        dstIntraServerVec_.push_back(dstRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CommStar::CreateInterLinks()
{
    HcclResult ret = HCCL_SUCCESS;
    u32 targetDevicePhyId;
    u32 deviceLogicId;
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > serverSocketsMap;
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > clientSocketsMap;

    if (dstInterServerMap_.size() + dstInterClientMap_.size() == 0) {
        HCCL_DEBUG("[Create][InterLinks] do not need create links.");
        return HCCL_SUCCESS;
    }

    if (paraVector_[rank_].devicePhyId == HOST_DEVICE_ID && isHostUseDevNic_) {
        std::string hostIp(paraVector_[rank_].hostIp.GetReadableAddress());
        for (auto phyNicInfo : rankDevicePhyIdNicInfoMap_[hostIp]) {
            targetDevicePhyId = phyNicInfo.first;
            CHK_RET(hrtGetDeviceIndexByPhyId(targetDevicePhyId, deviceLogicId));

            pyhIdResourseSockets_[targetDevicePhyId].reset(
                new (std::nothrow) HcclSocketManager(nicDeployInner_, deviceLogicId, targetDevicePhyId, userRank_));
            CHK_PTR_NULL(pyhIdResourseSockets_[targetDevicePhyId]);
            CHK_RET(pyhIdResourseSockets_[targetDevicePhyId]->Init());
            HCCL_DEBUG("[Create][InterLinks] dstInterServerMap size[%u], dstInterClientMap size[%u]",
                dstInterServerMap_.size(), dstInterClientMap_.size());

            for (auto &serverInfo : dstInterServerMap_) {
                if (targetDevicePhyId == serverInfo.second.devicePhyId) {
                    HCCL_DEBUG("[Create][InterLinks] targetDevicePhyId[%u], phyNicInfo.second[%s] serverInfo "
                        "dstRank[%u] serverInfo.second.devicePhyId[%u]", targetDevicePhyId,
                        phyNicInfo.second.GetReadableAddress(), serverInfo.first, serverInfo.second.devicePhyId);
                    ret = pyhIdResourseSockets_[targetDevicePhyId]->CreateSockets(tag_, true,
                        netDevCtxMap_[phyNicInfo.second], dstInterServerMap_, dstInterClientMap_,
                        serverSocketsMap, clientSocketsMap);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[Create][InterLinks] socket manager create connections failed, ret[%u]", ret), ret);
                    break;
                }
            }
        }
    } else {
        interSocketManager_.reset(
            new (std::nothrow) HcclSocketManager(nicDeployInner_, deviceLogicId_, devicePhyId_, userRank_));
        CHK_PTR_NULL(interSocketManager_);
        CHK_RET(interSocketManager_->Init());

        HCCL_INFO("[Create][InterLinks] dstInterServerMap size[%u], dstInterClientMap size[%u]",
            dstInterServerMap_.size(), dstInterClientMap_.size());

        ret = interSocketManager_->CreateSockets(tag_, true, netDevCtxMap_[paraVector_[rank_].nicIp[0]],
            dstInterServerMap_, dstInterClientMap_, serverSocketsMap, clientSocketsMap);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][InterLinks] socket manager create connections failed, ret[%u]", ret), ret);
    }

    return CreateLinksThread(serverSocketsMap, clientSocketsMap);
}

HcclResult CommStar::CreateLinksThread(
    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> &serverSocketsMap,
    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> &clientSocketsMap)
{
    HcclResult ret = HCCL_SUCCESS;
    for (auto &sockets : clientSocketsMap) {
        ret = CreateInterThread(CLIENT_ROLE_SOCKET, sockets.first, sockets.second);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][InterLinks] create inter thread failed, socket role[CLIENT_ROLE_SOCKET] "),
            ret);
    }

    for (auto &sockets : serverSocketsMap) {
        ret = CreateInterThread(SERVER_ROLE_SOCKET, sockets.first, sockets.second);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][InterLinks] create inter thread failed, socket role[SERVER_ROLE_SOCKET] "),
            ret);
    }

    return ret;
}

HcclResult CommStar::GetDevIP(const HcclIpAddress& hostIp, const u32& devicePhyId,
    HcclIpAddress& ip)
{
    std::string hostIpStr(hostIp.GetReadableAddress());
    CHK_PRT_RET(rankDevicePhyIdNicInfoMap_.find(hostIpStr) == rankDevicePhyIdNicInfoMap_.end() ||
        rankDevicePhyIdNicInfoMap_[hostIpStr].find(devicePhyId) ==
        rankDevicePhyIdNicInfoMap_[hostIpStr].end(), HCCL_ERROR("Get available device nic info fail,"\
            "hostIp[%s] devicePhyId[%u]", hostIpStr.c_str(), devicePhyId), HCCL_E_PARA);
    ip = rankDevicePhyIdNicInfoMap_[hostIpStr][devicePhyId];
    HCCL_DEBUG("Get available device nic info success, hostIp[%s] devicePhyId[%u] device ip[%s]",
        hostIpStr.c_str(), devicePhyId, ip.GetReadableAddress());

    return HCCL_SUCCESS;
}

HcclResult CommStar::SetMachinePara(MachineType machineType, const std::string &serverId, u32 dstRank,
    const std::vector<std::shared_ptr<HcclSocket> > &sockets, MachinePara &machinePara)
{
    CommBase::SetMachinePara(machineType, serverId, dstRank, sockets, machinePara);
    std::string localHostIpStr(paraVector_[rank_].hostIp.GetReadableAddress());
    std::string remoteHostIpStr(paraVector_[dstRank].hostIp.GetReadableAddress());
    if (paraVector_[rank_].devicePhyId == HOST_DEVICE_ID &&
        paraVector_[dstRank].devicePhyId != HOST_DEVICE_ID &&
        rankDevicePhyIdNicInfoMap_.find(localHostIpStr) != rankDevicePhyIdNicInfoMap_.end()) {
        CHK_PRT(GetDevIP(paraVector_[rank_].hostIp, paraVector_[dstRank].devicePhyId,
            machinePara.localIpAddr));

        u32 deviceLogicId;
        u32 phyId = static_cast<u32>(paraVector_[dstRank].devicePhyId);
        CHK_RET(hrtGetDeviceIndexByPhyId(phyId, deviceLogicId));
        machinePara.deviceLogicId = deviceLogicId;
    }
    if (paraVector_[dstRank].devicePhyId == HOST_DEVICE_ID &&
        paraVector_[rank_].devicePhyId != HOST_DEVICE_ID &&
        rankDevicePhyIdNicInfoMap_.find(remoteHostIpStr) != rankDevicePhyIdNicInfoMap_.end()) {
        CHK_PRT(GetDevIP(paraVector_[dstRank].hostIp, paraVector_[rank_].devicePhyId,
            machinePara.remoteIpAddr));
    }

    HCCL_INFO("selfIp[%s] selfPort[%u] peerIp[%s] peerPort[%u] deviceLogicId[%u].",
        machinePara.localIpAddr.GetReadableAddress(), machinePara.localSocketPort,
        machinePara.remoteIpAddr.GetReadableAddress(), machinePara.localSocketPort, machinePara.deviceLogicId);
    return HCCL_SUCCESS;
}

void CommStar::SetTransportParam(TransportPara &para, MachinePara &machinePara)
{
    CommBase::SetTransportParam(para, machinePara);
    para.selfIp = &machinePara.localIpAddr;
    para.selfPort = machinePara.localSocketPort;
    para.peerIp = &machinePara.remoteIpAddr;
    para.peerPort = machinePara.remoteSocketPort;
    para.proxyDevLogicId = machinePara.deviceLogicId;
    HCCL_INFO("SetTransportParam proxyDevLogicId[%u] deviceType is %d",
        para.proxyDevLogicId, paraVector_[rank_].deviceType);

    if (paraVector_[rank_].deviceType == DevType::DEV_TYPE_NOSOC) {
        para.qpMode = NORMAL_QP_MODE;
        para.devLogicId = HOST_DEVICE_ID;
        para.isHdcMode = false;
        para.remoteIsHdc = GetRemoteIsHdc();
        para.isESPs = true;
        HCCL_INFO("selfIp[%s] selfPort[%u] peerIp[%s] peerPort[%u] qpMode[%d].", para.selfIp->GetReadableAddress(),
            para.selfPort, para.peerIp->GetReadableAddress(), para.peerPort, para.qpMode);
        return;
    }

    if (paraVector_[rank_].devicePhyId == HOST_DEVICE_ID) {
        if (paraVector_[rank_].deviceType == DevType::DEV_TYPE_910B) {
            para.qpMode = OPBASE_QP_MODE_EXT;
        } else {
            para.qpMode = OPBASE_QP_MODE;
        }
        para.devLogicId = HOST_DEVICE_ID;
        para.remoteIsHdc = GetRemoteIsHdc();
        para.isESPs = true;
    } else {
        if (paraVector_[rank_].deviceType == DevType::DEV_TYPE_910B) {
            para.qpMode = (GetWorkflowMode() ==
                HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? OPBASE_QP_MODE_EXT : OFFLINE_QP_MODE_EXT);
        } else {
            para.qpMode = (GetWorkflowMode() ==
                HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? OPBASE_QP_MODE : OFFLINE_QP_MODE);
        }
        para.devLogicId = machinePara.deviceLogicId;
    }
    para.isHdcMode = true;
    HCCL_INFO("selfIp[%s] selfPort[%u] peerIp[%s] peerPort[%u] qpMode[%d].", para.selfIp->GetReadableAddress(),
        para.selfPort, para.peerIp->GetReadableAddress(), para.peerPort, para.qpMode);
}

HcclResult CommStar::CreateExchangerNetwork()
{
    HCCL_DEBUG("CommStar do not need to Create ExchangerNetwork");
    return HCCL_SUCCESS;
}
}  // namespace hccl

