/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_exchange_agent.h"
#include <iostream>
#include <sstream>
#include "externalinput_pub.h"
#include "adapter_error_manager_pub.h"
#include "config.h"
#include "sal_pub.h"
namespace hccl {
constexpr s32 DEVICE_LOGIC_ID_LENGTH = 4;

TopoInfoExchangeAgent::TopoInfoExchangeAgent(HcclIpAddress &serverIp, u32 serverPort, std::string identifier,
    HcclNetDevCtx netDevCtx, HcclBasicRankInfo localRankInfo)
    : serverIP_(serverIp),
      serverPort_(serverPort),
      identifier_(identifier),
      localRankInfo_(localRankInfo),
      clusterTopoInfo_(),
      netDevCtx_(netDevCtx)
{}

TopoInfoExchangeAgent::~TopoInfoExchangeAgent()
{
    Teardown();
}

HcclResult TopoInfoExchangeAgent::Setup()
{
    HcclResult ret = Connect(serverIP_, serverPort_, socket_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeAgent][Setup]TopoExchangeAgent: "\
        "connect server[%s : %u] failed", serverIP_.GetReadableAddress(), serverPort_), ret);
    HCCL_INFO("TopoExchangeAgent: client connect with server ip[%s] port[%u] success.",
        serverIP_.GetReadableAddress(), serverPort_);

    CHK_RET(DetectClusterTopoInfo(socket_, clusterTopoInfo_));
    
    CHK_RET(SaveClusterInfo(clusterTopoInfo_));

    CHK_RET(VerifyClusterInfo(clusterTopoInfo_));

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::Teardown()
{
    CHK_RET(Disconnect(socket_));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::GetConnection(std::shared_ptr<HcclSocket> &socket)
{
    socket = socket_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetupByMasterInfo()
{
    isByMasterInfo_ = true;
    CHK_RET(Setup());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::DetectClusterTopoInfo(
    std::shared_ptr<HcclSocket> socket, RankTable_t &clusterTopoInfo)
{
    RankTable_t localBasicInfo;
    CHK_RET(ConstructRankTableMsg(localBasicInfo));
    CHK_RET(SendClusterInfo(socket, localBasicInfo));
    HCCL_INFO("topo exchange client send rank basic info success.");

    CHK_RET(RecvClusterInfo(socket, clusterTopoInfo));
    HCCL_INFO("topo exchange client get rank basic info success.");

    CHK_RET(SetServerIdx(clusterTopoInfo));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetServerIdx(RankTable_t &clusterInfo) const
{
    struct ServerSortInfo {
        u32 serverPosition;
        u32 selectedRankId;
    };
    std::vector<ServerSortInfo> serverSortInfoVec;
    for (u32 i = 0; i < clusterInfo.serverList.size(); i++) {
        for (u32 j = 0; j < clusterInfo.rankList.size(); j++) {
            if (clusterInfo.rankList[j].serverId == clusterInfo.serverList[i].serverId) {
                // 每个server的rankid都是连续的，只需要取每个server里任意一个rankid进行排序
                ServerSortInfo serverSortInfo;
                serverSortInfo.serverPosition = i;
                serverSortInfo.selectedRankId = clusterInfo.rankList[j].rankId;
                serverSortInfoVec.push_back(serverSortInfo);
                break;
            }
        }
    }
    sort(serverSortInfoVec.begin(), serverSortInfoVec.end(), [](const ServerSortInfo &a,
        const ServerSortInfo &b) { return a.selectedRankId < b.selectedRankId; });
    // 遍历ranklist，根据serverid获取serveridx
    for (u32 serverIdx = 0; serverIdx < serverSortInfoVec.size(); serverIdx++) {
        for (u32 j = 0; j < clusterInfo.rankList.size(); j++) {
            if (clusterInfo.rankList[j].serverId ==
                clusterInfo.serverList[serverSortInfoVec[serverIdx].serverPosition].serverId) {
                clusterInfo.rankList[j].serverIdx = serverIdx;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::GetClusterTopoInfo(RankTable_t &clusterInfo)
{
    clusterInfo.nicDeploy = clusterTopoInfo_.nicDeploy;
    clusterInfo.deviceNum = clusterTopoInfo_.deviceNum;
    clusterInfo.serverNum = clusterTopoInfo_.serverNum;
    clusterInfo.superPodNum = clusterTopoInfo_.superPodNum;
    clusterInfo.rankNum = clusterTopoInfo_.rankNum;
    clusterInfo.rankList = clusterTopoInfo_.rankList;
    clusterInfo.serverList = clusterTopoInfo_.serverList;

    return HCCL_SUCCESS;
}
HcclResult TopoInfoExchangeAgent::GetIdentifier(u32 &indentify)
{
    indentify = identifierNum_;
    return HCCL_SUCCESS;
}
HcclResult TopoInfoExchangeAgent::Connect(HcclIpAddress &serverIp, u32 port,
    std::shared_ptr<HcclSocket> &socket)
{
    std::string tag = TOPO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(port);
    EXECEPTION_CATCH((socket = std::make_shared<HcclSocket>(tag,
        netDevCtx_, serverIp, port, HcclSocketRole::SOCKET_ROLE_CLIENT)), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(socket);
    CHK_RET(socket->Init());
    CHK_RET(socket->Connect());

    return GetConnection(serverIp, port, socket);
}

HcclResult TopoInfoExchangeAgent::GetConnection(HcclIpAddress &serverIp, u32 port,
    std::shared_ptr<HcclSocket> &socket)
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            RPT_INPUT_ERR(true, "EI0006", std::vector<std::string>({"reason"}), \
                std::vector<std::string>({GET_SOCKET_TIMEOUT_REASON}));
            HCCL_ERROR("[Get][Connection]topo exchange agent get socket timeout! timeout[%lld]", timeout);
            return HCCL_E_TIMEOUT;
        }

        HcclSocketStatus status = socket->GetStatus();
        if (status == HcclSocketStatus::SOCKET_CONNECTING) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (status != HcclSocketStatus::SOCKET_OK) {
            HCCL_ERROR("[Get][Connection]server: get socket failed ret[%d]", status);
            return HCCL_E_TCP_CONNECT;
        } else {
            HCCL_INFO("TopoInfoExchangeAgent get socket success.");
            std::string agentID;
            if (isByMasterInfo_) {
                GenerateAgentID(localRankInfo_, agentID);
            } else {
                std::string rankID = std::to_string(localRankInfo_.rank);
                agentID = std::string(16 - rankID.length(), '0') + rankID;  // agent id为rank id，16位，左对齐补零
            }
            char agentBuf[MAX_AGENT_BUF_SIZE] = {0};
            s32 sRet = memcpy_s(agentBuf, sizeof(agentBuf), agentID.c_str(), agentID.size());
            CHK_PRT_RET(sRet != EOK, HCCL_ERROR("memcpy_s failed, errorno[%d]", sRet), HCCL_E_MEMORY);
            HcclResult ret = socket->Send(&agentBuf, sizeof(agentBuf));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][Connection]errNo[0x%016llx] agentID[%s] send local rank id to remote "\
                    "by client fd_handle failed, ret[%u]", HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), agentBuf, ret), ret);

            ret = socket->Send(&localRankInfo_.rankSize, sizeof(localRankInfo_.rankSize));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][Connection]errNo[0x%016llx] rank[%u] send local rank num[%u] to "\
                    "remote by client fd_handle failed, ret[%u]", HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER),
                    localRankInfo_.rank, localRankInfo_.rankSize, ret), ret);

            HCCL_INFO("local rank[%u] get socket connection with server[%s] port[%u] success.",
                localRankInfo_.rank, serverIp.GetReadableAddress(), port);
            break;
        }
    }
    return HCCL_SUCCESS;
}

std::string TopoInfoExchangeAgent::Dec2Hex(s32 i, u32 width)
{
    std::string temp;
    std::stringstream ss;
    ss << std::hex << i;
    ss >> temp;
    if (width > temp.size()) {
        return std::string((width - temp.size()), '0') + temp;
    } else {
        HCCL_WARNING("Dec2Hex: length[%u] is over width[%u]", temp.size(), width);
    }
    return temp;
}

void TopoInfoExchangeAgent::GenerateAgentID(HcclBasicRankInfo &localRankInfo, std::string &agentID)
{
    struct in_addr addr = localRankInfo.hostIP.GetBinaryAddress().addr;
    struct in6_addr addr6 = localRankInfo.hostIP.GetBinaryAddress().addr6;
    if (localRankInfo.hostIP.IsIPv6()) {
        for (size_t i = 0; i < sizeof(addr6.s6_addr); i++) {
            agentID += Dec2Hex(addr6.s6_addr[i], 2); // 转换为2位十六进制数据，左对齐补零
        }
    } else {
        for (size_t i = 0; i < sizeof(addr.s_addr) / sizeof(u8); i++) {
            agentID += Dec2Hex(*(reinterpret_cast<u8 *>(&addr.s_addr) + i), 2); // 转换为2位十六进制数据，左对齐补零
        }
    }
    agentID.append("/");
    std::string devID = std::to_string(localRankInfo.deviceLogicID);
    CHK_PRT_RET(devID.size() > DEVICE_LOGIC_ID_LENGTH, HCCL_ERROR("deviceLogicID[%s] is invalid", devID.c_str()),);
    // device id转换为4位十进制数字，左对齐补零
    agentID.append(std::string((DEVICE_LOGIC_ID_LENGTH - devID.size()), '0') + devID);
    return;
}

HcclResult TopoInfoExchangeAgent::Disconnect(std::shared_ptr<HcclSocket> &socket)
{
    CHK_RET(DisconnectSocket(socket));
    socket = nullptr;

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SendClusterInfo(std::shared_ptr<HcclSocket> socket, const RankTable_t &clusterInfo)
{
    nlohmann::json basicJson;
    CHK_RET(Struct2Json(clusterInfo, basicJson));
    basicJson[PROP_STEP] = currentStep_;  // add step to verify.
    std::string buffer = basicJson.dump();
    u32 msgLen = buffer.length();
    CHK_RET(SendClusterInfoMsg(socket, clusterInfo, buffer, msgLen));
    currentStep_++;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::RecvClusterInfo(std::shared_ptr<HcclSocket> socket, RankTable_t &clusterInfo)
{
    CHK_RET(RecvClusterInfoMsg(socket, clusterInfo));
    if (isByMasterInfo_) {
        u32 indentify = 0;
        CHK_PRT_RET(socket->Recv(reinterpret_cast<char *>(&indentify), sizeof(indentify)) != HCCL_SUCCESS,
            HCCL_ERROR("[Recv][ClusterInfoMsg]receive indentify from fdhandle failed"), HCCL_E_INTERNAL);
        identifierNum_ = indentify;
    }
    currentStep_++;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::ConstructRankTableMsg(RankTable_t &clusterInfo)
{
    RankInfo_t myRankInfo;
    myRankInfo.rankId = localRankInfo_.rank;
    myRankInfo.hostIp = localRankInfo_.hostIP;
    myRankInfo.deviceInfo.devicePhyId = localRankInfo_.devicePhysicID;
    myRankInfo.deviceInfo.deviceIp = localRankInfo_.deviceIP;
    myRankInfo.superPodId = localRankInfo_.superPodId;
    myRankInfo.superDeviceId = localRankInfo_.superDeviceId;
    myRankInfo.serverId = localRankInfo_.hostIP.GetReadableIP();

    ServerInfo_t myServerInfo;
    myServerInfo.serverId = myRankInfo.serverId;

    clusterInfo.nicDeploy = localRankInfo_.nicDeploy;
    clusterInfo.rankList.push_back(myRankInfo);
    clusterInfo.serverList.push_back(myServerInfo);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::SetTransportInfo(RankTable_t &clusterInfo)
{
    CHK_PRT_RET(clusterInfo.rankList.size() <= localRankInfo_.rank, HCCL_ERROR("[Set][TransportInfo]rank list is "\
        "invalid. size[%zu] should be greater than myRank[%u].", clusterInfo.rankList.size(), localRankInfo_.rank),
        HCCL_E_INTERNAL);
    RankInfo_t& myRankInfo = clusterInfo.rankList[localRankInfo_.rank];
    TransportInfo_t transportInfo = {0};

    for (u32 index = 0; index < clusterInfo.rankList.size(); index++) {
        transportInfo.dstRankId = clusterInfo.rankList[index].rankId;
        HcclResult ret = DetectTransportType(myRankInfo, clusterInfo.rankList[index], transportInfo.transportType);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Set][TransportInfo]rank[%u] detect transport type failed, ret[%u]. "\
                "remote[%u]", localRankInfo_.rank, ret, transportInfo.dstRankId), ret);
        myRankInfo.transportInfo.push_back(transportInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::DetectTransportType(const RankInfo_t& localRankInfo,
    const RankInfo_t& remoteRankInfo, TransportType& transportType) const
{
    if (remoteRankInfo.serverId == localRankInfo.serverId) {
            transportType = TransportType::TRANS_TYPE_P2P;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterInfo(const RankTable_t &clusterInfo)
{
    CHK_PRT_RET((clusterInfo.rankList.size() != localRankInfo_.rankSize),
        HCCL_ERROR("[Verify][ClusterInfo]rank num[%u] is different with rank list size[%zu] in total topo rank "\
        "info.", localRankInfo_.rankSize, clusterInfo.rankList.size()), HCCL_E_PARA);

    CHK_PRT_RET((clusterInfo.rankNum != localRankInfo_.rankSize), HCCL_ERROR("[Verify][ClusterInfo]rank num[%u] "\
        "is different with rank num[%u] in total topo rank info.", localRankInfo_.rankSize, clusterInfo.rankNum),
        HCCL_E_PARA);

    CHK_PRT_RET((clusterInfo.serverList.size() != clusterInfo.serverNum), HCCL_ERROR("[Verify][ClusterInfo]server "\
        "num[%u] is different with server list size[%zu] in total topo rank info.", clusterInfo.serverNum,
        clusterInfo.serverList.size()), HCCL_E_PARA);

    CHK_PRT_RET((clusterInfo.nicDeploy != localRankInfo_.nicDeploy), HCCL_ERROR("[Verify][ClusterInfo]nicDeploy "\
        "[%u] is different with nicDeploy[%u] in total topo rank info.", localRankInfo_.nicDeploy,
        clusterInfo.nicDeploy), HCCL_E_PARA);

    CHK_RET(VerifyClusterRankID(clusterInfo));
    if (localRankInfo_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(VerifyClusterDeviceIP(clusterInfo));
    }
    std::map<std::string, std::vector<RankInfo_t>> serverMap;
    for (uint32_t i = 0; i < clusterInfo.rankList.size(); i++) {
        auto iter = serverMap.find(clusterInfo.rankList[i].serverId);
        if (iter == serverMap.end()) {
            std::vector<RankInfo_t> vec;
            vec.push_back(clusterInfo.rankList[i]);
            serverMap.insert({clusterInfo.rankList[i].serverId, vec});
        } else {
            serverMap[clusterInfo.rankList[i].serverId].push_back(clusterInfo.rankList[i]);
        }
    }

    CHK_PRT_RET((clusterInfo.serverNum != serverMap.size()), HCCL_ERROR("[Verify][ClusterInfo]server num[%u] is "\
        "different with server num[%u] in total topo rank info.", clusterInfo.serverNum, serverMap.size()),
        HCCL_E_PARA);

    uint32_t deviceNumInServer = 0;
    for (auto &server : serverMap) {
        CHK_PRT_RET((server.second.size() == 0), HCCL_ERROR("[Verify][ClusterInfo]server ip[%s] has %u device.",
            server.first.c_str(), server.second.size()), HCCL_E_PARA);
        if (deviceNumInServer != 0) {
            HCCL_WARNING("[Verify][ClusterInfo]server ip[%s] has %u devices, other server has %u.",
                server.first.c_str(), server.second.size(), deviceNumInServer);
        }
        deviceNumInServer = server.second.size();
        HcclResult ret = VerifyServerDevicePhysicID(server.second);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Verify][ClusterInfo]server id[%s] verify device physic id failed.",
            server.first.c_str()), HCCL_E_PARA);
    }

    if (clusterInfo.serverNum > 1) {
        CHK_RET(CheckRankIpFamily(clusterInfo.rankList));
    }

    // 超节点校验
    CHK_RET(VerifyClusterSuperPodInfo(clusterInfo.rankList));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterDeviceIP(const RankTable_t &clusterInfo)
{
    if (clusterInfo.rankList.size() == 1) {
        return HCCL_SUCCESS;
    }
    if (clusterInfo.serverList.size() == 1) {
        // 单机场景对 device ip不做要求
        return HCCL_SUCCESS;
    }
    if (clusterInfo.superPodNum == 1 && GetExternalInputInterHccsDisable() == false) {
        // 单超节点，并且节点间走HCCS场景，device ip不做要求
        return HCCL_SUCCESS;
    }
    for (u32 i = 0; i < (clusterInfo.rankList.size() - 1); i++) {
        for (u32 j = (i + 1); j < clusterInfo.rankList.size(); j++) {
            bool err = HasRepeatedIP(clusterInfo.rankList[i].deviceInfo.deviceIp,
                clusterInfo.rankList[j].deviceInfo.deviceIp);
            CHK_PRT_RET(err, HCCL_ERROR("[Verify][ClusterDeviceIP]rank[%u]'s device ip is repeated with rank[%u].",
                clusterInfo.rankList[i].rankId, clusterInfo.rankList[j].rankId), HCCL_E_PARA);
        }
    }
    return HCCL_SUCCESS;
}

bool TopoInfoExchangeAgent::HasRepeatedIP(const std::vector<HcclIpAddress> &deviceAIP,
    const std::vector<HcclIpAddress> &deviceBIP) const
{
    for (u32 i = 0; i < deviceAIP.size(); i++) {
        for (u32 j = 0; j < deviceBIP.size(); j++) {
            if (deviceAIP[i] == deviceBIP[j]) {
                HCCL_WARNING("device ip[%s] is repeated.", deviceAIP[i].GetReadableAddress());
                return true;
            }
        }
    }
    return false;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterRankID(const RankTable_t &clusterInfo) const
{
    if (clusterInfo.rankList.size() == 1) {
        return HCCL_SUCCESS;
    }
    for (u32 i = 0; i < (clusterInfo.rankList.size() - 1); i++) {
        for (u32 j = (i + 1); j < clusterInfo.rankList.size(); j++) {
            bool err = (clusterInfo.rankList[i].rankId == clusterInfo.rankList[j].rankId);
            CHK_PRT_RET(err, HCCL_ERROR("[Verify][ClusterRankID]rank id[%u] is repeated.",
                clusterInfo.rankList[i].rankId), HCCL_E_PARA);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyServerDevicePhysicID(const std::vector<RankInfo_t> &serverInfo) const
{
    if (serverInfo.size() == 1) {
        return HCCL_SUCCESS;
    }
    for (u32 i = 0; i < (serverInfo.size() - 1); i++) {
        for (u32 j = (i + 1); j < serverInfo.size(); j++) {
            bool err = (serverInfo[i].deviceInfo.devicePhyId == serverInfo[j].deviceInfo.devicePhyId);
            CHK_PRT_RET(err, HCCL_ERROR("[Verify][ServerDevicePhysicID]rank[%u] and rank[%u] has the same device "\
                "physic id[%d].", serverInfo[i].rankId, serverInfo[j].rankId, serverInfo[i].deviceInfo.devicePhyId),
                HCCL_E_PARA);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeAgent::VerifyClusterSuperPodInfo(const std::vector<RankInfo_t> &rankInfo) const
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    CHK_PRT_RET(deviceType != DevType::DEV_TYPE_910_73,
        HCCL_DEBUG("[Verify][SuperPodInfo]deviceType[%d] does not need verify superPod info", deviceType),
        HCCL_SUCCESS);

    // 获取每个超节点内的serverId
    std::map<std::string, std::set<std::string>> superPodSrvIdMap; // super_pod_id -> serverId
    std::map<std::string, std::set<u32>> superPodSdidMap; // super_pod_id -> superDeviceId
    for (u32 i = 0; i < rankInfo.size(); i++) {
        auto iter = superPodSrvIdMap.find(rankInfo[i].superPodId);
        if (iter == superPodSrvIdMap.end()) {
            std::set<std::string> serverIdSet;
            serverIdSet.insert(rankInfo[i].serverId);
            superPodSrvIdMap.insert({rankInfo[i].superPodId, serverIdSet});
        } else if (iter->second.find(rankInfo[i].serverId) == iter->second.end()) {
            iter->second.insert(rankInfo[i].serverId);
        }

        auto it = superPodSdidMap.find(rankInfo[i].superPodId);
        if (it == superPodSdidMap.end()) {
            std::set<u32> superDeviceIdSet;
            superDeviceIdSet.insert(rankInfo[i].superDeviceId);
            superPodSdidMap.insert({rankInfo[i].superPodId, superDeviceIdSet});
        } else if (it->second.find(rankInfo[i].superDeviceId) == it->second.end()) {
            it->second.insert(rankInfo[i].superDeviceId);
        } else {
            // 超节点内superDeviceId在超节点内唯一
            CHK_PRT_RET(it->second.find(rankInfo[i].superDeviceId) != it->second.end(),
                HCCL_ERROR("[Verify][SuperPodInfo]superDeviceId[0x%x] in superPod[%s]"
                "is already exist.",
                rankInfo[i].superDeviceId, it->first.c_str()),
                HCCL_E_PARA);
        }
    }

    // 校验每个超节点内的server数量一致
    u32 serverNumPerPod = 0;
    for (auto iter = superPodSrvIdMap.begin(); iter != superPodSrvIdMap.end(); ++iter) {
        if (iter == superPodSrvIdMap.begin()) {
            serverNumPerPod = superPodSrvIdMap.begin()->second.size();
        }
        u32 serverNumCurPod = iter->second.size();
        CHK_PRT_RET(serverNumPerPod != serverNumCurPod,
            HCCL_ERROR("[Verify][SuperPodInfo]serverNum[%u] in superPod[%s] and serverNum[%u] in superPod[%s] "\
            "are different.", serverNumPerPod, superPodSrvIdMap.begin()->first.c_str(),
            serverNumCurPod, iter->first.c_str()), HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}
}
