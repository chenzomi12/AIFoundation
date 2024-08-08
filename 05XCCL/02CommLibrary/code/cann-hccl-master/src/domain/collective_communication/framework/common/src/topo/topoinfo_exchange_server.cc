/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_exchange_server.h"
#include <thread>
#include <fstream>
#include <iostream>
#include "externalinput_pub.h"
#include "config.h"
#include "hccl_socket.h"
#include "sal_pub.h"
#include "topoinfo_exchange_dispatcher.h"

namespace hccl {
const u32 DISPLAY_RANKNUM_PERLINE = 8;
using namespace std;
TopoInfoExchangeServer::TopoInfoExchangeServer(HcclIpAddress &hostIP, u32 hostPort,
    const std::vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx,
    std::shared_ptr<HcclSocket> listenSocket, const std::string &identifier)
    : hostIP_(hostIP),
      hostPort_(hostPort),
      whitelist_(whitelist),
      netDevCtx_(netDevCtx),
      listenSocket_(listenSocket),
      identifier_(identifier)
{
}

TopoInfoExchangeServer::~TopoInfoExchangeServer()
{
}

HcclResult TopoInfoExchangeServer::Setup()
{
    HcclResult ret;
    HcclResult error = HCCL_SUCCESS;

    do {
        ret = Connect(connectSockets_);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[TopoInfoExchangeServer][Setup]cluster topo exchange server connect client failed"),
            error = ret);
        HCCL_INFO("cluster topo exchange server connect with all agent success.");

        RankTable_t rankTable;
        ret = GetRanksBasicInfo(connectSockets_, rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[TopoInfoExchangeServer][Setup]GetRanksBasicInfo failed"),
            error = ret);
        HCCL_INFO("cluster topo exchange server get rank basic info from all agent success.");

        TopoInfoExchangeDispather dispatcher(this);
        ret = dispatcher.BroadcastRankTable(connectSockets_, rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[TopoInfoExchangeServer][Setup]Broadcast Rank Basic Infos failed"), error = ret);
        HCCL_INFO("cluster topo exchange server send rank basic info to all agent success.");

        ret = StopSocketListen(whitelist_, hostIP_, hostPort_);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[TopoInfoExchangeServer][Setup]topo exchange server stop socket listen failed."), error = ret);
    } while (0);

    if (error) {
        CHK_RET(Disconnect(connectSockets_));
        CHK_RET(StopNetwork(whitelist_, hostIP_, hostPort_));
    }

    HCCL_INFO("cluster topo exchange server completed, exit[%u].", error);

    return error;
}

HcclResult TopoInfoExchangeServer::Teardown()
{
    CHK_RET(Disconnect(connectSockets_));
    CHK_RET(StopNetwork(whitelist_, hostIP_, hostPort_));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetConnections(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets)
{
    connectSockets = connectSockets_;
    return HCCL_SUCCESS;
}


HcclResult TopoInfoExchangeServer::SetupByMasterInfo()
{
    isByMasterInfo_ = true;
    CHK_RET(Setup());
    return HCCL_SUCCESS;
}
HcclResult TopoInfoExchangeServer::Connect(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets)
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    u32 expectSocketNum = 1;
    u32 previousRankNum = 0;
    while (expectSocketNum > 0) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[Get][Connection]topo exchange server get socket timeout! timeout[%d s]",
                GetExternalInputHcclLinkTimeOut());
            DisplayConnectionedRank(connectSockets);
            return HCCL_E_TIMEOUT;
        }

        std::shared_ptr<HcclSocket> socket;
        std::string tag = TOPO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(hostPort_);
        HcclResult ret = listenSocket_->Accept(tag, socket);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("listenSocket_->Accept completed.");
            u32 rankNum = 0;
            CHK_RET(GetRemoteFdAndRankSize(socket, connectSockets, rankNum));
            expectSocketNum = (previousRankNum == 0) ? rankNum : expectSocketNum;
            CHK_RET(VerifyRemoteRankNum(previousRankNum, rankNum));

            expectSocketNum -= 1;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetRemoteFdAndRankSize(std::shared_ptr<HcclSocket> &socket,
    std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets, u32 &rankSize)
{
    std::string agentID;
    CHK_RET(RecvRemoteAgentID(socket, agentID));
    auto iter = connectSockets.find(agentID);
    CHK_PRT_RET(iter != connectSockets.end(),
        HCCL_ERROR("[Get][Connection]GetConnection failed. agnet[%s] has been connected.", agentID.c_str()),
        HCCL_E_INTERNAL);
    connectSockets.insert({ agentID, socket });

    CHK_RET(RecvRemoteRankNum(socket, rankSize));

    u32 rankID = 0;
    if (!isByMasterInfo_) {
        CHK_RET(SalStrToULong(agentID, HCCL_BASE_DECIMAL, rankID));
    }

    bool isRankIdUnAvailable = isByMasterInfo_ ? (false) : (rankID >= rankSize);
    CHK_PRT_RET(isRankIdUnAvailable, HCCL_ERROR("[Get][Connection]rank"
        " num[%u] from remote[%s] invalid.", rankSize, agentID.c_str()), HCCL_E_INTERNAL);
    HCCL_INFO("get remote rank[%s / %u] success.", agentID.c_str(), rankSize);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::DisplayConnectionedRank(
    const std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets)
{
    vector<string> ranksInfo;
    for (auto it : connectSockets) {
        ranksInfo.push_back(it.first);
    }
    u64 ranksLen = ranksInfo.size();
    u64 lineNum = (ranksInfo.size() % DISPLAY_RANKNUM_PERLINE == 0) ?  (ranksInfo.size()/DISPLAY_RANKNUM_PERLINE) :
                                                                    (ranksInfo.size()/DISPLAY_RANKNUM_PERLINE + 1);
    HCCL_ERROR("[TopoInfoExchangeServer][DisplayConnectionedRank]total connected num is [%llu],line num is [%llu]",
               ranksLen, lineNum);
    for (u64 i = 0; i < lineNum; i++) {
        string tmpRankList;
        for (u32 j = 0; j < DISPLAY_RANKNUM_PERLINE; j++) {
            u32 ranksInfoIndex = i * DISPLAY_RANKNUM_PERLINE + j;
            if (ranksInfoIndex < ranksInfo.size()) {
                tmpRankList += "[" + ranksInfo[ranksInfoIndex] + "]";
            } else {
                break;
            }
            tmpRankList += ((j == DISPLAY_RANKNUM_PERLINE - 1 || ranksInfoIndex == ranksInfo.size() - 1) ? ";" : ",");
        }
        HCCL_ERROR("[TopoInfoExchangeServer][DisplayConnectionedRank]connected rankinfo[LINE %llu]: %s",
            i, tmpRankList.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::Disconnect(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets)
{
    std::unique_lock<std::mutex> lock(lock_);
    for (auto &socket : connectSockets) {
        CHK_RET(DisconnectSocket(socket.second));
    }
    connectSockets.clear();
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::DeleteSocketWhiteList(u32 port,
    const std::vector<HcclIpAddress> &whitelist)
{
    std::vector<SocketWlistInfo> wlistInfosVec;
    for (auto ip : whitelist) {
        SocketWlistInfo wlistInfo = {0};
        wlistInfo.connLimit = HOST_SOCKET_CONN_LIMIT;
        wlistInfo.remoteIp.addr = ip.GetBinaryAddress().addr;
        wlistInfo.remoteIp.addr6 = ip.GetBinaryAddress().addr6;
        std::string tag = TOPO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(port);
        s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1);
        if (sRet != EOK) {
            HCCL_ERROR("[Delete][SocketWhiteList]memory copy failed. errorno[%d]", sRet);
            return HCCL_E_MEMORY;
        }
        wlistInfosVec.push_back(wlistInfo);
    }

    listenSocket_->DelWhiteList(wlistInfosVec);

    HCCL_INFO("delete socket white list success. total: %zu", whitelist.size());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::StopSocketListen(const std::vector<HcclIpAddress> &whitelist,
    HcclIpAddress &hostIP, u32 hostPort)
{
    if (listenSocket_) {
        if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
            CHK_RET(DeleteSocketWhiteList(hostPort, whitelist));
        }
        CHK_RET(listenSocket_->DeInit());
        listenSocket_ = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::StopNetwork(const std::vector<HcclIpAddress> &whitelist,
    HcclIpAddress &hostIP, u32 hostPort)
{
    std::unique_lock<std::mutex> lock(lock_);
    CHK_RET(StopSocketListen(whitelist, hostIP, hostPort));

    netDevCtx_ = nullptr;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::RecvRemoteAgentID(std::shared_ptr<HcclSocket> socket, std::string& agentID)
{
    char agentBuf[MAX_AGENT_BUF_SIZE] = {0};
    HcclResult ret = socket->Recv(agentBuf, sizeof(agentBuf));
    agentBuf[MAX_AGENT_BUF_SIZE - 1] = '\0';
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Recv][RemoteRankID]GetRemoteRankID receive rank id failed. ret[%d] ", ret), ret);
    agentID = agentBuf;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::RecvRemoteRankNum(std::shared_ptr<HcclSocket> socket, u32& remoteRankNum)
{
    HcclResult ret = socket->Recv(reinterpret_cast<char *>(&remoteRankNum), sizeof(remoteRankNum));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Recv][RemoteRankNum]GetRemoteRankID receive rank num failed. ret[%d]", ret), ret);
    CHK_PRT_RET((remoteRankNum == 0), HCCL_ERROR("[Recv][RemoteRankNum]GetRemoteRankNum receive rank num "\
        "failed. rank num is zero."), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::VerifyRemoteRankNum(u32& previousRankNum, u32 remoteRankNum) const
{
    if (previousRankNum == 0) {
        previousRankNum = remoteRankNum;
    } else {
        CHK_PRT_RET((remoteRankNum != previousRankNum),
            HCCL_ERROR("[Verify][RemoteRankNum]VerifyRemoteRankNum failed. remoteRankNum[%u] is difference "\
                "with others[%u].", remoteRankNum, previousRankNum), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetRanksBasicInfo(
    const std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets, RankTable_t &rankTable)
{
    HcclResult ret;
    u32 socketIndex = 0; // socket已经经过rankid（or serverip +deviceid排序）
    for (auto &handle : connectSockets) {
        ret = GetRankBasicInfo(handle.second, rankTable);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Get][RanksBasicInfo]GetRankBasicInfo from rank[%s] failed, ret[%d]",
            handle.first.c_str(), ret), ret);
        if (isByMasterInfo_ && rankTable.rankList.size() > 0) { // masterInfo场景下无法获取rankid
            rankTable.rankList.back().rankId = socketIndex;
        }
        HCCL_INFO("GetRankBasicInfo from rank[%s] success.", handle.first.c_str());
        socketIndex ++;
    }
    CHK_RET(SortRankList(rankTable));
    currentStep_++;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetRanksTransInfo(
    const std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets, RankTable_t &rankTable)
{
    HcclResult ret;
    u32 socketIndex = 0;
    for (auto &handle : connectSockets) {
        RankTable_t tmpRankTable;
        ret = RecvClusterInfoMsg(handle.second, tmpRankTable);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Get][RanksTransInfo]RecvClusterInfoMsg from rank[%s] failed, ret[%u]", handle.first.c_str(),
            ret),
            ret);
        CHK_PRT_RET(tmpRankTable.rankList.size() == 0,
            HCCL_ERROR("[Get][RanksTransInfo]received rank list "
            "is empty."),
            HCCL_E_INTERNAL);
        for (u32 i = 0; i < tmpRankTable.rankList.size(); i++) {
            u32 currRank = isByMasterInfo_ ? socketIndex : tmpRankTable.rankList[i].rankId;
            if ((tmpRankTable.rankList[i].transportInfo.size()) != 0) {
                if (rankTable.rankList[currRank].transportInfo.size() == 0) {
                    rankTable.rankList[currRank] = tmpRankTable.rankList[i];
                } else {
                    HCCL_ERROR("[Get][RanksTransInfo]GetRanksTransInfo: rank[%u] transportInfo has existed.", currRank);
                    return HCCL_E_INTERNAL;
                }
            }
        }
        socketIndex++;
        HCCL_INFO("RecvClusterInfoMsg from rank[%s] success.", handle.first.c_str());
    }
    currentStep_++;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::SendIndentify(std::shared_ptr<HcclSocket> socket, u32 indentify) const
{
    HcclResult ret = socket->Send(&indentify, sizeof(indentify));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Send][ClusterInfoMsg]errNo[0x%016llx] ra send indentify failed! "\
            "ret[%u]", HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), ret), ret);

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeServer::GetRankBasicInfo(std::shared_ptr<HcclSocket> socket, RankTable_t &rankTable)
{
    RankTable_t tmpRankTable;
    CHK_RET(RecvClusterInfoMsg(socket, tmpRankTable));

    CHK_PRT_RET(tmpRankTable.rankList.size() == 0, HCCL_ERROR("[Get][RankBasicInfo]received rank list is "\
        "empty."), HCCL_E_INTERNAL);
    CHK_PRT_RET(tmpRankTable.serverList.size() == 0, HCCL_ERROR("[Get][RankBasicInfo]received server list "\
        "is empty."), HCCL_E_INTERNAL);

    for (u32 i = 0; i < tmpRankTable.rankList.size(); i++) {
        rankTable.rankList.push_back(tmpRankTable.rankList[i]);
    }

    if (rankTable.serverList.size() == 0) {
        rankTable.serverList = tmpRankTable.serverList;
    } else {
        for (u32 i = 0; i < tmpRankTable.serverList.size(); i++) {
            if (!DoServerIdExist(rankTable, tmpRankTable.serverList[i].serverId)) {
                rankTable.serverList.push_back(tmpRankTable.serverList[i]);
            }
        }
    }

    CHK_RET(GetCommonTopoInfo(rankTable, tmpRankTable));

    return HCCL_SUCCESS;
}

bool TopoInfoExchangeServer::DoServerIdExist(const RankTable_t& rankTable, const std::string& serverId) const
{
    for (u32 i = 0; i < rankTable.serverList.size(); i++) {
        if (rankTable.serverList[i].serverId == serverId) {
            return true;
        }
    }
    return false;
}

HcclResult TopoInfoExchangeServer::GetCommonTopoInfo(RankTable_t &rankTable, const RankTable_t &orginRankTable) const
{
    if (rankTable.rankNum == 0) {
        rankTable.nicDeploy = orginRankTable.nicDeploy;
        HCCL_INFO("get rank basicInfo nicDeploy[%u]", rankTable.nicDeploy);
    } else {
        CHK_PRT_RET(rankTable.nicDeploy != orginRankTable.nicDeploy,
            HCCL_ERROR("[Get][CommonTopoInfo]compare nicDeploy failed. curr[%u], recv[%u]",
                rankTable.nicDeploy, orginRankTable.nicDeploy), HCCL_E_INTERNAL);
    }

    rankTable.serverNum = rankTable.serverList.size();
    rankTable.rankNum = rankTable.rankList.size();
    CHK_RET(GetDevNum(rankTable.rankList, rankTable.deviceNum));
    CHK_RET(GetSuperPodNum(rankTable.rankList, rankTable.superPodNum));
    HCCL_INFO("get rank basicInfo serverNum[%u] rankNum[%u] deviceNum[%u] superPodNum[%u], nicDeploy[%u].",
        rankTable.serverNum, rankTable.rankNum, rankTable.deviceNum, rankTable.superPodNum, rankTable.nicDeploy);
    return HCCL_SUCCESS;
}

bool RankIdCompare(const RankInfo_t& i, const RankInfo_t& j)
{
    return (i.rankId > j.rankId);
}

HcclResult TopoInfoExchangeServer::SortRankList(RankTable_t &rankTable) const
{
    std::sort(rankTable.rankList.begin(), rankTable.rankList.end(), RankIdCompare);
    return HCCL_SUCCESS;
}
}
