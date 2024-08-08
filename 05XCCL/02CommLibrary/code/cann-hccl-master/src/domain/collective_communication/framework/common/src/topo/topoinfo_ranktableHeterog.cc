/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableHeterog.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "externalinput_pub.h"
#include "hccl_comm_pub.h"
#include "config.h"
#include "workflow_pub.h"

using namespace std;
using namespace hccl;

constexpr u32 MAX_PORT_ID = 65535;
TopoinfoRanktableHeterog::TopoinfoRanktableHeterog(const std::string &rankTableM,
    const std::string &identify, DevType deviceType)
    : TopoInfoRanktableParser(rankTableM, identify), deviceType_(deviceType)
{
}

TopoinfoRanktableHeterog::~TopoinfoRanktableHeterog()
{
}

HcclResult TopoinfoRanktableHeterog::Init()
{
    // 根据rankTable类型标记执行读文件或读字符串,将内容保存在json对象fileContent_中
    CHK_RET(LoadRankTableString(rankTableFile_));
    // 解析rankTable
    CHK_RET(ParserClusterInfo(params_, rankTable_));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetSelfClusterInfo(HcclCommParams &params)
{
    // 获取params
    params.rank = params_.rank;
    params.userRank = params_.rank;
    params.totalRanks = params_.totalRanks;
    params.logicDevId = params_.logicDevId;
    params.serverId = params_.serverId;
    params.deviceType = params_.deviceType;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetClusterInfo(hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable)
{
    CHK_RET(GetClusterInfo(rankTable));
    CHK_RET(GetSelfClusterInfo(params));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetClusterInfo(RankTable_t &clusterInfo)
{
    // 获取rankInfo
    clusterInfo.deviceNum = rankTable_.deviceNum;
    clusterInfo.serverNum = rankTable_.serverNum;
    clusterInfo.nicDeploy = rankTable_.nicDeploy;
    clusterInfo.rankNum = rankTable_.rankNum;
    clusterInfo.rankList = rankTable_.rankList;
    clusterInfo.serverList = rankTable_.serverList;
    clusterInfo.collectiveId = rankTable_.collectiveId;
    clusterInfo.version = rankTable_.version;
    clusterInfo.mode = rankTable_.mode;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::ParserClusterInfo(hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable)
{
    // 获取ranktable info信息
    CHK_RET(GetRanktableInfo(rankTable));
    if (!IsTaskNumCalMode()) {
        CHK_RET(CheckNicDeployConsistence(rankTable));
        rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
        std::sort(rankTable.rankList.begin(), rankTable.rankList.end(),
            [&](const RankInfo_t &a, const RankInfo_t &b) -> bool {return a.rankId < b.rankId;});

        unordered_map<std::string, u32> minRankPerServerMap;
        for (auto &iter : rankTable.rankList) {
            if (minRankPerServerMap.find(iter.serverId) == minRankPerServerMap.end()) {
                minRankPerServerMap[iter.serverId] = iter.rankId;
            } else if (minRankPerServerMap[iter.serverId] > iter.rankId) {
                minRankPerServerMap[iter.serverId] = iter.rankId;
            }
        }
        map<u32, std::string> minRankInServerMap;
        for (auto &iter : minRankPerServerMap) {
            minRankInServerMap.insert(std::make_pair(iter.second, iter.first));
        }

        u32 serverIndex = 0;
        for (auto &iter : minRankInServerMap) {
            for (u32 rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
                if (rankTable.rankList[rankIndex].serverId == iter.second) {
                    rankTable.rankList[rankIndex].serverIdx = serverIndex;
                    ServerInfo_t serverinfo;
                    NetworkInfo_t networkInfo;
                    serverinfo.serverId = iter.second;
                    networkInfo.ipAddr = rankTable.rankList[rankIndex].hostIp;
                    serverinfo.networkInfo.push_back(networkInfo);
                    std::vector<ServerInfo_t>::iterator found = find(rankTable.serverList.begin(),
                        rankTable.serverList.end(), serverinfo);
                    if (found == rankTable.serverList.end()) {
                        rankTable.serverList.push_back(serverinfo);
                    }
                }
            }
            serverIndex++;
        }

        u32 rankId = INVALID_VALUE_RANKID;
        if (SalStrToULong(identify_, HCCL_BASE_DECIMAL, rankId) != HCCL_SUCCESS) {
            RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({"error_reason", "ranktable_path"}), \
                std::vector<std::string>({"The identify must be digit.", "The ranktable path configured in the "\
                "training can be found in the plogs."}));
            HCCL_ERROR("[Parser][ClusterInfo]errNo[0x%016llx] identify[%s] is invalid",
                HCOM_ERROR_CODE(HCCL_E_PARA), identify_.c_str());
            return HCCL_E_PARA;
        }

        // 校验rank id合法性
        if (rankId >= rankTable.rankList.size() || rankId < 0) {
            HCCL_ERROR("[Parse][ClusterInfo]rankId[%u] is invalid", rankId);
            return HCCL_E_PARA;
        }
        CHK_PRT_RET(rankId != rankTable.rankList[rankId].rankId,
            HCCL_ERROR("[Parse][ClusterInfo]check rankList[%u] rankId[%u] failed", rankId,
                rankTable.rankList[rankId].rankId), HCCL_E_UNAVAIL);

        // params内容填入
        params.rank = rankId;
        params.logicDevId = rankTable.rankList[rankId].deviceInfo.devicePhyId;
        params.serverId = rankTable.rankList[rankId].serverId;
        params.totalRanks = rankTable.rankNum;
        params.deviceType = deviceType_;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetRanktableInfo(RankTable_t &clusterInfo)
{
    // 清空list
    clusterInfo.serverList.clear();
    clusterInfo.rankList.clear();

    // 获取信息
    std::string collective_id;
    nlohmann::json node_list;
    std::string version;
    std::string mode;

    CHK_RET(GetJsonProperty(fileContent_, "collective_id", collective_id));
    if (collective_id.length() > COLLECTIVEID_MAX_LEN) {
        HCCL_ERROR("[Get][RanktableInfo]errNo[0x%016llx] collectiveId length is over than %d bytes.",
            HCOM_ERROR_CODE(HCCL_E_PARA), collective_id.length());
        return HCCL_E_PARA;
    }
    CHK_RET(GetJsonProperty(fileContent_, "node_list", node_list));
    CHK_RET(GetJsonProperty(fileContent_, "version", version));

    if (fileContent_.find("mode") != fileContent_.end()) {
        CHK_RET(GetJsonProperty(fileContent_, "mode", mode));
        CHK_RET(CheckMode(mode));
    }

    HCCL_DEBUG("[rankTableJson] -> collectiveId: [%s], nodeListSize: [%zu]", collective_id.c_str(), node_list.size());

    // 保存serverNum
    clusterInfo.serverNum = node_list.size();
    if (clusterInfo.serverNum == 0) {
        HCCL_ERROR("[Get][RanktableInfo]errNo[0x%016llx] node num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    // 获得single node信息
    for (u32 index = 0; index < clusterInfo.serverNum; index++) {
        CHK_RET(GetSingleNode(node_list, index, clusterInfo));
    }

    // 保存deviceNum和rankNum
    CHK_RET(GetDevNum(clusterInfo.rankList, clusterInfo.deviceNum));
    clusterInfo.rankNum = clusterInfo.rankList.size();
    clusterInfo.collectiveId = collective_id;
    clusterInfo.version = version;
    clusterInfo.mode = mode;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::CheckNicDeployConsistence(RankTable_t &clusterInfo) const
{
    // 检查rankList的大小
    CHK_PRT_RET(clusterInfo.rankList.size() == 0, HCCL_DEBUG("rank list size is 0, skip nic deply check."),
        HCCL_SUCCESS);
    NICDeployment tmpNicDeply = clusterInfo.rankList.begin()->hostIp.IsInvalid() ?
        NICDeployment::NIC_DEPLOYMENT_DEVICE : NICDeployment::NIC_DEPLOYMENT_HOST;
    for (auto &it:clusterInfo.rankList) {
        CHK_PRT_RET((tmpNicDeply == NICDeployment::NIC_DEPLOYMENT_DEVICE && !it.hostIp.IsInvalid()) ||
            (tmpNicDeply == NICDeployment::NIC_DEPLOYMENT_HOST &&
            it.hostIp.IsInvalid()), HCCL_ERROR("[Get][RanktableInfo] errNo[0x%016llx] " \
                "hostIp config bettewn ranks is different.", HCOM_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::CheckMode(std::string &mode) const
{
    if (mode == "tcp" || mode == "rdma") {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[Get][RanktableInfo]errNo[0x%016llx] mode[%s] is not supported, support for [tcp] and [rdma]",
        HCOM_ERROR_CODE(HCCL_E_PARA), mode.c_str());
    return HCCL_E_PARA;
}

HcclResult TopoinfoRanktableHeterog::GetHostPort(const u32 &localRank, u32 &hostPort)
{
    u32 basePort = (GetExternalInputHcclIfBasePort() == HCCL_INVALIED_IF_BASE_PORT) ?
        HOST_PARA_BASE_PORT : GetExternalInputHcclIfBasePort();
    hostPort = basePort + localRank;
    CHK_PRT_RET(hostPort > MAX_PORT_ID, HCCL_ERROR("[Get][HostPort]invalid port id[%u]", hostPort), HCCL_E_INTERNAL);
    if (hostPortMap_.find(hostPort) == hostPortMap_.end()) {
        return HCCL_SUCCESS;
    }
    while (hostPortMap_[hostPort] == HOST_PORT_USED) {
        hostPort++;
        CHK_PRT_RET(hostPort > MAX_PORT_ID, HCCL_ERROR("[Get][HostPort]invalid port id[%u]", hostPort),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetRanks(const nlohmann::json &NodeListObj, u32 objIndex,
    RankTable_t &clusterInfo, std::string &serverId, u32 &serverIdx, HcclIpAddress &nodeIp)
{
    HCCL_DEBUG("Get Node[%u]: nodeAddr:[%s], nodeIdx:[%u]", objIndex, serverId.c_str(), serverIdx);
    // 获取信息
    nlohmann::json Ranks;
    CHK_RET(GetJsonArrayMemberProperty(NodeListObj, objIndex, "ranks", Ranks));

    HCCL_DEBUG("[%s.json] -> rank_list: size:%zu", fileName_.c_str(), Ranks.size());
    CHK_PRT_RET(Ranks.size() == 0, HCCL_ERROR("[Get][Ranks]Ranks size is zero"), HCCL_E_PARA);

    // 获取单rank信息
    for (u32 index = 0; index < Ranks.size(); index++) {
        CHK_RET(GetSingleRank(Ranks, index, clusterInfo, serverId, serverIdx, nodeIp));
    }
    // 重新分配host port
    for (auto &rankInfo : clusterInfo.rankList) {
        if (rankInfo.hostPort == HCCL_INVALIED_IF_BASE_PORT) {
            CHK_RET(GetHostPort(rankInfo.localRank, rankInfo.hostPort));
        }
        HCCL_DEBUG("rank id: %u localRank:%u host port: %u", rankInfo.rankId, rankInfo.localRank, rankInfo.hostPort);
    }
    hostPortMap_.clear();
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetSingleNode(const nlohmann::json &NodeListObj, u32 objIndex,
    RankTable_t &clusterInfo)
{
    // 获取信息
    HcclResult ret;
    std::string nodeAddr;
    std::string serverId;
    CHK_RET(GetJsonArrayMemberProperty(NodeListObj, objIndex, "node_addr", nodeAddr));
    // 将serverId添加到资源池,内部会进行IP地址校验，如果资源池中有serverId，则报错
    serverId = nodeAddr;
    CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, serverId,
        JsonCheckOpType::CHECK_OP_TYPE_INSERT));
    // 设置serverIdx
    u32 serverIdx;
    GenerateServerIdx(serverId, serverIdx);

    // 计算node ip
    HcclIpAddress nodeIp;
    CHK_RET(ConvertIpAddress(nodeAddr, nodeIp));

    // 处理ranklist
    ret = GetRanks(NodeListObj, objIndex, clusterInfo, serverId, serverIdx, nodeIp);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][GetRanks]get rank list error:nodeAddr:[%s]", serverId.c_str()), ret);
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetSingleRank(const nlohmann::json &ranksObj, u32 objIndex,
    RankTable_t &clusterInfo, std::string &serverId, u32 &serverIdx, HcclIpAddress &nodeIp)
{
    // 获取rank_id
    std::string rankId;
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "rank_id", rankId));

    s32 devicePhyId = 0;
    std::string devPhyIdStr;
    if (GetJsonArrayMemberProperty(ranksObj, objIndex, "device_id", devPhyIdStr) == HCCL_E_NOT_FOUND) {
        devicePhyId = HOST_DEVICE_ID;
    } else {
        HCCL_DEBUG("[Get][GetRanks]device_id:[%s]", devPhyIdStr.c_str());
        CHK_RET(SalStrToInt(devPhyIdStr, HCCL_BASE_DECIMAL, devicePhyId));
    }

    s32 port = 0;
    std::string portStr;
    if (GetJsonArrayMemberProperty(ranksObj, objIndex, "port", portStr) == HCCL_E_NOT_FOUND) {
        port = 0;
    } else {
        CHK_RET(SalStrToInt(portStr, HCCL_BASE_DECIMAL, port));
    }

    // 如果device没有网卡，则缺省，通信使用node_addr
    HcclIpAddress rankIp;
    std::string rankIpStr;
    if (GetJsonArrayMemberProperty(ranksObj, objIndex, "rank_ip", rankIpStr) == HCCL_E_NOT_FOUND) {
        rankIp = nodeIp;
    } else {
        if (rankIpStr.empty()) {
            rankIp = nodeIp;
        } else {
            CHK_RET(ConvertIpAddress(rankIpStr, rankIp));
        }
    }

    // rankList
    RankInfo_t rankInfo;
    rankInfo.localRank = objIndex;
    rankInfo.serverId = serverId;
    rankInfo.serverIdx = serverIdx;
    rankInfo.hostIp = nodeIp;
    rankInfo.deviceInfo.devicePhyId = devicePhyId;
    rankInfo.deviceInfo.port = port;
    rankInfo.deviceInfo.deviceIp.push_back(rankIp);
    if (SalStrToULong(rankId, HCCL_BASE_DECIMAL, rankInfo.rankId) != HCCL_SUCCESS) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "The rankid in ranktable is invalid. Please check ranktable",
            "The ranktable path configured in the training can be found in the plogs." }));
        HCCL_ERROR("[Get][SingleRank]errNo[0x%016llx] rankid[%s] is invalid",
            HCOM_ERROR_CODE(HCCL_E_PARA), rankId.c_str());
        return HCCL_E_PARA;
    }
    rankInfo.podName = "";  // podname在新场景下置空
    // ranktable中port无效，设置为环境变量HCCL_IF_BASE_PORT+该rank的local_rank_id,否则按照配置的port使用
    if (port == 0) {
        rankInfo.hostPort = HCCL_INVALIED_IF_BASE_PORT;
    } else {
        rankInfo.hostPort = port;
        hostPortMap_[rankInfo.hostPort] = HOST_PORT_USED;
    }
    clusterInfo.rankList.push_back(rankInfo);
    HCCL_DEBUG("[%s.json]->rankId[%u], nodeAddr[%s]", fileName_.c_str(),
        rankInfo.rankId, rankInfo.serverId.c_str());

    return HCCL_SUCCESS;
}
