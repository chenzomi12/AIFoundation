/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config.h"
#include <arpa/inet.h>
#include <cctype>
#include <fcntl.h>
#include <securec.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include<map>
#include<set>
#include "topoinfo_ranktableParser_pub.h"
#include "./topo/topoinfo_ranktableStandard.h"
#include "./topo/topoinfo_ranktableConcise.h"
#include "./topo/topoinfo_ranktableHeterog.h"
#include "./topo/topoinfo_roletableParser.h"
#include "comm.h"
#include "externalinput_pub.h"

using namespace std;
using namespace hccl;

HcclResult CfgGetClusterInfo(const std::string &rankTableM, const std::string &identify, hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable, DevType deviceType)
{
    TopoInfoRanktableParser myTopoRanktable(rankTableM, identify);
    CHK_RET(myTopoRanktable.Init());
    // 获取rankTable版本
    CHK_RET(myTopoRanktable.GetRanktableVersion(rankTable.version));
    // 根据rankTable有没有版本信息属性和版本信息确定解析的方式
    std::unique_ptr<TopoInfoRanktableParser> pTopoRanktable = nullptr;
    if (rankTable.version.compare(HCCL_CLUSTER_VERSION) == 0 ||
        rankTable.version.compare(SUPERPOD_CLUSTER_VERSION) == 0) {
        pTopoRanktable.reset(new (std::nothrow) TopoinfoRanktableConcise(rankTableM, identify));
    } else if (rankTable.version.compare(HETEROG_CLUSTER_VERSION) == 0) {
        pTopoRanktable.reset(new (std::nothrow) TopoinfoRanktableHeterog(rankTableM, identify, deviceType));
    } else if (rankTable.version.compare("Standard") == 0) {
        pTopoRanktable.reset(new (std::nothrow) TopoinfoRanktableStandard(rankTableM, identify));
    } else {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "Ranktable version is not supported", "The ranktable path configured "
            "in the training can be found in the plogs." }));
        HCCL_ERROR("[Get][RanktableVersion]version[%s] is not support", rankTable.version.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    // 检查指针是否为空
    CHK_SMART_PTR_NULL(pTopoRanktable);
    // 执行初始化，加载rankTable并进行解析
    CHK_RET(pTopoRanktable->Init());
    // 将解析到的内容保存到入参hcomInfo中
    HcclResult ret = pTopoRanktable->GetClusterInfo(params, rankTable);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][ClusterInfo]identify[%s],get cluterInfo info error",
        identify.c_str()), ret);

    CHK_PRT_RET((rankTable.serverNum == 0), HCCL_ERROR("[Get][ClusterInfo]serverNum is zero."), HCCL_E_PARA);
    CHK_RET(CheckRankListInfo(rankTable.rankList));

    if (rankTable.serverNum > 1) {
        CHK_RET(CheckRankIpFamily(rankTable.rankList));
    }
    if (rankTable.version.compare(HETEROG_CLUSTER_VERSION) == 0) {
        // 异构场景无需检查
        return HCCL_SUCCESS;
    } else {
        CHK_RET(CheckRankListBaseInfo(rankTable.deviceNum, rankTable.serverNum));
        CHK_RET(CheckDeviceNumValid(rankTable.rankList, rankTable.deviceNum,
            rankTable.serverNum, rankTable.version));
    }
    return HCCL_SUCCESS;
}

HcclResult CfgGetClusterInfoWithoutDev(const std::string &rankTableM, const std::string &identify,
    hccl::HcclCommParams &params, hccl::RankTable_t &rankTable)
{
    TopoInfoRanktableParser myTopoRanktable(rankTableM, identify);
    CHK_RET(myTopoRanktable.Init());
    // 获取rankTable版本
    CHK_RET(myTopoRanktable.GetRanktableVersion(rankTable.version));
    // 根据rankTable有没有版本信息属性和版本信息确定解析的方式
    std::unique_ptr<TopoInfoRanktableParser> pTopoRanktable = nullptr;
    if (rankTable.version.compare(HCCL_CLUSTER_VERSION) == 0) {
        pTopoRanktable.reset(new (std::nothrow) TopoinfoRanktableConcise(rankTableM, identify));
    } else if (rankTable.version.compare(HETEROG_CLUSTER_VERSION) == 0) {
        pTopoRanktable.reset(new (std::nothrow) TopoinfoRanktableHeterog(rankTableM, identify));
    } else if (rankTable.version.compare("Standard") == 0) {
        pTopoRanktable.reset(new (std::nothrow) TopoinfoRanktableStandard(rankTableM, identify));
    } else {
        HCCL_ERROR("[Get][RanktableVersion]version[%s] is not support", rankTable.version.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    // 检查指针是否为空
    CHK_SMART_PTR_NULL(pTopoRanktable);
    // 执行初始化，加载rankTable并进行解析
    CHK_RET(pTopoRanktable->Init());
    // 将解析到的内容保存到入参params、rankTable中
    HcclResult ret = pTopoRanktable->GetClusterInfo(params, rankTable);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][ClusterInfo]identify[%s],get cluterInfo info error",
        identify.c_str()), ret);

    CHK_RET(CheckRankListInfo(rankTable.rankList));
    CHK_RET(CheckDeviceNumValid(rankTable.rankList, rankTable.deviceNum,
        rankTable.serverNum, rankTable.version));
    return HCCL_SUCCESS;
}

HcclResult CheckRankId(const char *rankId)
{
    CHK_PTR_NULL(rankId);
    string temp = rankId;

    for (u32 index = 0; index < temp.length(); index++) {
        if (!isdigit(temp[index])) {
            HCCL_ERROR("[Check][RankId]errNo[0x%016llx] check rankid is not digit", HCOM_ERROR_CODE(HCCL_E_PARA));
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CheckRankTableConfigInfo(const std::vector<RankInfo_t> &rankList, u32 deviceNum, u32 serverNum)
{
    if (rankList.size() != deviceNum) {
        HCCL_ERROR("[Check][RankTableConfigInfo]errNo[0x%016llx] rankList size[%llu] neq deviceNum[%u]",
            HCOM_ERROR_CODE(HCCL_E_PARA), rankList.size(), deviceNum);
        return HCCL_E_PARA;
    }
    CHK_RET(CheckGroupRankList(rankList, deviceNum, serverNum));
    return HCCL_SUCCESS;
}

HcclResult ShowRanktableConfigInfo(const bool cloudFlag, hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable)
{
    if (cloudFlag) {
        CHK_RET(DisplayCloudRankTableInfo(params, rankTable));
    } else {
        CHK_RET(DisplayRanktableInfo(params, rankTable));
    }
    return HCCL_SUCCESS;
}

HcclResult DisplayCloudRankTableInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable)
{
    HCCL_DEBUG(
        "rank_table:\n \"Unique groupNum\":\"%u\",\n \"Unique deviceNum\":\"%u\",\n \"Unique serverNum\":\"%u\"\n",
        rankTable.groupNum, rankTable.deviceNum, rankTable.serverNum);

    HCCL_DEBUG("params:\n \"uniqueID\":\"%s\"\n", params.id.internal);
    return HCCL_SUCCESS;
}

HcclResult DisplayRanktableInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable)
{
    std::string nicName = "\"para_plane_nic_name\":[";

    for (u32 i = 0; i < rankTable.nicNames.size(); i++) {
        if (i != 0) {
            nicName += ",";
        }
        std::string tmpString = rankTable.nicNames[i].c_str();
        nicName += "\"";
    }
    nicName += "],";
    HCCL_DEBUG(
        "rank_table:\n \"Unique deviceNum\":\"%u\", \n \"Unique serverNum\":\"%u\", \n \"para_plane_nic_location\""\
        ":\"%u\", \n \"para_plane_nic_num\":\"%u\", \n %s\n",
        rankTable.deviceNum, rankTable.serverNum, rankTable.nicDeploy,
        rankTable.nicNum, nicName.c_str());

    HCCL_DEBUG("params:\n \"uniqueID\":\"%s\"\n", params.id.internal);
    return HCCL_SUCCESS;
}
HcclResult DisplayRanktableInfo(const hccl::RankTable_t &rankTable)
{
    std::string nicName = "\"para_plane_nic_name\":[";

    for (u32 i = 0; i < rankTable.nicNames.size(); i++) {
        if (i != 0) {
            nicName += ",";
        }
        nicName += rankTable.nicNames[i];
    }
    nicName += "]";

    std::string deviceInfoStr = "\"device_information\":[\n";
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        deviceInfoStr += "{rankID[" + to_string(rankTable.rankList[i].rankId) + "],";
        deviceInfoStr += "serverId[" + rankTable.rankList[i].serverId + "],";
        deviceInfoStr += "deviceId[" + to_string(rankTable.rankList[i].deviceInfo.devicePhyId) + "],";
        string tmpString = rankTable.rankList[i].deviceInfo.deviceIp[0].GetReadableAddress();
        deviceInfoStr += "deviceIp[" + tmpString + "],";
        tmpString = rankTable.rankList[i].hostIp.GetReadableAddress();
        deviceInfoStr += "hostIp[" + tmpString + "]},\n";
    }
    deviceInfoStr += "]";

    HCCL_INFO(
        "rank_table:\n \"Unique deviceNum\":\"%u\", \n \"Unique serverNum\":\"%u\", \n \"para_plane_nic_location\""\
        ":\"%u\", \n \"para_plane_nic_num\":\"%u\",\n%s,\n%s\n",
        rankTable.deviceNum, rankTable.serverNum, rankTable.nicDeploy,
        rankTable.nicNum, nicName.c_str(), deviceInfoStr.c_str());
    return HCCL_SUCCESS;
}

HcclResult GetDevNum(const std::vector<RankInfo_t> &rankList, u32 &devNum)
{
    devNum = 0;
    for (auto &iter : rankList) {
        if (iter.deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            devNum++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult GetServerNum(const std::vector<RankInfo> &rankList, u32 &serverNum)
{
    serverNum = 0;
    std::set<u32> serverSet;
    for (auto &iter : rankList) {
        serverSet.insert(iter.serverIdx);
    }
    serverNum = serverSet.size();
    return HCCL_SUCCESS;
}

HcclResult GetDevNum(const std::vector<RankInfo> &rankList, u32 &devNum)
{
    devNum = 0;
    for (auto &iter : rankList) {
        if (iter.devicePhyId != HOST_DEVICE_ID) {
            devNum++;
        }
    }
    return HCCL_SUCCESS;
}

template <typename rankTable>
HcclResult GetSuperPodNums(const std::vector<rankTable> &rankList, u32 &superPodNum)
{
    superPodNum = 0;
    std::set<std::string> superPodIds;

    for (u32 index = 0; index < rankList.size(); index++) {
        // superPodId为空时, 返回超节点数量为0, 按照非超节点模式处理
        CHK_PRT_RET(rankList[index].superPodId.empty(),
            HCCL_DEBUG("ranks[%u] superPodId[%s] is empty, set superPodNum to zero", index,
            rankList[index].superPodId.c_str()),
            HCCL_SUCCESS);

        if (superPodIds.find(rankList[index].superPodId) == superPodIds.end()) {
            superPodIds.insert(rankList[index].superPodId);
        }
    }
    superPodNum = superPodIds.size();
    return HCCL_SUCCESS;
}

HcclResult GetSuperPodNum(const std::vector<RankInfo_t> &rankList, u32 &superPodNum)
{
    (void)GetSuperPodNums(rankList, superPodNum);
    return HCCL_SUCCESS;
}

HcclResult GetSuperPodNum(const std::vector<RankInfo> &rankList, u32 &superPodNum)
{
    (void)GetSuperPodNums(rankList, superPodNum);
    return HCCL_SUCCESS;
}

HcclResult CheckGroupRankList(const std::vector<RankInfo_t> &rankList, u32 deviceNum, u32 serverNum)
{
    u32 realDevNum = 0;
    CHK_RET(GetDevNum(rankList, realDevNum));
    CHK_RET(CheckAverageDev(realDevNum, serverNum));
    CHK_RET(CheckRankListInfo(rankList, realDevNum, serverNum));
    return HCCL_SUCCESS;
}

HcclResult CheckDeviceId(const std::vector<RankInfo_t> &rankList, u32 deviceNum, u32 serverNum)
// each server should has same device Id may not be continous
{
    if (serverNum == 0) {
        HCCL_ERROR("[Check][DeviceId]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    (void)deviceNum;
    std::map<std::string, std::set<s32> > serverDeviceMapList;
    for (auto it = rankList.begin(); it != rankList.end(); it++) {
        if (it->deviceInfo.devicePhyId == HOST_DEVICE_ID) {
            continue;
        }
        std::string tmpServerId = it->serverId;
        auto search = serverDeviceMapList.find(tmpServerId);
        if (search != serverDeviceMapList.end()) {
            auto rs = serverDeviceMapList[tmpServerId].insert(it->deviceInfo.devicePhyId);
            if (!rs.second) {
                RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                    std::vector<std::string>({ "device id repeat for one server. Please check ranktable",
                    "The ranktable path configured in the training can be found in the plogs." }));
                HCCL_ERROR("[Check][DeviceId]errNo[0x%016llx] check ranklist[%u], device id repeat for one server",
                           HCOM_ERROR_CODE(HCCL_E_PARA), it->rankId);
                return HCCL_E_PARA;
            }
        } else {
            std::set<s32> deviceSet;
            deviceSet.insert(it->deviceInfo.devicePhyId);
            serverDeviceMapList.insert(std::pair<std::string, std::set<s32> >(tmpServerId, deviceSet));
        }
    }
    if (serverDeviceMapList.size() == 0) {
        HCCL_ERROR("[Check][DeviceId]errNo[0x%016llx] for all ranklist, server num is zero",
            HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// 检查deviceNum、serverNum范围，rankList中rank id范围及是否升序连续分布
HcclResult CheckRankListBaseInfo(u32 deviceNum, u32 serverNum)
{
    HCCL_INFO("START CheckRankListBaseInfo");
    if (deviceNum == 0) {
        HCCL_ERROR("[Check][RankListBaseInfo]errNo[0x%016llx] device num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    if (serverNum == 0) {
        HCCL_ERROR("[Check][RankListBaseInfo]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckRankListInfo(const std::vector<RankInfo_t> &rankList)
{
    // rankList have been sorted when parseing rank table,
    // check the continuity of sorted rankList
    HCCL_INFO("START CheckRankListInfo");
    for (u32 index = 0; index < rankList.size(); index++) {
        if (rankList[index].rankId != index) {
            RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                std::vector<std::string>({ "The rank ID is used repeatedly or the rank ID exceeds the "
                "max rankId.",
                "The ranktable path configured in the training can be found in the plogs." }));
            HCCL_ERROR("[Check][RankListBaseInfo]errNo[0x%016llx] rankList[%u] rankId[%u] error",
                HCOM_ERROR_CODE(HCCL_E_PARA), index, rankList[index].rankId);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

// 校验rank ip family一致性
HcclResult CheckRankIpFamily(const std::vector<RankInfo_t> &rankList)
{
    HCCL_INFO("START CheckRankIpFamily");
    s32 hostFamily = 0;
    s32 deviceFamily = 0;
    for (u32 index = 0; index < rankList.size(); index++) {
        if (!rankList[index].hostIp.IsInvalid()) {
            CHK_PRT_RET(
                ((rankList[index].hostIp.GetFamily() != AF_INET) && (rankList[index].hostIp.GetFamily() != AF_INET6)),
                HCCL_ERROR("[Check][RankIpFamily]rank[%u] host ip family[%d] is invalid.", rankList[index].rankId,
                rankList[index].hostIp.GetFamily()),
                HCCL_E_PARA);

            CHK_PRT_RET((hostFamily != 0 && hostFamily != rankList[index].hostIp.GetFamily()),
                HCCL_ERROR("[Check][RankIpFamily]rank[%u] host ip family[%d] is not same with others[%d].",
                rankList[index].rankId, rankList[index].hostIp.GetFamily(), hostFamily),
                HCCL_E_PARA);
            hostFamily = rankList[index].hostIp.GetFamily();
        }

        // device ip不存在时, 无需校验
        if (rankList[index].deviceInfo.deviceIp.empty() ||
            ((rankList[index].deviceInfo.deviceIp.size() == 1) && rankList[index].deviceInfo.deviceIp[0].IsInvalid())) {
            continue;
        }

        for (auto &iter : rankList[index].deviceInfo.deviceIp) {
            CHK_PRT_RET(((iter.GetFamily() != AF_INET) && (iter.GetFamily() != AF_INET6)),
                HCCL_ERROR("[Check][RankIpFamily]rank[%u] device ip family[%d] is invalid.", rankList[index].rankId,
                iter.GetFamily()),
                HCCL_E_PARA);

            if (deviceFamily != 0 && deviceFamily != iter.GetFamily()) {
                const std::string ipFamilyError = "rank[" + std::to_string(rankList[index].rankId) + "] device ip \
                    family[" + std::to_string(iter.GetFamily()) + "] is not same with others[" + \
                    std::to_string(deviceFamily) + "].";
                RPT_ENV_ERR(true, "EI0001", std::vector<std::string>({"env", "tips"}),
                    std::vector<std::string>({ "RankIpFamily", ipFamilyError }));
                CHK_PRT_RET(true,
                    HCCL_ERROR("[Check][RankIpFamily]rank[%u] device ip family[%d] is not same with others[%d].",
                    rankList[index].rankId, iter.GetFamily(), deviceFamily),
                    HCCL_E_PARA);
            }
            deviceFamily = iter.GetFamily();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CheckRankListInfo(const std::vector<RankInfo_t> &rankList, u32 deviceNum, u32 serverNum)
{
    CHK_RET(CheckRankListBaseInfo(deviceNum, serverNum));
    CHK_RET(CheckRankListInfo(rankList));
    CHK_RET(CheckDeviceNumValid(rankList, deviceNum, serverNum));

    // 校验每个serverID下的deviceID是否都在同一范围
    CHK_RET(CheckDeviceId(rankList, deviceNum, serverNum));

    return HCCL_SUCCESS;
}

// 检查rank list中每个server id下的device数是否相同
HcclResult CheckDeviceNumValid(const std::vector<RankInfo_t> &rankList, u32 deviceNum,
                               u32 serverNum, std::string version)
{
    if (serverNum == 0) {
        HCCL_ERROR("[Check][DeviceNumValid]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if (version.compare(HETEROG_CLUSTER_VERSION) != 0) {
        DevType deviceType;
        CHK_RET(hrtGetDeviceType(deviceType));
        // 不对910B进行Server间卡数一致性的校验
        if (deviceType == DevType::DEV_TYPE_910B) {
            return HCCL_SUCCESS;
        }
    }

    std::map<std::string, u32> serverDeviceNumMapList;
    for (auto it = rankList.begin(); it != rankList.end(); it++) {
        if (it->deviceInfo.devicePhyId == HOST_DEVICE_ID) {
            continue;
        }
        std::string curServerId = it->serverId;
        auto search = serverDeviceNumMapList.find(curServerId);
        if (search != serverDeviceNumMapList.end()) {
            serverDeviceNumMapList[curServerId] = serverDeviceNumMapList[curServerId] + 1;
        } else {
            serverDeviceNumMapList.insert(std::pair<std::string, u32>(curServerId, 1));
        }
    }
    for (auto it = serverDeviceNumMapList.begin(); it != serverDeviceNumMapList.end(); it++) {
        if (it->second !=  (deviceNum / serverNum)) {
            RPT_INPUT_ERR(true,
                "EI0004",
                std::vector<std::string>({ "error_reason", "ranktable_path" }),
                std::vector<std::string>({
                "The devices num in ranktable is invalid", "Ensure that the devices num of each"\
                "server is consistent"
                }));
            HCCL_ERROR("[Check][DeviceNumValid]errNo[0x%016llx] devices num of each server error",
                HCOM_ERROR_CODE(HCCL_E_PARA));
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CheckPortValid(u32 port)
{
    if (port < PORT_MIN || port > PORT_MAX) {
        HCCL_ERROR("[Check][PortValid]errNo[0x%016llx] Port: [%u] not a valid port",
            HCOM_ERROR_CODE(HCCL_E_PARA), port);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckRoleAndRankConsistent(const hccl::RoleTableInfo &roleTableInfo, const hccl::HcclCommParams &params,
    const hccl::RankTable_t &rankTable)
{
    u32 roleTableSize = roleTableInfo.servers.size() + roleTableInfo.clients.size();
    if (rankTable.rankNum < roleTableSize) {
        HCCL_ERROR("[CheckRoleAndRankConsistent]errNo[0x%016llx] rank list size(%u): less than role size(%u)",
            HCOM_ERROR_CODE(HCCL_E_PARA), rankTable.rankNum, roleTableSize);
        return HCCL_E_PARA;
    }

    auto compareRoleAndRank = [&](RoleTableNodeInfo &role) -> HcclResult {
        bool isMatch = false;
        for (auto rank : rankTable.rankList) {
            if (rank.deviceInfo.devicePhyId == HOST_DEVICE_ID &&
                role.ipAddr == rank.hostIp && role.port == rank.hostPort) {
                isMatch = true;
                break;
            } else if (rank.deviceInfo.devicePhyId != HOST_DEVICE_ID && role.ipAddr == rank.deviceInfo.deviceIp[0] &&
                role.port == rank.deviceInfo.port) {
                isMatch = true;
                break;
            }
        }
        if (!isMatch) {
            HCCL_ERROR("[CheckRoleAndRankConsistent]role node notequ rank, role.ipAddr[%s] role.port[%u]",
                role.ipAddr.GetReadableIP(), role.port);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    };

    for (auto role : roleTableInfo.servers) {
        CHK_RET(compareRoleAndRank(role));
    }

    for (auto role : roleTableInfo.clients) {
        CHK_RET(compareRoleAndRank(role));
    }

    return HCCL_SUCCESS;
}

HcclResult CfgGetRoleTableInfo(const std::string &rankTableM, RoleTableInfo &roleTableInfo)
{
    TopoinfoRoletable myTopoRolektable(rankTableM);
    CHK_RET(myTopoRolektable.ParserRoleTable(roleTableInfo));

    return HCCL_SUCCESS;
}

void SetRetryEnable(DevType deviceType, const u32 &superPodNum, const u32 &serverNum,
    const u32 &deviceNumPerAggregation, bool &retryEnable)
{
    retryEnable = false;
    if (deviceType != DevType::DEV_TYPE_910_73) {
        retryEnable = false;
    } else if (superPodNum > 1) { // L2重执行
        retryEnable = GetExternalInputInterSuperPodRetryEnable();
    } else if (serverNum > 1) { // L1重执行
        retryEnable = GetExternalInputInterServerRetryEnable();
    } else if (deviceNumPerAggregation > 1) { // L0重执行
        retryEnable = GetExternalInputIntraServerRetryEnable();
    }

    HCCL_INFO("[Config][SetRetryEnable]deviceType[%d], superPodNum[%u], serverNum[%u], deviceNum[%u], retryEnable[%d].",
        deviceType, superPodNum, serverNum, deviceNumPerAggregation, retryEnable);
}