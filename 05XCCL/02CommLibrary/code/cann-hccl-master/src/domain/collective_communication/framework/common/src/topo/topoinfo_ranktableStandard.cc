/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableStandard.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "config.h"
#include "hccl_comm_pub.h"
#include "workflow_pub.h"

using namespace std;
using namespace hccl;


TopoinfoRanktableStandard::TopoinfoRanktableStandard(const std::string &rankTableM, const std::string &identify)
    : TopoInfoRanktableParser(rankTableM, identify)
{
}

TopoinfoRanktableStandard::~TopoinfoRanktableStandard()
{
}

HcclResult TopoinfoRanktableStandard::Init()
{
    CHK_RET(LoadRankTableString(rankTableFile_));
    CHK_RET(ParserClusterInfo(params_, rankTable_));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetSelfClusterInfo(HcclCommParams &params)
{
    // 获取芯片类型信息
    CHK_RET(hrtGetDeviceType(params.deviceType));
    params.rank = params_.rank;
    params.userRank = params_.rank;
    params.logicDevId = params_.logicDevId;
    params.totalRanks = params_.totalRanks;
    params.serverId = params_.serverId;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetClusterInfo(hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable)
{
    CHK_RET(GetClusterInfo(rankTable));
    CHK_RET(GetSelfClusterInfo(params));
    return HCCL_SUCCESS;
}
HcclResult TopoinfoRanktableStandard::GetClusterInfo(RankTable_t &clusterInfo)
{
    clusterInfo.nicDeploy = rankTable_.nicDeploy;
    clusterInfo.deviceNum = rankTable_.deviceNum;
    clusterInfo.serverNum = rankTable_.serverNum;
    clusterInfo.groupNum = rankTable_.groupNum;
    clusterInfo.nicNum = rankTable_.nicNum;
    clusterInfo.nicNames = rankTable_.nicNames;
    clusterInfo.rankNum = rankTable_.rankNum;
    clusterInfo.rankList = rankTable_.rankList;
    clusterInfo.serverList = rankTable_.serverList;

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::ParserClusterInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable)
{
    CHK_RET(GetDeployMode(cloudFlag_));
    HCCL_INFO("deploy mode is %s", cloudFlag_ ? "cloud" : "Laborratory");
    if (!IsTaskNumCalMode()) {
        CHK_RET(hrtGetDeviceType(params.deviceType));
    }

    u32 rankId = INVALID_VALUE_RANKID;
    if (cloudFlag_) {
        CHK_RET(GetCloudHcomInfo(params, rankTable, identify_, rankId));
    } else {
        if (!IsTaskNumCalMode()) {
            CHK_RET(CheckRankId(identify_.c_str()));
            if (SalStrToULong(identify_, HCCL_BASE_DECIMAL, rankId) != HCCL_SUCCESS) {
                RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                    std::vector<std::string>({ "The identify must be digit.",
                    "The ranktable path configured in the training can be found in the plogs." }));
                HCCL_ERROR("[Parser][ClusterInfo]errNo[0x%016llx] identify[%s] is invalid",
                    HCOM_ERROR_CODE(HCCL_E_PARA), identify_.c_str());
                return HCCL_E_PARA;
            }
        }
        CHK_RET(GetHcomInfo(params, rankTable));
    }

    if (!IsTaskNumCalMode()) {
        std::sort(rankTable.rankList.begin(), rankTable.rankList.end(),
            [&](const RankInfo_t &a, const RankInfo_t &b) -> bool {return a.rankId < b.rankId;});

        // 校验rank id合法性
        if (rankId >= rankTable.rankList.size()) {
            RPT_INPUT_ERR(true,
                "EI0004",
                std::vector<std::string>({"error_reason", "ranktable_path"}),
                std::vector<std::string>({
                    "The rankid is invalid.",
                    "The ranktable path configured in the training can be found in the plogs."
                })
            );
            HCCL_ERROR("[Parse][ClusterInfo]rankid[%u] is invalid", rankId);
            return HCCL_E_PARA;
        }
        CHK_PRT_RET(rankId != rankTable.rankList[rankId].rankId,
            HCCL_ERROR("[Parse][ClusterInfo]check rankList[%u] rankId[%u] failed",
                rankId, rankTable.rankList[rankId].rankId), HCCL_E_UNAVAIL);
        u32 devId = rankTable.rankList[rankId].deviceInfo.devicePhyId;
        CHK_RET(hrtGetDevice(&params.logicDevId));

        u32 devicePhyId = 0;
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(params.logicDevId), devicePhyId));

        CHK_PRT_RET(devicePhyId != static_cast<u32>(devId),
            HCCL_ERROR("[Parse][ClusterInfo]ranktable config devId[%d],but local devId[%u]", devId,
            devicePhyId), HCCL_E_UNAVAIL);

        params.rank = rankId;
        params.totalRanks = rankTable.rankNum;
        params.serverId = rankTable.rankList[rankId].serverId;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetHcomInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable)
{
    // para_plane_location
    std::string paraPlaneLocation;
    CHK_RET(GetJsonProperty(fileContent_, "para_plane_nic_location", paraPlaneLocation));

    HCCL_DEBUG("%s.json -> para_plane_location: %s", fileName_.c_str(), paraPlaneLocation.c_str());

    if (paraPlaneLocation == "host") { // 不支持host 网卡
        HCCL_ERROR("[Get][HcomInfo]errNo[0x%016llx] host nic is unsupport", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if ((params.deviceType == DevType::DEV_TYPE_910 || params.deviceType == DevType::DEV_TYPE_910B ||
         params.deviceType == DevType::DEV_TYPE_910_73) && paraPlaneLocation != "device") {
        HCCL_ERROR("[Get][HcomInfo]errNo[0x%016llx] paraPlaneLocation should be 'device'",
            HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    // 当前只支持device侧的网卡
    rankTable.nicDeploy = ((paraPlaneLocation == "device") ? NICDeployment::NIC_DEPLOYMENT_DEVICE :
        NICDeployment::NIC_DEPLOYMENT_RESERVED);
    // group_count
    std::string groupCount;
    CHK_RET(GetJsonProperty(fileContent_, "group_count", groupCount));

    HCCL_DEBUG("%s.json -> group_count: %s", fileName_.c_str(), groupCount.c_str());
    CHK_RET(SalStrToULong(groupCount, HCCL_BASE_DECIMAL, rankTable.groupNum));
    // 校验groupCount ，groupCount不能为0
    if (rankTable.groupNum == 0) {
        HCCL_ERROR("[Get][HcomInfo]errNo[0x%016llx] groupNum is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    // group_list
    CHK_RET(GetGroupList(params, rankTable));

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetServerList(const nlohmann::json &obj, u32 objIndex,
    hccl::RankTable_t &rankTable, u32 serverNum)
{
    if (serverNum == 0) {
        HCCL_ERROR("[Get][ServerList]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("get serverList[%u]", objIndex);
    rankTable.serverList.clear();
    nlohmann::json serverList;
    CHK_RET(GetJsonArrayMemberProperty(obj, objIndex, "server_list", serverList));

    HCCL_DEBUG("%s.json -> server_list[%u]: size:%zu", fileName_.c_str(), objIndex, serverList.size());
    if (serverList.size() == 0) {
        HCCL_ERROR("[Get][ServerList]errNo[0x%016llx] serverList[%u] size is zero",
            HCOM_ERROR_CODE(HCCL_E_PARA), objIndex);
        return HCCL_E_PARA;
    }
    if (serverList.size() != serverNum) {
        HCCL_ERROR("[Get][ServerList]errNo[0x%016llx] serverList[%u] size[%zu] neq server num[%u]",
            HCOM_ERROR_CODE(HCCL_E_PARA), objIndex, serverList.size(), serverNum);
        return HCCL_E_PARA;
    }

    for (u32 index = 0; index < serverList.size(); index++) {
        CHK_RET(GetSingleServer(serverList, index, rankTable));
    }

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetSingleServer(const nlohmann::json &serverListObj, u32 objIndex,
    hccl::RankTable_t &rankTable)
{
    ServerInfo_t serverInfo;
    std::string serverId;
    CHK_RET(GetJsonArrayMemberProperty(serverListObj, objIndex, "server_id", serverId));
    
    CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, serverId,
        JsonCheckOpType::CHECK_OP_TYPE_INSERT));
    HCCL_DEBUG("server id[%u]:[%s]", objIndex, serverId.c_str());
    serverInfo.serverId = serverId;
    // 解析内层-参数平面的网卡信息
    nlohmann::json paraPlaneInfo;
    CHK_RET(GetJsonArrayMemberProperty(serverListObj, objIndex, "para_plane_info", paraPlaneInfo));
    // server list中的网卡个数应该与ranktable中的网卡个数一致
    if (paraPlaneInfo.size() != rankTable.nicNum) {
        HCCL_ERROR("[Get][SingleServer]errNo[0x%016llx] paraPlaneInfo[%u] size[%zu] neq nicNum[%u]",
            HCOM_ERROR_CODE(HCCL_E_PARA), objIndex, paraPlaneInfo.size(), rankTable.nicNum);
        return HCCL_E_PARA;
    }
    serverInfo.networkInfo.clear();

    for (u32 innerIndex = 0; innerIndex < rankTable.nicNames.size(); innerIndex++) {
        NetworkInfo_t networdInfo;
        networdInfo.ethName = rankTable.nicNames[innerIndex];

        // 依照nicNames来搜索
        for (u32 i = 0; i < paraPlaneInfo.size(); i++) {
            auto findEth = paraPlaneInfo.at(i).find(networdInfo.ethName);
            if (findEth != paraPlaneInfo.at(i).end()) {  // 找到ethName
                std::string ethIp = findEth->get<std::string>();
                CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_ETH_IP, ethIp,
                    JsonCheckOpType::CHECK_OP_TYPE_INSERT));
                CHK_RET(ConvertIpAddress(ethIp, networdInfo.ipAddr));
                break;
            }
        }
        if (networdInfo.ipAddr.IsInvalid()) {
            HCCL_ERROR("[Get][SingleServer]errNo[0x%016llx] networdInfo [%s] ipAddr is invalid",
                HCOM_ERROR_CODE(HCCL_E_PARA), networdInfo.ethName.c_str());
            return HCCL_E_PARA;
        }
        HCCL_DEBUG("networdInfo[%u] [%s] ipAddr[%s]", objIndex, networdInfo.ethName.c_str(), \
            networdInfo.ipAddr.GetReadableAddress());
        serverInfo.networkInfo.push_back(networdInfo);
    }

    rankTable.serverList.push_back(serverInfo);
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetCloudHcomInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable,
    const std::string &identify, u32 &rank)
{
    HCCL_DEBUG("get cloud hcom info: identify[%s]", identify.c_str());
    // group_count
    std::string groupCount;
    CHK_RET(GetJsonProperty(fileContent_, "group_count", groupCount));
    HCCL_DEBUG("%s.json -> group_count: %s", fileName_.c_str(), groupCount.c_str());

    CHK_RET(SalStrToULong(groupCount, HCCL_BASE_DECIMAL, rankTable.groupNum));
    // 校验groupCount ，groupCount不能为0
    if (rankTable.groupNum == 0) {
        HCCL_ERROR("[Get][CloudHcomInfo]errNo[0x%016llx] group num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    devMap_.clear();
    // group_list
    CHK_RET(GetGroupList(params, rankTable));
    CHK_RET(GetSortClouldRankList(rankTable));

    if (!IsTaskNumCalMode()) {
        // 获取当前操作的逻辑ID并转换为物理ID
        s32 deviceLogicId = -1; // device logic id 的无效值
        CHK_RET(hrtGetDevice(&deviceLogicId));

        u32 devicePhyId = INVALID_UINT; // device phy id 的无效值
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devicePhyId));

        for (u32 index = 0; index < rankTable.rankList.size(); index++) {
            HCCL_INFO(" rank: %u  phyId:%u identify:%s podName:%s", rank, devicePhyId, identify.c_str(),
                rankTable.rankList[index].podName.c_str());
            if ((rankTable.rankList[index].podName == identify) &&
                    (rankTable.rankList[index].deviceInfo.devicePhyId == static_cast<s32>(devicePhyId))) {
                rank = rankTable.rankList[index].rankId;
                break;
            }
        }
    }

    CHK_RET(GetDevNum(rankTable.rankList, rankTable.deviceNum));
    rankTable.serverNum = devMap_.size();
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE; // 910A 对应的nicDeploy
    rankTable.rankNum = rankTable.rankList.size();

    // alg_type
    HCCL_INFO("%s.json -> rank %u : deviceNum is %u, serverNum is %u, nicDeploy is %u, rankNum is %u",
        fileName_.c_str(), rank, rankTable.deviceNum, rankTable.serverNum,
        rankTable.nicDeploy, rankTable.rankNum);

    return HCCL_SUCCESS;
}


HcclResult TopoinfoRanktableStandard::GetSortClouldRankList(hccl::RankTable_t &rankTable)
{
    // sort device id in each server
    for (auto iter = devMap_.begin(); iter != devMap_.end(); iter++) {
        if (!(iter->second).empty()) {
            std::sort((iter->second).begin(), (iter->second).end(), [&](const RankInfo_t &a,
                const RankInfo_t &b) -> bool {return a.deviceInfo.devicePhyId < b.deviceInfo.devicePhyId;});
        }
    }

    // sort rank and filling rankList
    rankTable.rankList.clear();
    u32 initialRank = 0;
    for (auto iterMap = devMap_.begin(); iterMap != devMap_.end(); iterMap++) {
        if (!(iterMap->second).empty()) {
            for (u32 vecIndex = 0; vecIndex < (iterMap->second).size(); vecIndex++) {
                (iterMap->second)[vecIndex].rankId = initialRank;
                rankTable.rankList.push_back((iterMap->second)[vecIndex]);
                initialRank++;
            }
        }
    }
    return HCCL_SUCCESS;
}


HcclResult TopoinfoRanktableStandard::GetSingleGroupDeviceCount(nlohmann::json &obj, u32 objIndex,
    hccl::RankTable_t &rankTable, u32 &deviceNum)
{
    // device_num
    std::string strDeviceNum;

    if (!cloudFlag_) {
        CHK_RET(GetJsonArrayMemberProperty(obj, objIndex, "device_num", strDeviceNum));
    } else {
        CHK_RET(GetJsonArrayMemberProperty(obj, objIndex, "device_count", strDeviceNum));
    }
    HCCL_DEBUG("%s.json -> device_num: %s", fileName_.c_str(), strDeviceNum.c_str());
    CHK_RET(SalStrToULong(strDeviceNum, HCCL_BASE_DECIMAL, deviceNum));
    if (deviceNum == 0) {
        HCCL_ERROR("[Get][SingleGroupDeviceCount]errNo[0x%016llx] device num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    rankTable.deviceNum += deviceNum;

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetLabSingleGroup(nlohmann::json &obj, u32 objIndex, hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable, u32 instanceNum)
{
    u32 uDeviceNum = 0;
    u32 uServerNum = 0;
    // device_num
    CHK_RET(GetSingleGroupDeviceCount(obj, objIndex, rankTable, uDeviceNum));

    // server_num
    std::string serverNum;
    CHK_RET(GetJsonArrayMemberProperty(obj, objIndex, "server_num", serverNum));
    HCCL_DEBUG("%s.json -> server_num: %s", fileName_.c_str(), serverNum.c_str());

    CHK_RET(SalStrToULong(serverNum, HCCL_BASE_DECIMAL, uServerNum));
    if (uServerNum == 0) {
        HCCL_ERROR("[Get][LabSingleGroup]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    rankTable.serverNum += uServerNum;

    if (params.deviceType != DevType::DEV_TYPE_310P3) {
        CHK_RET(CheckAverageDev(uDeviceNum, uServerNum));
    }
    // server_list
    if (static_cast<u32>(rankTable.nicDeploy) == 0) {  // 网卡挂载在host侧
        CHK_RET(GetServerList(obj, objIndex, rankTable, uServerNum));
    }

    if (instanceNum != uDeviceNum) {
        HCCL_ERROR("[Get][LabSingleGroup]errNo[0x%016llx] instance num error", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    rankTable.rankNum += instanceNum;

    return HCCL_SUCCESS;
}


HcclResult TopoinfoRanktableStandard::GetGroupList(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable)
{
    rankTable.rankList.clear();
    nlohmann::json groupList;
    CHK_RET(GetJsonProperty(fileContent_, "group_list", groupList));

    HCCL_DEBUG("group_list.size[%zu] groupNum[%u]", groupList.size(), rankTable.groupNum);
    CHK_PRT_RET(groupList.size() != rankTable.groupNum, HCCL_ERROR("[Get][GroupList]errNo[0x%016llx] "\
        "groupList size[%zu] error, groupNum[%u]", HCOM_ERROR_CODE(HCCL_E_PARA), groupList.size(),
        rankTable.groupNum), HCCL_E_PARA);

    rankTable.deviceNum = 0;
    rankTable.serverNum = 0;
    rankTable.rankNum = 0;
    for (u32 index = 0; index < groupList.size(); index++) {
        // group_name
        std::string groupName;
        CHK_RET(GetJsonArrayMemberProperty(groupList, index, "group_name", groupName));

        CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_GROUP_NAME, groupName,
            JsonCheckOpType::CHECK_OP_TYPE_INSERT));
        HCCL_DEBUG("%s.json -> group_name: %s", fileName_.c_str(), groupName.c_str());

        // instance_count
        std::string instanceCount;
        CHK_RET(GetJsonArrayMemberProperty(groupList, index, "instance_count", instanceCount));
        HCCL_DEBUG("%s.json -> rank_count: %s", fileName_.c_str(), instanceCount.c_str());

        u32 instanceNum = 0;
        CHK_RET(SalStrToULong(instanceCount, HCCL_BASE_DECIMAL, instanceNum));
        if (instanceNum == 0) {
            HCCL_ERROR("[Get][GroupList]errNo[0x%016llx] instance num[%u] invalid", HCOM_ERROR_CODE(HCCL_E_PARA),
                instanceNum);
            return HCCL_E_PARA;
        }
        u32 deviceNum = 0;
        if (!cloudFlag_) {
            CHK_RET(GetLabSingleGroup(groupList, index, params, rankTable, instanceNum));
            deviceNum = instanceNum;
        } else {
            CHK_RET(GetSingleGroupDeviceCount(groupList, index, rankTable, deviceNum));
        }

        nlohmann::json instanceList;
        CHK_RET(GetJsonArrayMemberProperty(groupList, index, "instance_list", instanceList));

        CHK_RET(GetInstanceList(instanceList, params, rankTable, instanceNum, deviceNum));
    }

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetInstanceList(nlohmann::json &instanceList, hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable, u32 instanceNum, u32 deviceNum)
{
    HCCL_DEBUG("get instanceList: instanceNum[%u], deviceNum[%u]", instanceNum, deviceNum);
    u32 checkCount = 0;
    u32 checkDevCount = 0;

    for (u32 podIndex = 0; podIndex < instanceList.size(); podIndex++) {
        std::string serverId;
        CHK_RET(GetJsonArrayMemberProperty(instanceList, podIndex, "server_id", serverId));
        HCCL_DEBUG("%s.json -> server_id: %s", fileName_.c_str(), serverId.c_str());
        if ((!cloudFlag_) && (static_cast<u32>(rankTable.nicDeploy) == 0)) {
            CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, serverId,
                JsonCheckOpType::CHECK_OP_TYPE_FIND));
        }

        u32 serverIdx;
        GenerateServerIdx(serverId, serverIdx);
        HCCL_DEBUG("instance id[%u]:[%s], serverIdx[%u]", podIndex, serverId.c_str(), serverIdx);

        nlohmann::json deviceList;
        CHK_RET(GetJsonArrayMemberProperty(instanceList, podIndex, "devices", deviceList));
        if (cloudFlag_) {
            CHK_RET(GetCloudDevList(instanceList, podIndex, deviceList, serverId, serverIdx));
        } else {
            CHK_RET(GetDevList(instanceList, podIndex, deviceList, params, rankTable, serverId, serverIdx));
        }
        checkDevCount = checkDevCount + deviceList.size();
        checkCount++;
    }

    HCCL_DEBUG("instance_num %u, check_count %u", instanceNum, checkCount);
    bool hcclCheck = (instanceNum != checkCount) || (deviceNum != checkDevCount);
    CHK_PRT_RET(hcclCheck,
        HCCL_ERROR("[Get][InstanceList]errNo[0x%016llx] check instanceNum[%u] or devNum[%u] error, checkInstance[%u], "\
            "checkDev[%u]", HCOM_ERROR_CODE(HCCL_E_PARA), instanceNum, deviceNum, checkCount,
            checkDevCount), HCCL_E_PARA);
    
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetCloudDevList(nlohmann::json &instanceList, u32 podIndex,
    nlohmann::json &deviceList, std::string &serverId, u32 &serverIdx)
{
    std::string podName;
    CHK_RET(GetJsonArrayMemberProperty(instanceList, podIndex, "pod_name", podName));
    HCCL_DEBUG("%s.json -> pod_name: %s", fileName_.c_str(), podName.c_str());
    
    CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_POD_NAME, podName,
        JsonCheckOpType::CHECK_OP_TYPE_INSERT));
    for (u32 deviceIndex = 0; deviceIndex < deviceList.size(); deviceIndex++) {
        std::string strDevid;
        CHK_RET(GetJsonArrayMemberProperty(deviceList, deviceIndex, "device_id", strDevid));

        u32 devicePhyId = 0;
        CHK_RET(SalStrToULong(strDevid, HCCL_BASE_DECIMAL, devicePhyId));
        if (devicePhyId > (HCCL_AISERVER_DEVICE_NUM - 1)) { // deviceid in 0 ~ 7
            HCCL_ERROR("[Get][CloudDevList]errNo[0x%016llx] devicePhyId[%u] more than [%u] is invalid",
                HCOM_ERROR_CODE(HCCL_E_PARA), devicePhyId, HCCL_AISERVER_DEVICE_NUM - 1);
            return HCCL_E_PARA;
        }
        HCCL_DEBUG("%s.json -> device_id: %s", fileName_.c_str(), strDevid.c_str());

        RankInfo_t rankinfo;
        // 1.非cloud场景下，网卡挂载在device侧2.cloud场景
        HcclIpAddress ipAddr;
        if (instanceList.size() > 1) {
            std::string deviceIp;
            CHK_RET(GetJsonArrayMemberProperty(deviceList, deviceIndex, "device_ip", deviceIp));
            HCCL_DEBUG("%s.json -> device_ip: %s", fileName_.c_str(), deviceIp.c_str());
            if (!deviceIp.empty()) {
                CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_DEVICE_IP, deviceIp,
                    JsonCheckOpType::CHECK_OP_TYPE_INSERT));

                HcclResult ret = ConvertIpAddress(deviceIp, ipAddr);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Get][CloudDevList]deviceIp[%s] is invalid", deviceIp.c_str()), ret);
            }
        } else {
            HCCL_INFO("sigle server don't need devIP");
        }
        rankinfo.deviceInfo.deviceIp.push_back(ipAddr);
        rankinfo.serverId = serverId;
        rankinfo.serverIdx = serverIdx;
        rankinfo.deviceInfo.devicePhyId = devicePhyId;
        rankinfo.podName = podName;

        // 回填dev_map_
        auto iter = devMap_.find(serverId);
        if (iter != devMap_.end()) {
            iter->second.push_back(rankinfo);  // 存在该服务器内相关dev的对应信息
        } else {
            std::vector<RankInfo_t> vecDev;
            vecDev.push_back(rankinfo);
            devMap_.insert(std::make_pair(serverId, vecDev));  // 不存在则新增一条map记录
        }
        HCCL_DEBUG("%s.json->serverId[%s], podName[%s], devicePhyId[%d]", fileName_.c_str(),
            rankinfo.serverId.c_str(), rankinfo.podName.c_str(), rankinfo.deviceInfo.devicePhyId);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetDevList(nlohmann::json &instanceList, u32 podIndex,
    nlohmann::json &deviceList, hccl::HcclCommParams &params, hccl::RankTable_t &rankTable,
    std::string &serverId, u32 &serverIdx)
{
    std::string rankId;
    CHK_RET(GetJsonArrayMemberProperty(instanceList, podIndex, "rank_id", rankId));
    HCCL_DEBUG("%s.json -> rankId: %s", fileName_.c_str(), rankId.c_str());
    for (u32 deviceIndex = 0; deviceIndex < deviceList.size(); deviceIndex++) {
        std::string strDevid;
        CHK_RET(GetJsonArrayMemberProperty(deviceList, deviceIndex, "device_id", strDevid));

        u32 devicePhyId = 0;
        CHK_RET(SalStrToULong(strDevid, HCCL_BASE_DECIMAL, devicePhyId));
        if ((params.deviceType != DevType::DEV_TYPE_310P3 &&
            params.deviceType != DevType::DEV_TYPE_910B &&
            params.deviceType != DevType::DEV_TYPE_910_73) &&
            (devicePhyId > (HCCL_AISERVER_DEVICE_NUM - 1))) {
            HCCL_ERROR("[Get][DevList]errNo[0x%016llx] device_id[%u] more than 7 is invalid",
                HCOM_ERROR_CODE(HCCL_E_PARA), devicePhyId);
            return HCCL_E_PARA;
        }
        HCCL_DEBUG("%s.json -> device_id: %s", fileName_.c_str(), strDevid.c_str());

        RankInfo_t rankinfo;
        // 1.非cloud场景下，网卡挂载在device侧2.cloud场景
        // 推荐网络场景，单servere需要使用RDMA网卡
        HcclIpAddress ipAddr;
        if (rankTable.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE &&
            (rankTable.serverNum > 0)) {
            std::string deviceIp;
            CHK_RET(GetJsonArrayMemberProperty(deviceList, deviceIndex, "device_ip", deviceIp));
            HCCL_DEBUG("%s.json -> device_ip: %s", fileName_.c_str(), deviceIp.c_str());
            if (deviceIp.compare("") != 0) {
                CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_DEVICE_IP, deviceIp,
                    JsonCheckOpType::CHECK_OP_TYPE_INSERT));
                HcclResult ret = ConvertIpAddress(deviceIp, ipAddr);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Get][DevList]deviceIp[%s] is invalid", deviceIp.c_str()), ret);
            }
        } else {
            HCCL_INFO("sigle server don't need devIP");
        }
        rankinfo.deviceInfo.deviceIp.push_back(ipAddr);

        rankinfo.serverId = serverId;
        rankinfo.serverIdx = serverIdx;
        rankinfo.deviceInfo.devicePhyId = devicePhyId;
        if (SalStrToULong(rankId, HCCL_BASE_DECIMAL, rankinfo.rankId) != HCCL_SUCCESS) {
            RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                std::vector<std::string>({ "The rankid in ranktable is invalid.",
                "The ranktable path configured in the training can be found in the plogs." }));
            HCCL_ERROR("[Get][DevList]errNo[0x%016llx] rankid[%s] is invalid",
                HCOM_ERROR_CODE(HCCL_E_PARA), rankId.c_str());
            return HCCL_E_PARA;
        }

        rankinfo.podName = "";  // podname在实验室场景下置空
        rankId = "";
        rankTable.rankList.push_back(rankinfo);
        HCCL_DEBUG("%s.json->rankId[%u], serverId[%s], devicePhyId[%d]", fileName_.c_str(),
            rankinfo.rankId, rankinfo.serverId.c_str(), rankinfo.deviceInfo.devicePhyId);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableStandard::GetDeployMode(bool &cloudFlag) const
{
    cloudFlag = fileContent_.find("deploy_mode") == fileContent_.end() ;
    return HCCL_SUCCESS;
}
