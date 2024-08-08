/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableConcise.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "log.h"
#include "externalinput_pub.h"
#include "hccl_comm_pub.h"
#include "config.h"
#include "workflow_pub.h"

using namespace std;
using namespace hccl;


TopoinfoRanktableConcise::TopoinfoRanktableConcise(const std::string &rankTableM, const std::string &identify)
    : TopoInfoRanktableParser(rankTableM, identify)
{
}

TopoinfoRanktableConcise::~TopoinfoRanktableConcise()
{
}

HcclResult TopoinfoRanktableConcise::Init()
{
    CHK_RET(LoadRankTableString(rankTableFile_));
    CHK_RET(ParserClusterInfo(params_, rankTable_));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetClusterInfo(RankTable_t &clusterInfo)
{
    clusterInfo.nicDeploy = rankTable_.nicDeploy;
    clusterInfo.deviceNum = rankTable_.deviceNum;
    clusterInfo.serverNum = rankTable_.serverNum;
    clusterInfo.superPodNum = rankTable_.superPodNum;
    clusterInfo.rankNum = rankTable_.rankNum;
    clusterInfo.rankList = rankTable_.rankList;
    clusterInfo.serverList = rankTable_.serverList;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSelfClusterInfo(HcclCommParams &params)
{
    // 获取芯片类型信息
    params.deviceType = params_.deviceType;
    params.rank = params_.rank;
    params.userRank = params_.rank;
    params.logicDevId = params_.logicDevId;
    params.totalRanks = params_.totalRanks;
    params.serverId = params_.serverId;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetClusterInfo(hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable)
{
    CHK_RET(GetClusterInfo(rankTable));
    CHK_RET(GetSelfClusterInfo(params));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::ParserClusterInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable)
{
    if (!IsTaskNumCalMode()) {
        CHK_RET(hrtGetDeviceType(params.deviceType));
    }
    // 获取ranktable info信息
    CHK_RET(GetRanktableInfo(rankTable));
    for (auto &rankInfo : rankTable.rankList) {
        HCCL_DEBUG("ParserClusterInfo serverId %s, rankId %u, superDeviceId 0x%x, superPodId %s",
            rankInfo.serverId.c_str(), rankInfo.rankId, rankInfo.superDeviceId, rankInfo.superPodId.c_str());
    }

    if (IsTaskNumCalMode()) {
        HCCL_INFO("[ParserClusterInfo] get task num cal mode.");
        return HCCL_SUCCESS;
    }

    CHK_RET(CheckNicDeployConsistence(rankTable));
    rankTable.nicDeploy = (rankTable.rankList.size() != 0 &&
        !rankTable.rankList.begin()->hostIp.IsInvalid()) ? NICDeployment::NIC_DEPLOYMENT_HOST :
            NICDeployment::NIC_DEPLOYMENT_DEVICE;
    std::sort(rankTable.rankList.begin(), rankTable.rankList.end(),
        [&](const RankInfo_t &a, const RankInfo_t &b) -> bool {return a.rankId < b.rankId;});

    u32 rankId = INVALID_VALUE_RANKID;
    if (IsAllDigit(identify_.c_str()) != HCCL_SUCCESS ||
        SalStrToULong(identify_, HCCL_BASE_DECIMAL, rankId) != HCCL_SUCCESS) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "The rank_id must be an digit.", "The ranktable path configured "
            "in the training can be found in the plogs." }));
        HCCL_ERROR("[Parser][ClusterInfo]errNo[0x%016llx] rank_id[%s] is invalid",
            HCOM_ERROR_CODE(HCCL_E_PARA), identify_.c_str());
        return HCCL_E_PARA;
    }

    // 校验rank id合法性
    if (rankId >= rankTable.rankList.size()) {
        RPT_ENV_ERR(true, "EI0004", std::vector<std::string>({"error_reason", "ranktable_path"}), \
            std::vector<std::string>({"Use a rank ID that exceeds the rank size in the ranktable.", rankTableFile_}));
        HCCL_ERROR("[Parse][ClusterInfo]rankId[%u] is invalid", rankId);
        return HCCL_E_PARA;
    }

    RPT_ENV_ERR(rankId != rankTable.rankList[rankId].rankId, "EI0004",
        std::vector<std::string>({ "error_reason", "ranktable_path" }),
        std::vector<std::string>(
        { "The 'rank_id' in the ranktable must start from 0 or it is used repeatedly", rankTableFile_ }));
    CHK_PRT_RET(rankId != rankTable.rankList[rankId].rankId,
        HCCL_ERROR("[Parse][ClusterInfo]check rankList[%u] rankId[%u] failed", rankId,
            rankTable.rankList[rankId].rankId), HCCL_E_UNAVAIL);
    u32 devId = rankTable.rankList[rankId].deviceInfo.devicePhyId;
    CHK_RET(hrtGetDevice(&params.logicDevId));

    u32 devicePhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(params.logicDevId), devicePhyId));
    RPT_ENV_ERR(devicePhyId != static_cast<u32>(devId), "EI0004",
        std::vector<std::string>({ "error_reason", "ranktable_path" }),
        std::vector<std::string>({ "The ranktable config devId is inconsistent with "
        "the local devId.",
        rankTableFile_ }));

    CHK_PRT_RET(devicePhyId != static_cast<u32>(devId),
        HCCL_ERROR("[Parse][ClusterInfo]ranktable config devId[%d],but local devId[%u]",
        devId, devicePhyId), HCCL_E_UNAVAIL);

    params.rank = rankId;
    params.serverId = rankTable.rankList[rankId].serverId;
    params.totalRanks = rankTable.rankNum;

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::CheckNicDeployConsistence(RankTable_t &clusterInfo) const
{
    CHK_PRT_RET(clusterInfo.rankList.size() == 0, HCCL_DEBUG("rank list size is 0, skip nic deply check."),
        HCCL_SUCCESS);
    NICDeployment tmpNicDeply = clusterInfo.rankList.begin()->hostIp.IsInvalid() ?
        NICDeployment::NIC_DEPLOYMENT_DEVICE : NICDeployment::NIC_DEPLOYMENT_HOST;
    for (auto &it:clusterInfo.rankList) {
        CHK_PRT_RET((tmpNicDeply == NICDeployment::NIC_DEPLOYMENT_DEVICE && !it.hostIp.IsInvalid()) ||
        (tmpNicDeply == NICDeployment::NIC_DEPLOYMENT_HOST &&
            it.hostIp.IsInvalid()), HCCL_ERROR("[Get][RanktableInfo]errNo"
            "[0x%016llx] hostIp config bettewn ranks is different.", HCOM_ERROR_CODE(HCCL_E_PARA)),  HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetRanktableInfo(RankTable_t &clusterInfo)
{
    // server_list
    CHK_RET(GetServerList(fileContent_, clusterInfo));
    CHK_RET(GetSuperPodList(fileContent_, clusterInfo));
    CHK_RET(GetDevNum(clusterInfo.rankList, clusterInfo.deviceNum));
    clusterInfo.rankNum = clusterInfo.rankList.size();
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetServerList(const nlohmann::json &obj, RankTable_t &clusterInfo)
{
    clusterInfo.serverList.clear();
    nlohmann::json serverList;
    CHK_RET(GetJsonProperty(obj, "server_list", serverList));
    HCCL_DEBUG("[%s.json] -> server_list: size:[%zu]", fileName_.c_str(), serverList.size());

    // 获取serverCount并校验
    std::string serverCount;
    HcclResult ret = GetJsonProperty(obj, "server_count", serverCount);
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[%s.json] -> group_count: [%s]", fileName_.c_str(), serverCount.c_str());
        CHK_RET(SalStrToULong(serverCount, HCCL_BASE_DECIMAL, clusterInfo.serverNum));

        // 校验serverCount
        if (serverList.size() != clusterInfo.serverNum) {
            RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                std::vector<std::string>({ "The 'server_count' in ranktable is invalid.", "The ranktable path "
                "configured in the training can be found in the plogs." }));

            HCCL_ERROR("[Get][ServerList]errNo[0x%016llx] serverList size[%zu] neq server num[%u]",
                HCOM_ERROR_CODE(HCCL_E_PARA), serverList.size(), clusterInfo.serverNum);
            return HCCL_E_PARA;
        }
    } else if (ret == HCCL_E_NOT_FOUND) {
        clusterInfo.serverNum = serverList.size();
        HCCL_WARNING("[Get][ServerList]The 'server_count' in ranktable is not found, "\
            "set server_count to server_list size[%u]", serverList.size());
    } else {
        HCCL_ERROR("[Get][ServerList]get server_count error, ret[%d]", ret);
        return ret;
    }
    if (clusterInfo.serverNum == 0) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "the 'server_list' in the ranktable is empty", "Please check the "
            "'server_list' in ranktable" }));
        HCCL_ERROR("[Get][RanktableInfo]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[Get][ServerList]serverNum is [%u]", clusterInfo.serverNum);

    for (u32 index = 0; index < serverList.size(); index++) {
        // get single server info
        CHK_RET(GetSingleServer(serverList, index, clusterInfo));
    }
    // 获取device
    return HCCL_SUCCESS;
}


HcclResult TopoinfoRanktableConcise::GetSingleServer(const nlohmann::json &serverListObj, u32 objIndex,
    RankTable_t &clusterInfo)
{
    HcclResult ret;
    std::string serverId;
    CHK_RET(GetJsonArrayMemberProperty(serverListObj, objIndex, "server_id", serverId));
    if (serverId.empty()) {
        HCCL_ERROR("[Get][JsonArrayMemberProperty]errNo[0x%016llx] serverId[%s] is empty",
            HCOM_ERROR_CODE(HCCL_E_PARA), serverId.c_str());
        return HCCL_E_PARA;
    }
    // 将serverId添加到资源池,内部会进行IP地址校验，如果资源池中有serverId，则报错
    CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, serverId,
        JsonCheckOpType::CHECK_OP_TYPE_INSERT));
    HCCL_DEBUG("server id[%u]:[%s]", objIndex, serverId.c_str());

    u32 serverIdx;
    GenerateServerIdx(serverId, serverIdx);
    HCCL_DEBUG("server id[%u]:[%s], serverIdx[%u]", objIndex, serverId.c_str(), serverIdx);
    std::string hostNicIp;
    HcclIpAddress hostIp;
    ret = GetJsonArrayMemberProperty(serverListObj, objIndex, "host_ip", hostNicIp);
    CHK_PRT_RET(ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Get][SingleServer]get host ip error"), ret);
    HCCL_DEBUG("[%s.json] -> host_ip: [%s]. ret[%u]", fileName_.c_str(), hostNicIp.c_str(), ret);
    if (ret != HCCL_E_NOT_FOUND) {
        CHK_RET(ConvertIpAddress(hostNicIp, hostIp));
    }

    // 处理ranklist
    ret = GetDeviceList(serverListObj, objIndex, clusterInfo, serverId, serverIdx, hostIp);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][SingleServer]get dev list error:serverId[%s]",
        serverId.c_str()), ret);

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetDeviceList(const nlohmann::json &serverListObj, u32 objIndex,
    RankTable_t &clusterInfo, std::string &serverId, u32 &serverIdx, HcclIpAddress &hostIp)
{
    HCCL_DEBUG("Get GetDeviceList[%u]: serverId[%s]", objIndex, serverId.c_str());

    nlohmann::json deviceList;
    CHK_RET(GetJsonArrayMemberProperty(serverListObj, objIndex, "device", deviceList));

    HCCL_DEBUG("[%s.json] -> device_list: size:%zu", fileName_.c_str(), deviceList.size());

    CHK_PRT_RET(deviceList.size() == 0, HCCL_ERROR("[Get][DeviceList]deviceList size is zero"), HCCL_E_PARA);

    for (u32 index = 0; index < deviceList.size(); index++) {
        // get single server info
        CHK_RET(GetSingleDevice(deviceList, index, clusterInfo, serverId, serverIdx, hostIp));
    }

    // 检查devip的数目是否一致
    u32 rankListSize = clusterInfo.rankList.size();
    CHK_PRT_RET(rankListSize == 0, HCCL_ERROR("[Get][DeviceList]get ranklist is zero"), HCCL_E_PARA);

    u32 checkDeviceIpSize = 0;
    for (u32 index = 0; index < clusterInfo.rankList.size(); index++) {
        if (index == 0) {
            checkDeviceIpSize = clusterInfo.rankList[0].deviceInfo.deviceIp.size();
        }
        if (clusterInfo.rankList[index].deviceInfo.deviceIp.size() != checkDeviceIpSize) {
            HCCL_ERROR("[Get][DeviceList]device[%u] size[%u] neq first device size[%u] error", index,
                clusterInfo.rankList[index].deviceInfo.deviceIp.size(), checkDeviceIpSize);
            return HCCL_E_PARA;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleDevice(const nlohmann::json &deviceListObj, u32 objIndex,
    RankTable_t &clusterInfo, std::string &serverId, u32 &serverIdx, HcclIpAddress &hostIp)
{
    // 获取rank_id
    std::string rankId;
    CHK_RET(GetJsonArrayMemberProperty(deviceListObj, objIndex, "rank_id", rankId));
    HCCL_DEBUG("[%s.json] -> rankId: [%s]", fileName_.c_str(), rankId.c_str());

    // 获取device type
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // 获取device_id
    std::string strDevid;
    CHK_RET(GetJsonArrayMemberProperty(deviceListObj, objIndex, "device_id", strDevid));

    u32 devicePhyId = 0;
    CHK_RET(SalStrToULong(strDevid, HCCL_BASE_DECIMAL, devicePhyId));

    if ((deviceType == DevType::DEV_TYPE_310P3 || deviceType == DevType::DEV_TYPE_910B ||
        deviceType == DevType::DEV_TYPE_910_73) &&  devicePhyId > (MAX_MODULE_DEVICE_NUM - 1)) {
        // deviceid in 0 ~ 15
        HCCL_ERROR("[Get][SingleDevice]errNo[0x%016llx] device_id[%u] more than 15 is invalid",
            HCOM_ERROR_CODE(HCCL_E_PARA), devicePhyId);
        return HCCL_E_PARA;
    } else if ((deviceType != DevType::DEV_TYPE_310P3 && deviceType != DevType::DEV_TYPE_910B &&
        deviceType != DevType::DEV_TYPE_910_73) && devicePhyId > (HCCL_AISERVER_DEVICE_NUM - 1)) {
        // deviceid in 0 ~ 7
        HCCL_ERROR("[Get][SingleDevice]errNo[0x%016llx] device_id[%u] more than 7 is invalid",
            HCOM_ERROR_CODE(HCCL_E_PARA), devicePhyId);
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[%s.json] -> device_id: [%s]", fileName_.c_str(), strDevid.c_str());
    
    RankInfo_t rankinfo;
    rankinfo.serverId = serverId;
    rankinfo.serverIdx = serverIdx;
    rankinfo.hostIp = hostIp;
    rankinfo.deviceInfo.devicePhyId = devicePhyId;

    if (rankinfo.hostIp.IsInvalid()) {
        CHK_RET(GetSingleDeviceIp(deviceListObj, objIndex, clusterInfo, rankinfo));
    }

    if (SalStrToULong(rankId, HCCL_BASE_DECIMAL, rankinfo.rankId) != HCCL_SUCCESS) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "The rankid in ranktable is invalid. Please check ranktable",
            "The ranktable path configured in the training can be found in the plogs." }));
        HCCL_ERROR("[Get][SingleRank]errNo[0x%016llx] rankid[%s] is invalid",
            HCOM_ERROR_CODE(HCCL_E_PARA), rankId.c_str());
        return HCCL_E_PARA;
    }

    rankinfo.podName = "";  // podname在新场景下置空
    rankId = "";

    string version;
    CHK_RET(GetRanktableVersion(version));
    if (version.compare(SUPERPOD_CLUSTER_VERSION) != 0) {
        clusterInfo.rankList.push_back(rankinfo);
        HCCL_DEBUG("[%s.json]->rankId[%u], serverId[%s], devicePhyId[%d]", fileName_.c_str(),
            rankinfo.rankId, rankinfo.serverId.c_str(), rankinfo.deviceInfo.devicePhyId);

        return HCCL_SUCCESS;
    }

    CHK_RET(GetSingleSuperDeviceId(deviceListObj, objIndex, clusterInfo, rankinfo));
    clusterInfo.rankList.push_back(rankinfo);
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::SplitString(const std::string& str, const std::string& strC,
    std::vector<std::string>& strVector) const
{
    std::string::size_type startPos = 0;
    std::string::size_type foundPos = str.find(strC);

    while (foundPos != std::string::npos) {
        strVector.push_back(str.substr(startPos, foundPos - startPos));
        startPos = foundPos + strC.size();
        foundPos = str.find(strC, startPos);
    }
    if (startPos != str.length()) {
        strVector.push_back(str.substr(startPos));
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleDeviceIp(const nlohmann::json &deviceListObj, u32 objIndex,
    RankTable_t &clusterInfo, RankInfo_t &rankinfo)
{
    // 获取device_ip （可能有多个）
    std::string deviceIp;
    HcclResult ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "device_ip", deviceIp);
    // 多机和走roce网卡ranktable必须有“device_ip”字段，单机可以没有
    if (clusterInfo.serverNum > 1 || (GetExternalInputIntraRoceSwitch() == 1)) {
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Get][SingleDeviceIp]'device_ip' is not set correctly,"\
                       "must be set when multi Server or HCCL_INTRA_ROCE_ENABLE enabled"), ret);
    } else if (clusterInfo.serverNum == 1 && ret == HCCL_E_NOT_FOUND) {
        HcclIpAddress invalidAddr;
        rankinfo.deviceInfo.deviceIp.push_back(invalidAddr);
        HCCL_WARNING("[Get][SingleDeviceIp]'device_ip' in ranktable is not set!");
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Get][SingleDeviceIp]errNo[0x%016llx] 'device_ip' is not set correctly",
                HCOM_ERROR_CODE(HCCL_E_PARA)), ret);
    }
    HCCL_DEBUG("[%s.json] -> device_ip: [%s]", fileName_.c_str(), deviceIp.c_str());

    // 处理字符串device_ip
    std::vector<std::string> strDeviceIp;
    if (deviceIp != "") {
        CHK_RET(SplitString(deviceIp, ",", strDeviceIp));

        CHK_PRT_RET(strDeviceIp.size() == 0, HCCL_ERROR("[Get][SingleDeviceIp]in device:deviceip size is zero"),
            HCCL_E_PARA);
        for (u32 index = 0; index < strDeviceIp.size(); index++) {
            CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_DEVICE_IP, strDeviceIp[index],
                JsonCheckOpType::CHECK_OP_TYPE_INSERT));
            HcclIpAddress ipAddr;
            CHK_RET(ConvertIpAddress(strDeviceIp[index], ipAddr));
            rankinfo.deviceInfo.deviceIp.push_back(ipAddr);
        }
    } else {
        HcclIpAddress invalidAddr;
        rankinfo.deviceInfo.deviceIp.push_back(invalidAddr);
        HCCL_WARNING("objIndex[%u],'device_ip' is not set", objIndex);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperDeviceId(const nlohmann::json &deviceListObj, u32 objIndex,
    RankTable_t &clusterInfo, RankInfo_t &rankinfo)
{
    // 获取super_device_id
    HcclResult ret;
    std::string strSuperDeviceId;
    ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "super_device_id", strSuperDeviceId);

    CHK_PRT_RET(ret == HCCL_E_NOT_FOUND,
        HCCL_WARNING("[Get][SingleSuperDeviceId]'super_device_id' is not found"), HCCL_SUCCESS);

    u32 superDeviceId = 0;
    ret = SalStrToULong(strSuperDeviceId, HCCL_BASE_DECIMAL, superDeviceId);
    if (ret != HCCL_SUCCESS) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "The 'super_device_id' must be an digit.",
            "The ranktable path configured in the training can be found in the plogs." }));
        HCCL_ERROR("[Get][SingleSuperDevice]errNo[0x%016llx] super_device_id[%s] is invalid",
            HCOM_ERROR_CODE(HCCL_E_PARA), strSuperDeviceId.c_str());
        return ret;
    }
 
    rankinfo.superDeviceId = superDeviceId;
    HCCL_DEBUG("[%s.json] -> super_device_id: [%s]", fileName_.c_str(), strSuperDeviceId.c_str());
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSuperPodList(const nlohmann::json &obj, RankTable_t &clusterInfo)
{
    string version;
    CHK_RET(GetRanktableVersion(version));
    CHK_PRT_RET(version.compare(SUPERPOD_CLUSTER_VERSION) != 0,
        HCCL_INFO("[Get][SuperPodList]ranktable version[%s], do nothing.", version.c_str()), HCCL_SUCCESS);

    HcclResult ret;
    nlohmann::json superPodList;
    ret = GetJsonProperty(obj, "super_pod_list", superPodList);
    CHK_PRT_RET(ret == HCCL_E_NOT_FOUND,
        HCCL_WARNING("[Get][SuperPodList]'super_pod_list' is not found"), HCCL_SUCCESS);
    CHK_PRT_RET(ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Get][SuperPodList]'super_pod_list' in ranktable is not set correctly, ret[%d]", ret), ret);
    HCCL_DEBUG("[%s.json]super_pod_list -> : size:[%zu]", fileName_.c_str(), superPodList.size());

    for (u32 index = 0; index < superPodList.size(); index++) {
        CHK_RET(GetSingleSuperPod(superPodList, index, clusterInfo));
    }

    clusterInfo.superPodNum = superPodList.size();
    CHK_RET(CheckSuperPodInfo(clusterInfo));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperPod(const nlohmann::json &superPodList, u32 objIndex,
    RankTable_t &clusterInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string superPodId;
    ret = GetJsonArrayMemberProperty(superPodList, objIndex, "super_pod_id", superPodId);
    if (ret != HCCL_SUCCESS || superPodId.empty()) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "the 'super_pod_id' in the ranktable is invalid or empty",
            "Please check the 'super_pod_id' in ranktable" }));
        HCCL_ERROR("[Get][JsonArrayMemberProperty]errNo[0x%016llx] super_pod_id[%s] is invalid",
            HCOM_ERROR_CODE(ret), superPodId.c_str());
        return ret;
    }

    // 将superPodId添加到资源池进行查重校验
    CHK_RET(CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SUPER_POD_ID, superPodId,
        JsonCheckOpType::CHECK_OP_TYPE_INSERT));
    HCCL_DEBUG("superPod id[%u]:[%s]", objIndex, superPodId.c_str());

    // 处理ranklist
    ret = GetSuperPodServerList(superPodList, objIndex, clusterInfo, superPodId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][SingleSuperPod]get server list error:superPodId[%s]",
        superPodId.c_str()), ret);

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSuperPodServerList(const nlohmann::json &superPodList, u32 objIndex,
    RankTable_t &clusterInfo, std::string superPodId)
{
    HCCL_DEBUG("GetSuperPodServerList[%u]: superPodId[%s]", objIndex, superPodId.c_str());

    nlohmann::json superPodServerList;
    CHK_RET(GetJsonArrayMemberProperty(superPodList, objIndex, "server_list", superPodServerList));
    for (u32 index = 0; index < superPodServerList.size(); index++) {
        // get single super pod server info
        CHK_RET(GetSingleSuperPodSever(superPodServerList, index, clusterInfo, superPodId));
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperPodSever(const nlohmann::json &superPodServerList, u32 objIndex,
    RankTable_t &clusterInfo, std::string superPodId)
{
    std::string serverId;
    HcclResult ret = HCCL_SUCCESS;
    do {
        // 获取server_id
        ret = GetJsonArrayMemberProperty(superPodServerList, objIndex, "server_id", serverId);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Get][SingleSuperPodSever]errNo[0x%016llx]server_id is not found or invalid",
            HCCL_ERROR_CODE(ret)),);

        // 将super_pod_list下的server_id在资源池中进行校验
        ret = CheckUniqueAndInsertPool(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, serverId,
            JsonCheckOpType::CHECK_OP_TYPE_FIND);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Get][SingleSuperPodSever]errNo[0x%016llx]server_id[%s] is not found in server_list",
            HCCL_ERROR_CODE(ret), serverId.c_str()),);
        HCCL_DEBUG("[%s.json]super_pod_list -> server_id: [%s]", fileName_.c_str(), serverId.c_str());
    } while (0);

    // server_id未找到, 或与server_list中的server_id不一致
    if (ret != HCCL_SUCCESS) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "the 'server_id' in the ranktable 'super_pod_list' is invalid "\
            "or not found in server_list", "Please check the 'server_id' in ranktable" }));
        return ret;
    }

    bool isFound = false;
    for (RankInfo_t& rankInfo : clusterInfo.rankList) {
        if (rankInfo.serverId == serverId) {
            rankInfo.superPodId = superPodId;
            isFound = true;
        }
    }
    CHK_PRT_RET(isFound == false,
        HCCL_ERROR("[Get][SingleSuperPodSever]server_id[%s] in super_pod_list is not in server_list",
        serverId.c_str()), HCCL_E_PARA);

    HCCL_DEBUG("[%s.json]super_pod_list -> server_id[%s], super_pod_id[%s]",
        fileName_.c_str(), serverId.c_str(), superPodId.c_str());

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::CheckSuperPodInfo(RankTable_t &clusterInfo) const
{
    std::map<std::string, std::set<std::string>> superPodMap; // superPodId -> serverId
    std::map<std::string, std::set<u32>> superPodSdidMap;    // super_pod_id -> superDeviceId
    for (RankInfo_t& rankInfo : clusterInfo.rankList) {
        auto it  = superPodMap.find(rankInfo.superPodId);
        if (it == superPodMap.end()) {
            std::set<std::string> serverIdSet;
            serverIdSet.insert(rankInfo.serverId);
            superPodMap.insert({rankInfo.superPodId, serverIdSet});
        } else if (it->second.find(rankInfo.serverId) == it->second.end()) {
            it->second.insert(rankInfo.serverId);
        }
        // 用户忘记配置，superDeviceId等于无效值
        CHK_PRT_RET(rankInfo.superDeviceId == INVALID_UINT,
            HCCL_ERROR("[Check][SuperPodInfo]superDeviceId[0x%x] is invalid in rankId[%u], "
            "the configuration may be missing, please check the ranktable config!",
            rankInfo.superDeviceId, rankInfo.rankId), HCCL_E_PARA);
        
        auto iter = superPodSdidMap.find(rankInfo.superPodId);
        if (iter == superPodSdidMap.end()) {
            std::set<u32> superDeviceIdSet;
            superDeviceIdSet.insert(rankInfo.superDeviceId);
            superPodSdidMap.insert({rankInfo.superPodId, superDeviceIdSet});
        } else if (iter->second.find(rankInfo.superDeviceId) == iter->second.end()) {
            iter->second.insert(rankInfo.superDeviceId);
        } else {
            // 超节点内superDeviceId在超节点内唯一
            CHK_PRT_RET(iter->second.find(rankInfo.superDeviceId) != iter->second.end(),
                HCCL_ERROR("[Verify][SuperPodInfo]superDeviceId[0x%x] in superPod[%s]"
                "is already exist.",
                rankInfo.superDeviceId, iter->first.c_str()),
                HCCL_E_PARA);
        }
    }

    u32 serverNumPerPod = 0;
    u32 serverNumTotal = 0;
    for (auto it = superPodMap.begin(); it != superPodMap.end(); ++it) {
        if (it == superPodMap.begin()) {
            serverNumPerPod = it->second.size();
        }
        // 校验每个超节点的server数相等
        CHK_PRT_RET(serverNumPerPod != it->second.size(),
            HCCL_ERROR("[Get][SuperPodList]serverNum[%u] in superPodId[%s] is not equal to serverNum[%u] "\
            "in superPodId[%s]", serverNumPerPod, it->first.c_str(), it->second.size(),
            superPodMap.begin()->first.c_str()), HCCL_E_PARA);
        serverNumTotal += serverNumPerPod;
    }

    // 校验super_pod_list和原有server_list的server数量一致
    CHK_PRT_RET(serverNumTotal != clusterInfo.serverNum,
        HCCL_ERROR("[Get][SuperPodList]serverNum[%u] in super_pod_list and serverNum[%u] in server_list "\
        "are inconsistent", serverNumTotal, clusterInfo.serverNum), HCCL_E_PARA);
    return HCCL_SUCCESS;
}
