/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_parse.h"
#include <string>
#include <unordered_set>
#include <algorithm>
#include "config.h"

using namespace std;

struct HcclAiServerValid4PRanksVectorHashFuc {
    std::size_t operator()(const std::vector<s32> key) const
    {
        size_t ret = 0;
        for (auto it : key) {
            ret ^= it;
        }
        return ret;
    }
};

// aiserver内连接信息rank合法选择
const std::unordered_set<std::vector<s32>, HcclAiServerValid4PRanksVectorHashFuc> HCCL_AISERVER_VAILD_4P_RANKS = {
    {0, 1, 4, 5},
    {0, 2, 4, 6},
    {0, 3, 4, 7},
    {1, 2, 5, 6},
    {1, 3, 5, 7},
    {2, 3, 6, 7},
    {0, 1, 2, 3},
    {4, 5, 6, 7}
};

namespace hccl {
TopoInfoParse::TopoInfoParse()
{
}

TopoInfoParse::~TopoInfoParse()
{
}

HcclResult TopoInfoParse::Init(const RankTable_t &rankTable, const std::string &serverId, const u32 deviceNumPerServer)
{
    CHK_PRT_RET(deviceNumPerServer == 0, HCCL_ERROR("cur device num per server is 0"), HCCL_E_PARA);
    deviceNum_ = rankTable.deviceNum;
    serverNum_ = rankTable.serverNum;
    superPodNum_ = rankTable.superPodNum;
    serverId_ = serverId;
    nicDeploy_ = rankTable.nicDeploy;
    deviceNumPerServer_ = deviceNumPerServer;
    multiServerDiffDeviceNumMode_ = (serverNum_ * deviceNumPerServer_) == deviceNum_ ? false : true;
    CHK_RET(hrtGetDeviceType(deviceType_));
    for (auto rankInfo : rankTable.rankList) {
        RankInfo curRankInfo;
        curRankInfo.devicePhyId = rankInfo.deviceInfo.devicePhyId;
        curRankInfo.serverId = rankInfo.serverId;
        curRankInfo.userRank = rankInfo.rankId;
        curRankInfo.nicIp = rankInfo.deviceInfo.deviceIp;
        curRankInfo.superDeviceId = rankInfo.superDeviceId;
        curRankInfo.superPodId = rankInfo.superPodId;
        rankList_.push_back(curRankInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::Init(const std::vector<RankInfo> &rankList, const std::string &serverId,
    const u32 deviceNumPerServer)
{
    CHK_PRT_RET(deviceNumPerServer == 0, HCCL_ERROR("cur device num per server is 0"), HCCL_E_PARA);
    rankList_ = rankList;
    serverId_ = serverId;
    deviceNumPerServer_ = deviceNumPerServer;
    CHK_RET(GetDevNum(rankList, deviceNum_));
    CHK_RET(GetServerNum(rankList, serverNum_));
    CHK_RET(GetSuperPodNum(rankList, superPodNum_));
    multiServerDiffDeviceNumMode_ = (serverNum_ * deviceNumPerServer_) == deviceNum_ ? false : true;
    CHK_RET(hrtGetDeviceType(deviceType_));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::GetServerInnerLinkInfo(std::unordered_map<u32, u32> &pairLinkCounter,
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> &pairLinkInfo)
{
    std::vector<RankInfo> serverInnerInfo;
    CHK_RET(TransformRankInfoByServerId(serverInnerInfo));

    CHK_PRT_RET(serverInnerInfo.size() == 0,
        HCCL_ERROR("[Get][ServerInnerLinkInfo]server info input is empty, "
        "serverid[%s]",
        serverId_.c_str()),
        HCCL_E_PARA);
    pairLinkInfo.clear();
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::PXI_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] = 0;
    for (auto it_local : serverInnerInfo) {
        for (auto it_dest : serverInnerInfo) {
            if (it_local.devicePhyId == it_dest.devicePhyId || it_local.devicePhyId == HOST_DEVICE_ID ||
                it_dest.devicePhyId == HOST_DEVICE_ID) {
                continue;
            }
            LinkTypeInServer linkType;
            CHK_RET(hrtGetPairDeviceLinkType(it_local.devicePhyId, it_dest.devicePhyId, linkType));
            pairLinkInfo[static_cast<u32>(linkType)][it_local.devicePhyId].push_back(it_dest.devicePhyId);
            pairLinkCounter[static_cast<u32>(linkType)]++;
        }
    }
    for (auto it : pairLinkInfo) {
        HCCL_DEBUG("pair link information linkType[%u], size[%llu]", it.first, it.second.size());
    }
    for (auto it : pairLinkCounter) {
        HCCL_DEBUG("pair link counter information linkType[%u], size[%llu]", it.first, it.second);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::TransformRankInfoByServerId(std::vector<hccl::RankInfo> &serverInnerInfo)
{
    // 按server重新组织rank信息，便于后续校验及信息填写
    for (auto tmpRankInfo : rankList_) {
        if (tmpRankInfo.serverId == serverId_) {
            serverInnerInfo.push_back(tmpRankInfo);
        }
    }
    // 按设备Id从小到大的顺序排序
    std::sort(serverInnerInfo.begin(), serverInnerInfo.end(),
        [](const RankInfo &left, const RankInfo &right) { return left.devicePhyId < right.devicePhyId; });
    return HCCL_SUCCESS;
}

// nicIdx不只是校验，还有修改
HcclResult TopoInfoParse::ParseAndCheck(std::vector<u32> &nicIdx)
{
    CHK_RET(CheckInterServerDeviceId());
    CHK_RET(CheckRankTableNicInfo(nicIdx));
    CHK_RET(CheckServerInnerRankInfo());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::Check()
{
    CHK_RET(CheckInterServerDeviceId());
    CHK_RET(CheckServerInnerRankInfo());
    return HCCL_SUCCESS;
}

// server间device选取是否对称校验
HcclResult TopoInfoParse::CheckInterServerDeviceId()
{
    if (serverNum_ == 0) {
        HCCL_ERROR("[Check][DeviceId]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    std::map<std::string, std::set<s32>> serverDeviceMapList;
    for (auto it = rankList_.begin(); it != rankList_.end(); it++) {
        std::string tmpServerId = it->serverId;
        auto search = serverDeviceMapList.find(tmpServerId);
        if (search != serverDeviceMapList.end()) {
            auto rs = serverDeviceMapList[tmpServerId].insert(it->devicePhyId);
            if (it->devicePhyId == HOST_DEVICE_ID) {
                continue;
            }
            if (!rs.second) {
                RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                    std::vector<std::string>({ "device id repeat for one server. Please check ranktable",
                    "The ranktable path configured in the training can be found in the plogs." }));
                HCCL_ERROR("[Check][DeviceId]errNo[0x%016llx] check ranklist[%u], device id repeat for one server",
                    HCOM_ERROR_CODE(HCCL_E_PARA), it->userRank);
                return HCCL_E_PARA;
            }
        } else {
            std::set<s32> deviceSet;
            deviceSet.insert(it->devicePhyId);
            serverDeviceMapList.insert(std::pair<std::string, std::set<s32>>(tmpServerId, deviceSet));
        }
    }
    if (serverDeviceMapList.size() == 0) {
        HCCL_ERROR("[Check][DeviceId]errNo[0x%016llx] for all ranklist, server num is zero",
            HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::CheckAndAssignNicInfo(std::vector<u32> &nicIdx)
{
    if (serverNum_ == 1 || superPodNum_ == 1) {
        nicIdx.clear();
        for (u32 index = 0; index < rankList_.size(); index++) {
            if (serverId_ == rankList_[index].serverId && (rankList_[index].devicePhyId != HOST_DEVICE_ID)) {
                nicIdx.push_back(static_cast<u32>(rankList_[index].devicePhyId));
            }
        }
        std::sort(nicIdx.begin(), nicIdx.end());
    } else if (deviceNumPerServer_ == HCCL_AISERVER_DEVICE_NUM && !multiServerDiffDeviceNumMode_) {
        CHK_PRT_RET(nicIdx.size() == 0, HCCL_ERROR("[CheckAndAssign][NicInfo]nic idx size is 0"), HCCL_E_PARA);
        CHK_PRT_RET(deviceNum_ == 0, HCCL_ERROR("[CheckAndAssign][NicInfo]device num is 0"), HCCL_E_PARA);

        bool bRet = deviceNum_ % nicIdx.size() != 0; // 在n台server中，校验网口总数是否为1n/2n/4n/8n
        CHK_PRT_RET(bRet, HCCL_ERROR("[CheckAndAssign][NicInfo]nic total num[%zu] error", nicIdx.size()), HCCL_E_PARA);

        if (nicIdx.size() != deviceNumPerServer_) {
            // 网口裁剪
            // 按server重新组织rank信息
            std::map<std::string, std::vector<u32> > severNicsMap;
            for (size_t index = 0; index < rankList_.size(); ++index) {
                std::string serverId = rankList_[index].serverId;

                if (rankList_[index].nicIp.size() == 0 || rankList_[index].nicIp[0].IsInvalid()) {
                    continue;
                }
                // 以serverID为索引，将server下的ranks放入vector
                auto itr = severNicsMap.find(serverId);
                if (itr != severNicsMap.end()) {
                    itr->second.push_back(rankList_[index].devicePhyId);
                } else {
                    std::vector<u32> nicList;
                    nicList.push_back(rankList_[index].devicePhyId);
                    std::pair<std::string, std::vector<u32>> nicInfoPair(serverId, nicList);
                    severNicsMap.insert(nicInfoPair);
                }
            }

            // 每个server下的nicList按设备Id从小到大的顺序排序
            for (auto &iter : severNicsMap) {
                std::sort(iter.second.begin(), iter.second.end());
                if (nicIdx != iter.second) {
                    HCCL_ERROR("nic list should be the same between servers");
                    return HCCL_E_PARA;
                }
            }
        } else {
            // 网口满配
            return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}

// nicIdx做填充，nicIdx也是deviceId，deviceId做过的校验这里不再重复
HcclResult TopoInfoParse::CheckRankTableNicInfo(std::vector<u32> &nicIdx)
{
    // 在8P均使用的情况下校验nic的选择信息是否正确
    if (nicDeploy_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(CheckAndAssignNicInfo(nicIdx));
    } else if (nicDeploy_ == NICDeployment::NIC_DEPLOYMENT_HOST &&
        serverNum_ == deviceNum_ / HCCL_AISERVER_DEVICE_NUM) {
        nicIdx.assign({ 0, 1, 2, 3, 4, 5, 6, 7 }); // 如果每个server8个rank且为host nic，则网口为满配
    }
    return HCCL_SUCCESS;
}

// 校验server内4p场景下deivce选取是否合法，2p与标卡场景重合
HcclResult TopoInfoParse::CheckServerInnerRankInfo()
{
    // 校验server内device选取
    std::vector<s32> serverInnerDeviceInfo;
    for (u32 index = 0; index < rankList_.size(); index++) {
        if (serverId_ == rankList_[index].serverId && rankList_[index].devicePhyId != HOST_DEVICE_ID) {
            /* 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数 */
            serverInnerDeviceInfo.push_back(rankList_[index].devicePhyId);
        }
    }
    std::sort(serverInnerDeviceInfo.begin(), serverInnerDeviceInfo.end());

    if (deviceType_ == DevType::DEV_TYPE_910) {
        if (deviceNumPerServer_ != HCCL_DEVICE_NUM_FOUR) {
            return HCCL_SUCCESS;
        }
        std::string selectedDevice = "selected devices:";
        for (auto devicePhyId : serverInnerDeviceInfo) {
            selectedDevice += std::to_string(devicePhyId);
            selectedDevice += " ";
        }
        if (HCCL_AISERVER_VAILD_4P_RANKS.find(serverInnerDeviceInfo) == HCCL_AISERVER_VAILD_4P_RANKS.end()) {
            std::string errorManager =
                "The number of selected devices on the current server is 4. Please check the devices "
                "selected in ranktable.";
            errorManager += selectedDevice;
            RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                std::vector<std::string>({ errorManager, "The ranktable path configured "
                "in the training can be found in the plogs." }));
            HCCL_ERROR("%s", errorManager.c_str());
            return HCCL_E_PARA;
        }
        HCCL_DEBUG("%s", selectedDevice.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::IsAllRankSamePlane(bool &isAllRankSamePlane)
{
    // 只有1个rank，不考虑
    if (rankList_.size() == 1) {
        isAllRankSamePlane = true;
        return HCCL_SUCCESS;
    }

    auto isSameDevId = [&]()-> bool {
        for (size_t index = 0; index < rankList_.size() - 1; ++index) {
            if (rankList_[index].devicePhyId != rankList_[index + 1].devicePhyId) {
                return false;
            }
        }
        return true;
    };

    isAllRankSamePlane = isSameDevId();
    HCCL_DEBUG("[TopoInfoParse]curr comm isAllRankSamePlane[%d]", isAllRankSamePlane);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::IsSingleMeshAggregation(bool &isSingleMeshAggregation)
{
    if (deviceNumPerServer_ == deviceNum_) {
        // rank间都是hccs链接，则表明在同一个mesh cube中
        CHK_RET(IsAllRankConnectedWithHCCS(isSingleMeshAggregation));
    } else {
        isSingleMeshAggregation = false;
    }
    HCCL_DEBUG("[TopoInfoParse]curr comm isSingleMeshAggregation[%d]", isSingleMeshAggregation);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::IsAllRankConnectedWithHCCS(bool &isAllRankConnectedWithHCCS)
{
    for (u32 i = 0; i < rankList_.size(); i++) {
        for (u32 j = i + 1; j < rankList_.size(); j++) {
            LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
            if (rankList_[i].devicePhyId != HOST_DEVICE_ID && rankList_[j].devicePhyId != HOST_DEVICE_ID) {
                CHK_RET(hrtGetPairDeviceLinkType(rankList_[i].devicePhyId, rankList_[j].devicePhyId, linkType));
            }
            if (linkType == LinkTypeInServer::PXI_TYPE) {
                isAllRankConnectedWithHCCS = false;
                return HCCL_SUCCESS;
            }
        }
    }
    isAllRankConnectedWithHCCS = true;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoParse::GetDeviceNumInPerMeshAggregation(u32 devicePhyId, u32 &perAggregationNum)
{
    // 每个rank本身加上和他通过hccs互联的rank表示当前server Aggregation中的rank数量
    perAggregationNum = 1;
    for (u32 i = 0; i < rankList_.size(); i++) {
        if (serverId_ == rankList_[i].serverId && rankList_[i].devicePhyId != static_cast<s32>(devicePhyId)) {
            LinkTypeInServer linkType;
            CHK_RET(hrtGetPairDeviceLinkType(devicePhyId, rankList_[i].devicePhyId, linkType));
            if (linkType == LinkTypeInServer::HCCS_TYPE) {
                perAggregationNum++;
            }
        }
    }
    HCCL_DEBUG("[TopoInfoParse][GetDeviceNumInPerMeshAggregation]serverId[%s] devicePhyId[%u] perAggregationNum[%u]",
        serverId_.c_str(), devicePhyId, perAggregationNum);

    return HCCL_SUCCESS;
}
}
