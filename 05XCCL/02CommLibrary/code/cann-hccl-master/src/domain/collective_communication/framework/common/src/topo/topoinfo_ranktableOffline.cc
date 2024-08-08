/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableOffline.h"

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
#include "sal_pub.h"
#include "config.h"

using namespace hccl;

TopoinfoRanktableOffline::TopoinfoRanktableOffline(const std::string &rankTableM)
    : TopoInfoRanktableParser(rankTableM, "0"), deviceNumPerServer_(0), curServerDeviceNum_(0)
{
}

TopoinfoRanktableOffline::~TopoinfoRanktableOffline()
{
}

HcclResult TopoinfoRanktableOffline::GetSingleRank(const nlohmann::json &ranksObj, u32 objIndex,
    RankTable_t &clusterInfo, u32 &serverIdx, std::string &nodeId)
{
    RankInfo_t rankInfo;

    // 获取rank_id
    std::string rankInfoStr;
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "rank_id", rankInfoStr));
    if (SalStrToULong(rankInfoStr, HCCL_BASE_DECIMAL, rankInfo.rankId) != HCCL_SUCCESS) {
        HCCL_ERROR("[Get][SingleRank]errNo[0x%016llx] rankid[%s] is invalid",
            HCOM_ERROR_CODE(HCCL_E_PARA), rankInfoStr.c_str());
        return HCCL_E_PARA;
    }
    // 获取item_id, -1表示 host
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "item_id", rankInfoStr));
    CHK_RET(SalStrToInt(rankInfoStr, HCCL_BASE_DECIMAL, rankInfo.itemId));
    // 为了适配GetDevNum函数
    if (rankInfo.itemId == HOST_DEVICE_ID) {
        rankInfo.deviceInfo.devicePhyId = HOST_DEVICE_ID;
    } else {
        curServerDeviceNum_++;
    }
    // 获取rank_ip
    if (GetJsonArrayMemberProperty(ranksObj, objIndex, "rank_ip", rankInfoStr) == HCCL_E_NOT_FOUND) {
        rankInfo.hostIp = HcclIpAddress();
    } else {
        CHK_RET(ConvertIpAddress(rankInfoStr, rankInfo.hostIp));
    }

    rankInfo.serverId = nodeId;
    rankInfo.localRank = objIndex;
    rankInfo.serverIdx = serverIdx;
    clusterInfo.rankList.push_back(rankInfo);
    HCCL_INFO("[%s.json]->rankId[%u], serverIdx[%u]", fileName_.c_str(),
        rankInfo.rankId, rankInfo.serverIdx);

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableOffline::GetSingleNode(const nlohmann::json &NodeListObj, u32 objIndex,
    RankTable_t &clusterInfo, u32 &serverIdx)
{
    // 获取信息
    nlohmann::json Ranks;
    std::string nodeId;

    // 处理ranklist
    // 获取node_id
    CHK_RET(GetJsonArrayMemberProperty(NodeListObj, objIndex, "node_id", nodeId));
    HCCL_DEBUG("Get Node[%u]: serverIdx:[%u] nodeId[%s]", objIndex, serverIdx, nodeId.c_str());
    CHK_RET(GetJsonArrayMemberProperty(NodeListObj, objIndex, "rank_list", Ranks));

    HCCL_INFO("[%s.json] -> rank_list: size:%zu", fileName_.c_str(), Ranks.size());
    CHK_PRT_RET(Ranks.size() == 0, HCCL_ERROR("[Get][Ranks]Ranks size is zero"), HCCL_E_PARA);

    // 获取单rank信息
    for (u32 index = 0; index < Ranks.size(); index++) {
        CHK_RET(GetSingleRank(Ranks, index, clusterInfo, serverIdx, nodeId));
    }

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableOffline::GetRanktableInfo(RankTable_t &clusterInfo)
{
    // 清空list
    clusterInfo.rankList.clear();

    // 获取信息
    nlohmann::json node_list;

    CHK_RET(GetJsonProperty(fileContent_, "node_list", node_list));
    HCCL_DEBUG("[rankTableJson] -> nodeListSize: [%zu]", node_list.size());

    // 保存serverNum
    clusterInfo.serverNum = node_list.size();
    if (clusterInfo.serverNum == 0) {
        HCCL_ERROR("[Get][RanktableInfo]errNo[0x%016llx] node num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    // 获得single node信息
    for (u32 index = 0; index < clusterInfo.serverNum; index++) {
        curServerDeviceNum_ = 0;
        CHK_RET(GetSingleNode(node_list, index, clusterInfo, index));
        if (index == 0) {
            deviceNumPerServer_ = curServerDeviceNum_;
        } else if (curServerDeviceNum_ != deviceNumPerServer_) {
            HCCL_ERROR("[GetRanktableInfo] device num is diff.first node device num:[%d] cur node device num:[%d]",
                curServerDeviceNum_, deviceNumPerServer_);
            return HCCL_E_PARA;
        }
    }

    // 保存deviceNum和rankNum
    CHK_RET(GetDevNum(clusterInfo.rankList, clusterInfo.deviceNum));
    clusterInfo.rankNum = clusterInfo.rankList.size();

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableOffline::ParserClusterInfo(hccl::RankTable_t &rankTable)
{
    // 获取ranktable info信息
    CHK_RET(GetRanktableInfo(rankTable));
    std::sort(rankTable.rankList.begin(), rankTable.rankList.end(),
        [&](const RankInfo_t &a, const RankInfo_t &b) -> bool {return a.rankId < b.rankId;});

    // 校验ranktable是否正确
    CHK_RET(CheckRankListInfo(rankTable.rankList));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableOffline::Init()
{
    // 根据rankTable类型标记执行读文件或读字符串,将内容保存在json对象fileContent_中
    CHK_RET(LoadRankTableString(rankTableFile_));
    // 解析rankTable
    CHK_RET(ParserClusterInfo(rankTable_));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableOffline::GetClusterInfo(RankTable_t &clusterInfo)
{
    // 获取rankInfo
    clusterInfo.deviceNum = rankTable_.deviceNum;
    clusterInfo.serverNum = rankTable_.serverNum;
    clusterInfo.rankNum = rankTable_.rankNum;
    clusterInfo.rankList = rankTable_.rankList;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableOffline::GetDeviceNumPerServer(s32 &deviceNum)
{
    deviceNum = deviceNumPerServer_;
    return HCCL_SUCCESS;
}