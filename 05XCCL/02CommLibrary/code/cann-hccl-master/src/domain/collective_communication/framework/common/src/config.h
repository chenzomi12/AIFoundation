/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>
#include "base.h"
#include "hccl_types.h"
#include "topoinfo_struct.h"
#include "hccl_comm_pub.h"

HcclResult CfgGetClusterInfo(const std::string &rankTableM, const std::string &identify, hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable, DevType deviceType = DevType::DEV_TYPE_310P1);
HcclResult CfgGetClusterInfoWithoutDev(const std::string &rankTableM, const std::string &identify,
    hccl::HcclCommParams &params, hccl::RankTable_t &rankTable);
HcclResult CheckRankTableConfigInfo(const std::vector<hccl::RankInfo_t> &rankList, u32 deviceNum, u32 serverNum);
HcclResult ShowRanktableConfigInfo(const bool cloudFlag, hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable);
HcclResult CheckRankId(const char *rankId);
HcclResult DisplayCloudRankTableInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable);
HcclResult DisplayRanktableInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable);
HcclResult DisplayRanktableInfo(const hccl::RankTable_t &rankTable);
HcclResult CheckRankListBaseInfo(u32 deviceNum, u32 serverNum);
HcclResult CheckRankIpFamily(const std::vector<hccl::RankInfo_t> &rankList);
HcclResult CheckRankListInfo(const std::vector<hccl::RankInfo_t> &rankList);
HcclResult CheckRankListInfo(const std::vector<hccl::RankInfo_t> &rankList, u32 deviceNum, u32 serverNum);
HcclResult CheckDeviceId(const std::vector<hccl::RankInfo_t> &rankList, u32 deviceNum, u32 serverNum);
HcclResult CheckGroupRankList(const std::vector<hccl::RankInfo_t> &rankList, u32 deviceNum, u32 serverNum);
HcclResult CheckDeviceNumValid(const std::vector<hccl::RankInfo_t> &rankList, u32 deviceNum,
                               u32 serverNum, std::string version = "");
HcclResult CheckPortValid(u32 port);
HcclResult CheckRoleAndRankConsistent(const hccl::RoleTableInfo &roleTableInfo, const hccl::HcclCommParams &params,
    const hccl::RankTable_t &rankTable);
HcclResult CfgGetRoleTableInfo(const std::string &rankTableM, hccl::RoleTableInfo &roleTableInfo);
HcclResult GetDevNum(const std::vector<hccl::RankInfo_t> &rankList, u32 &devNum);
HcclResult GetDevNum(const std::vector<hccl::RankInfo> &rankList, u32 &devNum);
HcclResult GetServerNum(const std::vector<hccl::RankInfo> &rankList, u32 &serverNum);
HcclResult GetSuperPodNum(const std::vector<hccl::RankInfo_t> &rankList, u32 &superPodNum);
HcclResult GetSuperPodNum(const std::vector<hccl::RankInfo> &rankList, u32 &superPodNum);
void SetRetryEnable(DevType deviceType, const u32 &superPodNum, const u32 &serverNum,
    const u32 &deviceNumPerAggregation, bool &retryEnable);
#endif  // CONFIG_H
