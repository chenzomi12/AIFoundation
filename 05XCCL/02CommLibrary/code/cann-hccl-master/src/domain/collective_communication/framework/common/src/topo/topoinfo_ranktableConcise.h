/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_RANKTABLECONCISE_H
#define TOPOINFO_RANKTABLECONCISE_H

#include "topoinfo_ranktableParser_pub.h"
#include "base.h"
#include "hccl_comm_pub.h"
#include "hccl_types.h"

namespace hccl {
class TopoinfoRanktableConcise : public TopoInfoRanktableParser {
public:
    explicit TopoinfoRanktableConcise(const std::string &rankTableM, const std::string &identify);
    ~TopoinfoRanktableConcise() override;

    HcclResult Init() override;
    HcclResult GetClusterInfo(RankTable_t &clusterInfo) override;
    HcclResult GetSelfClusterInfo(HcclCommParams &params);
    HcclResult GetClusterInfo(hccl::HcclCommParams &params,
        hccl::RankTable_t &rankTable) override;
protected:
private:
    // 所有集群信息
    TopoinfoRanktableConcise(const TopoinfoRanktableConcise&);
    TopoinfoRanktableConcise& operator=(const TopoinfoRanktableConcise&);
    HcclResult ParserClusterInfo(hccl::HcclCommParams &params, hccl::RankTable_t &rankTable);
    HcclResult SplitString(const std::string& str, const std::string& strC, std::vector<std::string>& strVector) const;
    HcclResult GetRanktableInfo(RankTable_t &clusterInfo);
    HcclResult GetServerList(const nlohmann::json &obj, RankTable_t &clusterInfo);
    HcclResult GetSingleServer(const nlohmann::json &serverListObj, u32 objIndex, RankTable_t &clusterInfo);
    HcclResult GetDeviceList(const nlohmann::json &serverListObj, u32 objIndex, RankTable_t &clusterInfo,
        std::string &serverId, u32 &serverIdx, HcclIpAddress &hostIp);
    HcclResult GetSingleDevice(const nlohmann::json &deviceListObj, u32 objIndex,
        RankTable_t &clusterInfo, std::string &serverId, u32 &serverIdx, HcclIpAddress &hostIp);
    HcclResult GetSingleDeviceIp(const nlohmann::json &deviceListObj, u32 objIndex,
        RankTable_t &clusterInfo, RankInfo_t &rankinfo);
    HcclResult GetSingleSuperDeviceId(const nlohmann::json &deviceListObj, u32 objIndex,
        RankTable_t &clusterInfo, RankInfo_t &rankinfo);
    HcclResult CheckNicDeployConsistence(RankTable_t &clusterInfo) const;

    // 解析超节点信息
    HcclResult GetSuperPodList(const nlohmann::json &obj, RankTable_t &clusterInfo);
    HcclResult GetSingleSuperPod(const nlohmann::json &superPodList, u32 objIndex, RankTable_t &clusterInfo);
    HcclResult GetSuperPodServerList(const nlohmann::json &superPodList, u32 objIndex, RankTable_t &clusterInfo,
         std::string superPodId);
    HcclResult GetSingleSuperPodSever(const nlohmann::json &superPodServerList, u32 objIndex,
        RankTable_t &clusterInfo, std::string superPodId);
    HcclResult CheckSuperPodInfo(RankTable_t &clusterInfo) const;
};
}  // namespace hccl
#endif  // TOPOINFO_RANKTABLEPARSER_VER1_H
