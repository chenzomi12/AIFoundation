/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_CHECK_H
#define TOPOINFO_CHECK_H
#include <string>
#include <vector>
#include <set>
#include "log.h"
#include "common.h"
#include "topoinfo_struct.h"

namespace hccl {

// TopoInfoCheck、Parse，以通信域为粒度
class TopoInfoParse {
public:
    TopoInfoParse();
    ~TopoInfoParse();
    HcclResult Init(const RankTable_t &rankTable, const std::string &serverId, const u32 deviceNumPerServer);
    HcclResult Init(const std::vector<RankInfo> &rankList, const std::string &serverId, const u32 deviceNumPerServer);
    // worldgroup
    HcclResult ParseAndCheck(std::vector<u32> &nicIdx);
    // subgroup
    HcclResult Check();
    // 获取当前server rank间的链接信息
    HcclResult GetServerInnerLinkInfo(std::unordered_map<u32, u32> &pairLinkCounter,
        std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> &pairLinkInfo);
    // 获取当前通信域中rank所在网络平面是否都一致
    /***
     * 获取当前通信域中rank所在网络平面是否都一致
     * 1、910A、910B场景：每个server device选取同一平面的
     * 2、310p soc场景：2pg环境上只有一张网卡，其中2个device共用这个网卡
    ***/
    HcclResult IsAllRankSamePlane(bool &isAllRankSamePlane);
    // 获取当前通信域的rank是否都在同一个mesh cube中
    HcclResult IsSingleMeshAggregation(bool &isSingleMeshAggregation);
    // 获取每个mesh cube中的device num
    HcclResult GetDeviceNumInPerMeshAggregation(u32 devicePhyId, u32 &perAggregationNum);

private:
    // 获取当前server的rank信息
    HcclResult TransformRankInfoByServerId(std::vector<hccl::RankInfo> &serverInnerInfo);
    // server间device选取是否对称校验
    HcclResult CheckInterServerDeviceId();
    // 校验server内4p场景下deivce选取是否合法，2p与标卡场景重合
    HcclResult CheckServerInnerRankInfo();
    // 校验Nic选取是否正确，并在host网卡场景下进行nic填充
    HcclResult CheckRankTableNicInfo(std::vector<u32> &nicIdx);
    // 校验Nic选取合法性，并做去重处理
    HcclResult CheckAndAssignNicInfo(std::vector<u32> &nicIdx);
    // 判断当前ranklist中所有rank是否都是hccs直连
    HcclResult IsAllRankConnectedWithHCCS(bool &isAllRankConnectedWithHCCS);
    std::vector<RankInfo> rankList_;
    std::string serverId_;  // 当前rank的serverID
    DevType deviceType_;    // 当前rank的deviceType
    u32 deviceNum_;         // 当前集群的device num
    u32 serverNum_;         // 当前集群的server num
    u32 superPodNum_ = 0;         // 当前集群的superPod num
    u32 deviceNumPerServer_;    // 当前server的device num
    // 网卡挂载位置 0:host 1:device
    NICDeployment nicDeploy_;
    bool multiServerDiffDeviceNumMode_ = false;
};
}

#endif /* TOPOINFO_CHECK_H */
