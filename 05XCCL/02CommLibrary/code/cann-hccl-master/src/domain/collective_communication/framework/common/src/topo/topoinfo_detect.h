/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_DETECT_H
#define TOPOINFO_DETECT_H

#include <thread>
#include <vector>
#include "comm.h"
#include "hccl_comm_pub.h"
#include "topoinfo_exchange_server.h"
#include "topoinfo_exchange_agent.h"
#include "externalinput_pub.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"

namespace hccl {
class TopoInfoDetect {
public:
    explicit TopoInfoDetect();
    ~TopoInfoDetect();
    HcclResult SetupAgent(u32 rankSize, u32 myrank, const HcclRootHandle &rootInfo);
    HcclResult SetupAgentByMasterInfo(HcclIpAddress &localHostIp, const HcclRootHandle &rootInfo);
    HcclResult SetupServer(HcclRootHandle &rootInfo);
    HcclResult SetupServerByMasterInfo(const HcclIpAddress &masterIP, u32 masterPort, const HcclRootHandle &rootInfo);
    HcclResult Teardown();
    HcclResult WaitComplete(const HcclRootHandle &rootInfo);
    HcclResult GetCluterInfo(RankTable_t &clusterInfo);
    HcclResult GetLocalRankInfo(HcclBasicRankInfo &rankInfo);
    HcclResult GetRankId(u32 &rankId);
    HcclResult TransformRankTableStr(const RankTable_t &clusterInfo, std::string &ranktableStr);
    HcclResult GetAgentConnection(std::shared_ptr<HcclSocket> &connectSocket);
    HcclResult GetServerConnections(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets);
    HcclResult GenerateRootInfo(const HcclIpAddress &hostIP, u32 hostPort, u32 devicePhysicID, HcclRootHandle &rootInfo);

protected:
private:
    HcclResult TeardownAgent();
    HcclResult TeardownServer();
    HcclResult Struct2JsonRankTable(const RankTable_t &clusterInfo, nlohmann::json &ClusterJson);
    HcclResult GetRootHostIP(const std::vector<HcclIpAddress> &whitelist, HcclIpAddress &ip, u32 devPhyId);
    HcclResult StartNetwork(HcclIpAddress &hostIP, bool bInitDevNic);
    HcclResult StopNetwork(HcclIpAddress &hostIP, bool bInitDevNic);
    HcclResult StartRootNetwork(const std::vector<HcclIpAddress> &whitelist, const HcclIpAddress &hostIP, u32 usePort);
    HcclResult AddSocketWhiteList(u32 port,
        const std::vector<HcclIpAddress> &whitelist) const;
    HcclResult GenerateLocalRankInfo(u32 rankSize, u32 rankID, HcclBasicRankInfo &localRankInfo);
    HcclResult ReadHostSocketWhitelist(std::vector<HcclIpAddress> &whitelist) const;
    HcclResult GetAllHostIfInfos(std::vector<std::pair<std::string, HcclIpAddress>> &ifInfos, u32 devPhyId) const;
    HcclResult GetAllValidHostIfInfos(const std::vector<HcclIpAddress> &whitelist,
        std::vector<std::pair<std::string, HcclIpAddress>> &ifInfos, u32 devPhyId);
    HcclResult TransformDeviceList(const RankTable_t &clusterInfo, std::vector<RankInfo_t> &tmpRankList,
        nlohmann::json &perServerJson, u32 serverIndex);
    HcclResult TransformSuperPodList(const std::vector<RankInfo_t> &rankInfo, nlohmann::json &superPodListJson) const;
    void SetupTopoExchangeServer(s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort,
        std::vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket,
        bool isMasterInfo = false);
    HcclResult WaitTopoExchangeServerCompelte(u32 idx) const;
    void SetBootstrapHostIP(HcclIpAddress &ip) const;
    HcclIpAddress GetBootstrapHostIP() const;
    s32 deviceLogicID_;
    HcclBasicRankInfo localRankInfo_;
    RankTable_t clusterTopoInfo_;
    u32 identifierNum_;
    static std::map<u32, volatile u32> topoExchangeServerStatus_;
    static HcclIpAddress bootstrapHostIP_;
    HcclNetDevCtx serverPortCtx_{nullptr};
    HcclNetDevCtx agentPortCtx_{nullptr};
    HcclNetDevCtx devNicCtx_{nullptr};
    u32 devicePhysicID_{INVALID_UINT};
    std::shared_ptr<HcclSocket> listenSocket_{nullptr};
    HcclRootHandle rootInfo_;
    std::shared_ptr<hccl::TopoInfoExchangeAgent> pTopoExchangeAgent_{nullptr};
    std::shared_ptr<TopoInfoExchangeServer> pTopoExchangeServer_{nullptr};
    std::unique_ptr<std::thread> exchangeServerThreadPtr_{nullptr};
};
}  // namespace hccl
#endif /* TOPOINFO_DETECT_H */