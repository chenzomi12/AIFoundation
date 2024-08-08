/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_EXCHANGE_AGENT_H
#define TOPOINFO_EXCHANGE_AGENT_H

#include <vector>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "topoinfo_struct.h"
#include "topoinfo_exchange_base.h"
#include "comm.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"

namespace hccl {
using HcclBasicRankInfo = struct HcclBasicRankInfoDef {
    HcclIpAddress hostIP;
    u32 rank{0};
    u32 rankSize{0};
    NICDeployment nicDeploy{NICDeployment::NIC_DEPLOYMENT_DEVICE};
    DevType deviceType{DevType::DEV_TYPE_910};
    s32 deviceLogicID{0};
    u32 devicePhysicID{0};
    std::vector<HcclIpAddress> deviceIP;
    u32 superDeviceId{INVALID_UINT}; // 超节点内device id，超节点内唯一
    std::string superPodId{""};     // 超节点标识
};

class TopoInfoExchangeAgent : public TopoInfoExchangeBase {
public:
    explicit TopoInfoExchangeAgent(HcclIpAddress &serverIp, u32 serverPort, std::string identifier,
        HcclNetDevCtx netDevCtx, HcclBasicRankInfo localRankInfo);
    ~TopoInfoExchangeAgent() override;
    HcclResult Setup();
    HcclResult SetupByMasterInfo();
    HcclResult Teardown();
    HcclResult GetClusterTopoInfo(RankTable_t &clusterInfo);
    HcclResult GetIdentifier(u32 &indentify);
    HcclResult GetConnection(std::shared_ptr<HcclSocket> &socket);

private:
    HcclResult DetectClusterTopoInfo(std::shared_ptr<HcclSocket> socket, RankTable_t &clusterTopoInfo);
    HcclResult Connect(HcclIpAddress &serverIp, u32 port, std::shared_ptr<HcclSocket> &socket);
    HcclResult GetConnection(HcclIpAddress &serverIp, u32 port,
        std::shared_ptr<HcclSocket> &socket);
    HcclResult Disconnect(std::shared_ptr<HcclSocket> &socket);
    HcclResult SendClusterInfo(std::shared_ptr<HcclSocket> socket, const RankTable_t &clusterInfo);
    HcclResult RecvClusterInfo(std::shared_ptr<HcclSocket> socket, RankTable_t &clusterInfo);
    HcclResult SetServerIdx(RankTable_t &clusterInfo) const;
    HcclResult GenerateLocalRankInfo(u32 rankSize, u32 rankID);
    void GenerateAgentID(HcclBasicRankInfo &localRankInfo, std::string &agentID);
    HcclResult ConstructRankTableMsg(RankTable_t &clusterInfo);
    HcclResult SetTransportInfo(RankTable_t &clusterInfo);
    HcclResult VerifyClusterInfo(const RankTable_t &clusterInfo);
    HcclResult VerifyClusterDeviceIP(const RankTable_t &clusterInfo);
    HcclResult VerifyClusterRankID(const RankTable_t &clusterInfo) const;
    HcclResult VerifyServerDevicePhysicID(const std::vector<RankInfo_t> &serverInfo) const;
    HcclResult VerifyClusterSuperPodInfo(const std::vector<RankInfo_t> &rankInfo) const;

    bool HasRepeatedIP(const std::vector<HcclIpAddress> &deviceAIP, const std::vector<HcclIpAddress> &deviceBIP) const;
    HcclResult DetectTransportType(const RankInfo_t &localRankInfo, const RankInfo_t &remoteRankInfo,
        TransportType &transportType) const;
    std::string Dec2Hex(s32 i, u32 width);
    HcclIpAddress serverIP_;
    u32 serverPort_;
    std::string identifier_;
    SocketHandle socketHandle_;
    HcclBasicRankInfo localRankInfo_;
    RankTable_t clusterTopoInfo_;
    HcclNetDevCtx netDevCtx_{nullptr};
    std::shared_ptr<HcclSocket> socket_;
};
}  // namespace hccl

#endif /* TOPOINFO_EXCHANGE_AGENT_H */
