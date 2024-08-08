/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_EXCHANGE_SERVER_H
#define TOPOINFO_EXCHANGE_SERVER_H

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <thread>
#include <vector>
#include "topoinfo_struct.h"
#include "topoinfo_exchange_base.h"
#include "comm.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"

namespace hccl {
class TopoInfoExchangeServer : public TopoInfoExchangeBase {
public:
    explicit TopoInfoExchangeServer(HcclIpAddress &hostIP, u32 hostPort, const std::vector<HcclIpAddress> whitelist,
        HcclNetDevCtx netDevCtx, const std::shared_ptr<HcclSocket> listenSocket, const std::string &identifier);
    ~TopoInfoExchangeServer() override;
    HcclResult Setup();
    HcclResult SetupByMasterInfo();
    HcclResult Teardown();
    HcclResult GetConnections(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets);

private:
    HcclResult Connect(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets);
    HcclResult GetConnection(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets);
    HcclResult Disconnect(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets);
    HcclResult DeleteSocketWhiteList(u32 port, const std::vector<HcclIpAddress> &whitelist);
    HcclResult StopNetwork(const std::vector<HcclIpAddress> &whitelist,
        HcclIpAddress &hostIP, u32 hostPort);
    HcclResult StopSocketListen(const std::vector<HcclIpAddress> &whitelist,
        HcclIpAddress &hostIP, u32 hostPort);
    HcclResult GetRanksBasicInfo(
        const std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets, RankTable_t &rankTable);
    HcclResult GetRanksTransInfo(
        const std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets, RankTable_t &rankTable);
    HcclResult GetRankBasicInfo(std::shared_ptr<HcclSocket> socket, RankTable_t &rankTable);
    HcclResult GetCommonTopoInfo(RankTable_t &rankTable, const RankTable_t &orginRankTable) const;
    HcclResult SortRankList(RankTable_t &rankTable) const;
    HcclResult RecvRemoteAgentID(std::shared_ptr<HcclSocket> socket, std::string &agentID);
    HcclResult RecvRemoteRankNum(std::shared_ptr<HcclSocket> socket, u32 &remoteRankNum);
    HcclResult VerifyRemoteRankNum(u32 &previousRankNum, u32 remoteRankNum) const;
    HcclResult SendIndentify(std::shared_ptr<HcclSocket> socket, u32 indentify) const;
    HcclResult DisplayConnectionedRank(const std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets);
    bool DoServerIdExist(const RankTable_t &rankTable, const std::string &serverId) const;
    HcclResult GetRemoteFdAndRankSize(std::shared_ptr<HcclSocket> &socket,
        std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets, u32 &rankSize);
    HcclIpAddress hostIP_;
    u32 hostPort_;
    SocketHandle socketHandle_;
    std::vector<HcclIpAddress> whitelist_;
    HcclNetDevCtx netDevCtx_{nullptr};
    std::shared_ptr<HcclSocket> listenSocket_;
    friend class TopoInfoExchangeDispather;
    std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets_;
    std::mutex lock_;
    std::string identifier_;
};
}  // namespace hccl

#endif /* TOPOINFO_EXCHANGE_SERVER_H */
