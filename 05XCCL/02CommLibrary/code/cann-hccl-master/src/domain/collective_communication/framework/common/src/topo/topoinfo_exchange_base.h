/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_EXCHANGE_BASE_H
#define TOPOINFO_EXCHANGE_BASE_H

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <nlohmann/json.hpp>
#include "topoinfo_struct.h"
#include "comm.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"

namespace hccl {
constexpr u32 MAX_AGENT_BUF_SIZE = 128;
constexpr s32 TOPO_SERVERIP_OFFSET_OF_RANKID = 32;
constexpr int BIT_NUM_PER_BYTE = 8;

class TopoInfoExchangeBase {
public:
    explicit TopoInfoExchangeBase();
    virtual ~TopoInfoExchangeBase();
protected:
    HcclResult SaveClusterInfo(const RankTable_t &clusterInfo);
    HcclResult DisconnectSocket(std::shared_ptr<HcclSocket> socket) const;
    HcclResult BlockReceive(std::shared_ptr<HcclSocket> socket, char *buff, u32 size) const;
    HcclResult SendClusterInfoMsg(std::shared_ptr<HcclSocket> socket, const RankTable_t &clusterInfo,
                                    const std::string buffer, const u32 msgLen);
    HcclResult RecvClusterInfoMsg(std::shared_ptr<HcclSocket> socket, RankTable_t &clusterInfo);
    HcclResult parseJsonBuff(const char buff[], u32 buffLen, nlohmann::json& buffJson) const;
    HcclResult Json2Struct(const nlohmann::json& jClusterJson, RankTable_t &clusterInfo) const;
    HcclResult Struct2Json(const RankTable_t &clusterInfo, nlohmann::json& ClusterJson);
    HcclResult TransformRankListToJson(const RankTable_t &clusterInfo, nlohmann::json& rankListJson) const;
    u32 currentStep_; // topo detect 分为多个step， 用以校验server和agent的step是否一致。
    bool isByMasterInfo_ = false;
    u32 identifierNum_;
private:
    HcclResult GetCommonTopoInfo(RankTable_t &rankTable, const RankTable_t &orginRankTable);
    HcclResult SortRankList(RankTable_t &rankTable);
    friend class TopoInfoExchangeDispather;
};
}  // namespace hccl
#endif /* TOPOINFO_EXCHANGE_BASE_H */
