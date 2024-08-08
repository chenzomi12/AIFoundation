/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_roletableParser.h"

#include "externalinput_pub.h"
#include "hccl_comm_pub.h"

using namespace std;
using namespace hccl;

TopoinfoRoletable::TopoinfoRoletable(const std::string &rankTableM)
    : TopoInfoRanktableParser(rankTableM, "0")
{
}

TopoinfoRoletable::~TopoinfoRoletable()
{
}

HcclResult TopoinfoRoletable::GetSingleNode(const nlohmann::json &NodeListObj, u32 objIndex,
    std::vector<RoleTableNodeInfo> &nodes)
{
    u32 id;
    std::string ip;
    HcclIpAddress ipAddr;
    u32 port;

    if (GetJsonArrayMemberProperty(NodeListObj, objIndex, NODE_ID, id) == HCCL_E_NOT_FOUND) {
        HCCL_WARNING("[Parser][RoleTable]node id is not found!");
        id = INVALID_UINT;
    }

    CHK_RET(GetJsonArrayMemberProperty(NodeListObj, objIndex, NODE_IP, ip));
    CHK_RET(GetJsonArrayMemberProperty(NodeListObj, objIndex, NODE_PORT, port));
    CHK_RET(ConvertIpAddress(ip, ipAddr));

    RoleTableNodeInfo roleTableNodeInfo;
    roleTableNodeInfo.id = id;
    roleTableNodeInfo.ipAddr = ipAddr;
    roleTableNodeInfo.port = port;

    nodes.push_back(roleTableNodeInfo);
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRoletable::GetServersInfo(std::vector<RoleTableNodeInfo> &servers)
{
    nlohmann::json serverList;
    CHK_RET(GetJsonProperty(fileContent_, SERVER_LIST, serverList));
    // 获得single server信息
    for (u32 index = 0; index < serverList.size(); index++) {
        CHK_RET(GetSingleNode(serverList, index, servers));
    }
    HCCL_INFO("get servers info success.");
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRoletable::GetClientsInfo(std::vector<RoleTableNodeInfo> &clients)
{
    nlohmann::json clientList;
    CHK_RET(GetJsonProperty(fileContent_, CLIENT_LIST, clientList));
    // 获得single client信息
    for (u32 index = 0; index < clientList.size(); index++) {
        CHK_RET(GetSingleNode(clientList, index, clients));
    }
    HCCL_INFO("get clients info success.");
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRoletable::ParserRoleTable(RoleTableInfo &roleTableInfo)
{
    CHK_RET(LoadConfigString(rankTableFile_));
    HCCL_INFO("ParserRoleTable [%s].", rankTableFile_.c_str());
    CHK_RET(GetServersInfo(roleTableInfo.servers));
    CHK_RET(GetClientsInfo(roleTableInfo.clients));

    return HCCL_SUCCESS;
}
