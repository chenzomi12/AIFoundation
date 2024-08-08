/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_exchange_base.h"
#include <thread>
#include <iostream>
#include <fstream>
#include "externalinput_pub.h"
#include "mem_host_pub.h"
#include "json_utils.h"

namespace hccl {
TopoInfoExchangeBase::TopoInfoExchangeBase()
    : currentStep_(0)
{
}

TopoInfoExchangeBase::~TopoInfoExchangeBase()
{
}

HcclResult TopoInfoExchangeBase::SaveClusterInfo(const RankTable_t &clusterInfo)
{
    nlohmann::json basicJson;
    HcclResult ret = Struct2Json(clusterInfo, basicJson);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_WARNING("cluster info to json failed ,ret[%d]", ret), HCCL_E_INTERNAL);
    basicJson[PROP_STEP] = currentStep_;  // add step to verify.

    std::string buffer = basicJson.dump(2);
    HCCL_INFO("the rankinfo exchanged is %s", buffer.c_str());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::DisconnectSocket(std::shared_ptr<HcclSocket> socket) const
{
    if (socket) {
        socket->Close();
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::SendClusterInfoMsg(std::shared_ptr<HcclSocket> socket, const RankTable_t &clusterInfo,
                                                    const std::string buffer, const u32 msgLen)
{
    HcclResult ret = socket->Send(&msgLen, sizeof(msgLen));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Send][ClusterInfoMsg]errNo[0x%016llx] ra send msg length failed! "\
        "msgLen[%u], ret[%u]", HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), msgLen, ret), ret);

    ret = socket->Send(buffer.c_str(), msgLen);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Send][ClusterInfoMsg]errNo[0x%016llx] ra send failed! size[%u], ret[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), msgLen, ret), ret);

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::RecvClusterInfoMsg(std::shared_ptr<HcclSocket> socket, RankTable_t &clusterInfo)
{
    const u32 recvBufferLimit = 10 * 1024 * 1024; // 10 * 1024 * 1024 = 10MB
    u32 msgLen = 0;
    HcclResult ret = socket->Recv(reinterpret_cast<char *>(&msgLen), sizeof(msgLen));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Recv][ClusterInfoMsg]receive msg length from fdhandle failed, ret[%d]", ret), HCCL_E_INTERNAL);
    CHK_PRT_RET(((msgLen == 0) || (msgLen > recvBufferLimit)), HCCL_ERROR("[Recv][ClusterInfoMsg]receive msg length "\
        "from fdhandle failed, msg length is beyond [1 ~ %u].", recvBufferLimit), HCCL_E_INTERNAL);

    u32 recvBufferLen = msgLen + 1;
    HostMem recvMsg = HostMem::alloc(recvBufferLen);
    CHK_PTR_NULL(recvMsg.ptr());
    char *recvMsgBuf = static_cast<char *>(recvMsg.ptr());

    s32 sRet = memset_s(recvMsgBuf, recvBufferLen, 0, recvBufferLen);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Recv][ClusterInfoMsg]sockBuff memset falied"), HCCL_E_MEMORY);
    ret = socket->Recv(recvMsgBuf, msgLen);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Recv][ClusterInfoMsg]receive from fdhandle failed ,ret[%d]",
        ret), HCCL_E_INTERNAL);
    nlohmann::json jClusterJson;
    CHK_RET(parseJsonBuff(recvMsgBuf, recvBufferLen, jClusterJson));

    // Verify json basic info
    u32 step;
    CHK_RET(JsonUtils::GetJsonProperty(jClusterJson, PROP_STEP, step));

    CHK_PRT_RET(step != currentStep_, HCCL_ERROR("[Recv][ClusterInfoMsg]RecvClusterInfo step failed "\
        "step[%u] vs currentStep_[%u]", step, currentStep_), HCCL_E_INTERNAL);

    ret = Json2Struct(jClusterJson, clusterInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Recv][ClusterInfoMsg]step[%u] json to struct failed!", currentStep_),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::BlockReceive(std::shared_ptr<HcclSocket> socket, char *buff, u32 size) const
{
    CHK_PTR_NULL(buff);
    CHK_RET(socket->Recv(buff, size));
    return HCCL_SUCCESS;
}


HcclResult TopoInfoExchangeBase::parseJsonBuff(const char buff[], u32 buffLen, nlohmann::json& buffJson) const
{
    u32 len = strnlen(buff, buffLen);
    CHK_PRT_RET((len > buffLen || len == 0), HCCL_ERROR("[Parse][JsonBuff]buff len invalid, buff len[%u], msgLen[%u]",
        len, buffLen), HCCL_E_INTERNAL);

    CHK_RET(JsonUtils::ParseInformation(buffJson, buff));
    u32 step;
    CHK_RET(JsonUtils::GetJsonProperty(buffJson, PROP_STEP, step));
    if (step != currentStep_) {
        HCCL_ERROR("[Parse][JsonBuff]errNo[0x%016llx] received step[%u] is invalid , expect step is %u", \
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), step, currentStep_);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}


HcclResult TopoInfoExchangeBase::Json2Struct(const nlohmann::json& jClusterJson, RankTable_t &clusterInfo) const
{
    clusterInfo.nicDeploy = jClusterJson[PROP_DEPLOY_MODE];
    clusterInfo.deviceNum = jClusterJson[PROP_DEV_NUM];
    clusterInfo.serverNum = jClusterJson[PROP_SRV_NUM];
    clusterInfo.superPodNum = jClusterJson[PROP_SUPER_POD_NUM];
    clusterInfo.rankNum = jClusterJson[PROP_RANK_NUM];
    for (auto& rankInfoJson : jClusterJson[PROP_RANK_LIST]) {
        RankInfo_t rankInfo;
        rankInfo.rankId = rankInfoJson[PROP_RANK_ID];
        rankInfo.serverId = rankInfoJson[PROP_SERVER_ID];
        CHK_RET(rankInfo.hostIp.SetReadableAddress(rankInfoJson[PROP_HOST_IP]));
        rankInfo.deviceInfo.devicePhyId = rankInfoJson[PROP_DEV_INFO][PROP_DEV_ID];
        for (auto& devIp : rankInfoJson[PROP_DEV_INFO][PROP_DEV_IP]) {
            std::string ipStr = devIp;
            rankInfo.deviceInfo.deviceIp.emplace_back(ipStr);
        }

        rankInfo.superPodId = rankInfoJson[PROP_SUPER_POD_ID];
        rankInfo.superDeviceId = rankInfoJson[PROP_SUPER_DEVICE_ID];

        /* Optional: for second communication stage */
        if (rankInfoJson.find(PROP_TRANS_INFO) != rankInfoJson.end()) {
            for (auto& transInfoJson : rankInfoJson[PROP_TRANS_INFO]) {
                TransportInfo_t transportInfo;
                transportInfo.dstRankId = transInfoJson[PROP_DEST_RANK];
                transportInfo.transportType = transInfoJson[PROP_TRANS_TYPE];
                rankInfo.transportInfo.push_back(transportInfo);
            }
        }
        clusterInfo.rankList.push_back(rankInfo);
    }
    for (auto& serverInfoJson : jClusterJson[PROP_SERVER_LIST]) {
        ServerInfo_t serverInfo;
        serverInfo.serverId = serverInfoJson[PROP_SERVER_ID];
        for (auto& networkInfoJson : serverInfoJson[PROP_NETWORK_INFO_LIST]) {
            NetworkInfo_t networkInfo;
            networkInfo.ethName = networkInfoJson[PROP_NETWORK_ETHNAME];
            CHK_RET(networkInfo.ipAddr.SetReadableAddress(networkInfoJson[PROP_NETWORK_IPADDR]));
            networkInfo.networkPort = networkInfoJson[PROP_NETWORK_NETWORKPORT];
            CHK_RET(networkInfo.refIp.SetReadableAddress(networkInfoJson[PROP_NETWORK_REFIP]));
            networkInfo.planeID = networkInfoJson[PROP_NETWORK_PLANEID];
            serverInfo.networkInfo.push_back(networkInfo);
        }
        clusterInfo.serverList.push_back(serverInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::Struct2Json(const RankTable_t &clusterInfo, nlohmann::json& ClusterJson)
{
    nlohmann::json rankListJson;
    nlohmann::json serverListJson;

    TransformRankListToJson(clusterInfo, rankListJson);
    for (auto& serverInfo : clusterInfo.serverList) {
        nlohmann::json serverJson;
        serverJson[PROP_SERVER_ID] = serverInfo.serverId;
        nlohmann::json networkInfoListJson;
        for (auto& networkInfo : serverInfo.networkInfo) {
            nlohmann::json networkInfoJson;
            networkInfoJson[PROP_NETWORK_ETHNAME] = networkInfo.ethName;
            networkInfoJson[PROP_NETWORK_IPADDR] = std::string(networkInfo.ipAddr.GetReadableIP());
            networkInfoJson[PROP_NETWORK_NETWORKPORT] = networkInfo.networkPort;
            networkInfoJson[PROP_NETWORK_REFIP] = std::string(networkInfo.refIp.GetReadableIP());
            networkInfoJson[PROP_NETWORK_PLANEID] = networkInfo.planeID;
            networkInfoListJson.push_back(networkInfoJson);
        }
        serverJson[PROP_NETWORK_INFO_LIST] = networkInfoListJson;
        serverListJson.push_back(serverJson);
    }

    ClusterJson[PROP_RANK_NUM] = clusterInfo.rankNum;
    ClusterJson[PROP_DEV_NUM] = clusterInfo.deviceNum;
    ClusterJson[PROP_SRV_NUM] = clusterInfo.serverNum;
    ClusterJson[PROP_SUPER_POD_NUM] = clusterInfo.superPodNum;
    ClusterJson[PROP_DEPLOY_MODE] = clusterInfo.nicDeploy;
    ClusterJson[PROP_RANK_LIST] = rankListJson;
    ClusterJson[PROP_SERVER_LIST] = serverListJson;
    return HCCL_SUCCESS;
}
HcclResult TopoInfoExchangeBase::TransformRankListToJson(const RankTable_t &clusterInfo, nlohmann::json& rankListJson)
    const
{
    for (auto& rankInfo : clusterInfo.rankList) {
        nlohmann::json deviceIp;
        for (auto& devIp : rankInfo.deviceInfo.deviceIp) {
            deviceIp.push_back(std::string(devIp.GetReadableIP()));
        }
        nlohmann::json devInfoJson;
        devInfoJson[PROP_DEV_ID] = rankInfo.deviceInfo.devicePhyId;
        devInfoJson[PROP_DEV_IP] = deviceIp;
        nlohmann::json rankJson;
        rankJson[PROP_RANK_ID] = rankInfo.rankId;
        rankJson[PROP_SERVER_ID] = rankInfo.serverId;
        rankJson[PROP_HOST_IP] =
            (GetExternalInputHcclDeviceNicDisable()) ?
            std::string(rankInfo.hostIp.GetReadableIP()) : "0.0.0.0";
        rankJson[PROP_DEV_INFO] = devInfoJson;

        rankJson[PROP_SUPER_POD_ID] = rankInfo.superPodId;
        rankJson[PROP_SUPER_DEVICE_ID] = rankInfo.superDeviceId;

        /* Optional: for second communication stage */
        nlohmann::json transInfosJson;
        for (auto& transInfo : rankInfo.transportInfo) {
            nlohmann::json transInfoJson;
            transInfoJson[PROP_TRANS_TYPE] = transInfo.transportType;
            transInfoJson[PROP_DEST_RANK]  = transInfo.dstRankId;
            transInfosJson.push_back(transInfoJson);
        }
        if (!transInfosJson.empty()) {
            rankJson[PROP_TRANS_INFO] = transInfosJson;
        }
        rankListJson.push_back(rankJson);
    }
    return HCCL_SUCCESS;
}
}  // namespace hccl
