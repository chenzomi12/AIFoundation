/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_hccp_common.h"
#include "adapter_error_manager_pub.h"
#include "externalinput_pub.h"
#include "hccl_socket.h"
#include "hccl_socket_manager.h"
#include "sal_pub.h"

namespace hccl {
HcclSocketManager::HcclSocketManager(NICDeployment nicDeployment, s32 deviceLogicId, u32 devicePhyId, u32 userRank)
    : nicDeployment_(nicDeployment), deviceLogicId_(deviceLogicId), devicePhyId_(devicePhyId), userRank_(userRank),
      isSupCloseSockImmed_(true), wlistInfosMap_(), commSocketsMap_()
{
}

HcclSocketManager::~HcclSocketManager()
{
    DestroySockets();
}

std::map<PortInfo, std::shared_ptr<HcclSocket>> HcclSocketManager::serverSocketMap_;
std::map<PortInfo, Referenced> HcclSocketManager::serverSocketRefMap_;
std::mutex HcclSocketManager::serverMapMutex_;

HcclResult HcclSocketManager::Init()
{
    // isSupCloseSockImmed_ 标识是否支持“立即关闭Socket”, 在服务器间 RDMA 通讯时, 可能有数据在关闭时未完成发送
    // 当前 HcclSocketManager 会管理服务器间和服务器内的Socket, 以及HCCD 等多场景, 所以 GetIsSupSockBatchCloseImmed
    // 并不要求返回成功；以及为了兼容原场景, isSupCloseSockImmed_ 初始化默认值修改为 true
    if (devicePhyId_ != INVALID_UINT) {
        HcclResult ret = GetIsSupSockBatchCloseImmed(devicePhyId_, isSupCloseSockImmed_);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[Init]call GetIsSupSockBatchCloseImmed is failed.");
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclSocketManager::ServerInit(const HcclNetDevCtx netDevCtx, u32 port)
{
    HcclIpAddress localIp{0};
    CHK_RET(HcclNetDevGetLocalIp(netDevCtx, localIp));

    PortInfo portInfo(localIp, port);

    std::unique_lock<std::mutex> lock(serverMapMutex_);
    auto serverSocketInMap = serverSocketMap_.find(portInfo);
    if (serverSocketInMap != serverSocketMap_.end()) {
        auto &serverSocketRef = serverSocketRefMap_[portInfo];
        serverSocketRef.Ref();
        return HCCL_SUCCESS;
    }

    std::shared_ptr<HcclSocket> tempSocket;
    EXECEPTION_CATCH((tempSocket = std::make_shared<HcclSocket>(
        netDevCtx, port)), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(tempSocket);
    CHK_RET(tempSocket->Init());

    CHK_RET(tempSocket->Listen());

    HCCL_INFO("[Init][Server]ip[%s] port[%u]", localIp.GetReadableAddress(), port);
    serverSocketMap_.insert(std::make_pair(portInfo, tempSocket));

    Referenced ref;
    ref.Ref();
    serverSocketRefMap_.insert(std::make_pair(portInfo, ref));

    return HCCL_SUCCESS;
}

HcclResult HcclSocketManager::ServerDeInit(const HcclNetDevCtx netDevCtx, u32 port)
{
    HcclIpAddress localIp{0};
    CHK_RET(HcclNetDevGetLocalIp(netDevCtx, localIp));
    PortInfo portInfo(localIp, port);

    std::unique_lock<std::mutex> lock(serverMapMutex_);
    auto res = serverSocketMap_.find(portInfo);
    if (res == serverSocketMap_.end()) {
        return HCCL_SUCCESS;
    }

    auto &serverSocketRef = serverSocketRefMap_[portInfo];
    serverSocketRef.Unref();

    if (serverSocketRef.Count() == 0) {
        HCCL_INFO("[DeInit][Server]ip[%s] port[%u]", localIp.GetReadableAddress(), port);
        serverSocketMap_[portInfo]->DeInit();
        serverSocketMap_.erase(portInfo);
        serverSocketRefMap_.erase(portInfo);
    }

    return HCCL_SUCCESS;
}

// public
// 填加向本端创建链接的客户端"RANK+IP"白名单
HcclResult HcclSocketManager::AddWhiteList(const std::string &commTag,
    bool isInterLink, NicType socketType,
    const HcclIpAddress &localIp, const std::map<u32, HcclRankLinkInfo> &whiteListMap)
{
    if (whiteListMap.size() == 0) {
        HCCL_ERROR("[Add][WhiteList]client infos map or local Ip is empty.");
        return HCCL_E_PARA;
    }

    HcclResult ret;
    for (auto &res : serverSocketMap_) {
        if (res.second->GetLocalIp() == localIp) {
            std::vector<SocketWlistInfo> wlistInfosVec {};
            for (auto iter = whiteListMap.begin(); iter != whiteListMap.end(); iter++) {
                auto dstRankLinkInfo = iter->second;
                ret = ConstructWhiteList(commTag, isInterLink, socketType, dstRankLinkInfo,
                    wlistInfosVec);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Add][WhiteList]Construct white lists is failed. ret[%d]", ret), ret);
            }

            if (wlistInfosVec.size() > 0) {
                HCCL_INFO("[Add][WhiteList]wlist size[%u] ", wlistInfosVec.size());
                CHK_RET(res.second->AddWhiteList(wlistInfosVec));
                SaveWhiteListInfo(commTag, res.second, wlistInfosVec);
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclSocketManager::ConstructWhiteList(const std::string &commTag,
    bool isInterLink, NicType socketType,
    const HcclRankLinkInfo &dstRankLinkInfo, std::vector<SocketWlistInfo> &wlistInfosVec)
{
        SocketWlistInfo wlistInfo;
        u32 userRank = dstRankLinkInfo.userRank;
        for (u32 i = 0; i < dstRankLinkInfo.socketsPerLink; i++) {
            // 使用Client Rank作为确定标识,保证Client和Server的Tag一致
            std::string tag = MakeUniqueConnTag(commTag, isInterLink, userRank, i);
            wlistInfo.connLimit = GetConnLimit(socketType);
            s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1);
            if (sRet != EOK) {
                HCCL_ERROR("[Construct][WhiteList]memory copy failed. errorno[%d]", sRet);
                return HCCL_E_MEMORY;
            }

            wlistInfo.remoteIp.addr = dstRankLinkInfo.ip.GetBinaryAddress().addr;
            wlistInfo.remoteIp.addr6 = dstRankLinkInfo.ip.GetBinaryAddress().addr6;
            HCCL_DEBUG("[Construct][WhiteList]remoteIp[%s], tag[%s]",
                dstRankLinkInfo.ip.GetReadableAddress(), wlistInfo.tag);
            wlistInfosVec.push_back(wlistInfo);
        }

    return HCCL_SUCCESS;
}

// private
// 移除白名单
HcclResult HcclSocketManager::DelWhiteList(const std::string &commTag)
{
    std::unique_lock<std::mutex> lock(wlistMapMutex_);
    auto it = wlistInfosMap_.find(commTag);
    if (it != wlistInfosMap_.end()) {
        auto nicWlistInfosMap_ = it->second;
        for (auto iter = nicWlistInfosMap_.begin(); iter != nicWlistInfosMap_.end(); iter++) {
            auto wlistInfosVec_ = iter->second;
            iter->first->DelWhiteList(wlistInfosVec_);
        }
        wlistInfosMap_.erase(it);
    }
    return HCCL_SUCCESS;
}

// public API
// 与远端创建连接，异步接口
HcclResult HcclSocketManager::CreateSockets(const std::string &commTag,
    bool isInterLink, const HcclNetDevCtx netDevCtx, NicType socketType, HcclSocketRole localRole,
    const HcclIpAddress &localIp,
    const std::map<u32, HcclRankLinkInfo> &remoteInfos,
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > &socketsMap,
    std::map<u32, u32> &dstRankToUserRank,
    bool isSupportReuse)
{
    if (remoteInfos.size() == 0) {
        HCCL_ERROR("[Create][Sockets]remote infos map or local Ip is empty.");
        return HCCL_E_PARA;
    }

    HcclResult ret;
    for (auto iter = remoteInfos.begin(); iter != remoteInfos.end(); iter++) {
        std::vector<std::shared_ptr<HcclSocket> > rankSockets {};
        auto remoteIpIter = iter->second;
        u32 remoteUserRank = remoteIpIter.userRank;
        ret = CreateSockets(commTag, isInterLink, netDevCtx, socketType, localRole,
            localIp, remoteIpIter, rankSockets, isSupportReuse);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][Sockets]create to rank[%d] connection is failed. ret[%d]",
                iter->first, ret), ret);

        if (rankSockets.size() > 0) {
            socketsMap.insert(std::make_pair(iter->first, rankSockets));
            dstRankToUserRank.insert(std::make_pair(iter->first, remoteUserRank));
        }
    }
    return HCCL_SUCCESS;
}

// private
// 与指定IP创建Socket链接
HcclResult HcclSocketManager::CreateSockets(const std::string &commTag,
    bool isInterLink, const HcclNetDevCtx netDevCtx, NicType socketType, HcclSocketRole localRole,
    const HcclIpAddress &localIp, const HcclRankLinkInfo &remoteLinkInfo,
    std::vector<std::shared_ptr<HcclSocket> > &ipSockets,
    bool isSupportReuse)
{
    HcclResult ret;

    u32 remoteRank = remoteLinkInfo.userRank;
    auto remoteIp = remoteLinkInfo.ip;
    u32 remotePort = remoteLinkInfo.port;
    u32 socketsPerLink = remoteLinkInfo.socketsPerLink;

    // 支持复用，则先找下是否与相同的远端IP创建过链接
    if (isSupportReuse) {
        // 先根据远端IP(暂未管本端IP)找，找到了就返回，没找到正常创建
        u32 gotLinkNum = 0;
        GetSocketsByRankIP(commTag, remoteRank, remoteIp, socketsPerLink, ipSockets, gotLinkNum);
        socketsPerLink -= gotLinkNum;
        if (socketsPerLink == 0) {
            HCCL_INFO("[Create][Sockets]get reuse socket is success."
                "commTag[%s], remoteRank[%u], remoteIp[%s], localRank[%u].",
                commTag.c_str(), remoteRank, remoteIp.GetReadableIP(), userRank_);
            return HCCL_SUCCESS;
        }
    }

    ret = ConstructSockets(commTag, isInterLink, netDevCtx, socketsPerLink, socketType, remoteRank,
        remoteIp, remotePort, localIp, localRole, ipSockets);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Create][Sockets]construct socket is failed. ret[%d]", ret), ret);

    // 若作为客户端，需要发起Connect请求
    // 原实现是采用批量接口，当前使用Socket对象进行发起连接，无法使用批量接口，性能理论上可能有下降
    // 后继根据影响情况，看是否使用批量接口，若采用批量，则需要在更上层调用发起Connect请求才能最大程度地批量
    if (localRole == HcclSocketRole::SOCKET_ROLE_CLIENT) {
        for (u32 i = 0; i < ipSockets.size(); i++) {
            if (ipSockets[i]->GetStatus() != HcclSocketStatus::SOCKET_OK) {
                CHK_RET(ipSockets[i]->Connect());
            }
        }
    }

    SaveSockets(commTag, remoteRank, remoteIp, ipSockets);

    HCCL_INFO("[Create][Sockets]Create Sockets is success.");
    return HCCL_SUCCESS;
}

// private
// 当前仅用于析构时, 关闭所有Socket; 后继若有需要开放接口，注意 isSupCloseSockImmed_ 需要处理
void HcclSocketManager::DestroySockets()
{
    // 不对 isSupCloseSockImmed_ 进行判断处理, 强制关闭所有Socket
    std::unique_lock<std::mutex> lock(socketsMapMutex_);
    for (auto it = commSocketsMap_.begin(); it != commSocketsMap_.end(); it++) {
        auto rankSocketmap = it->second;
        for (auto iter = rankSocketmap.begin(); iter != rankSocketmap.end(); iter++) {
            DestroySockets(iter->second);
        }
        // 删除链接时，自动删除 WhiteList
        DelWhiteList(it->first);
    }
    commSocketsMap_.clear();
    return;
}

// public API
void HcclSocketManager::DestroySockets(const std::string &commTag)
{
    // 不支持立即关闭，那就直接返回，待 HcclSocketManager 析构时关闭
    if (!isSupCloseSockImmed_) {
        HCCL_INFO("[Destroy][Sockets]don't support immed close socket.");
        return;
    }

    std::unique_lock<std::mutex> lock(socketsMapMutex_);
    auto it = commSocketsMap_.find(commTag);
    if (it != commSocketsMap_.end()) {
        auto rankSocketmap = it->second;
        for (auto iter = rankSocketmap.begin(); iter != rankSocketmap.end(); iter++) {
            DestroySockets(iter->second);
        }
        commSocketsMap_.erase(it);
    }
    DelWhiteList(commTag);
    return;
}

// public API
// Destroy指定的远端Rank的所有的 IP 连接
// 暂无使用, 考虑删除
void HcclSocketManager::DestroySockets(const std::string &commTag, u32 rank)
{
    // 不支持立即关闭，那就直接返回，待 HcclSocketManager 析构时关闭
    if (!isSupCloseSockImmed_) {
        HCCL_INFO("[Destroy][Sockets]don't support immed close socket.");
        return;
    }

    std::unique_lock<std::mutex> lock(socketsMapMutex_);
    auto it = commSocketsMap_.find(commTag);
    if (it != commSocketsMap_.end()) {
        auto rankSocketmap = it->second;
        auto iter = rankSocketmap.find(rank);
        if (iter != rankSocketmap.end()) {
            DestroySockets(iter->second);
            rankSocketmap.erase(iter);
        }
    }
    return;
}

// private
void HcclSocketManager::DestroySockets(std::vector<std::shared_ptr<HcclSocket> > rankSockets)
{
    for (u32 j = 0; j < rankSockets.size(); j++) {
        auto temp = rankSockets[j];
        if (temp != nullptr) {
            temp->Close();
        }
    }

    return;
}

// public API
// isWaitEstablished 为 true 时, 连接建立完成后返回; 为 flase时, 连接请求发起后即返回. 默认为 true.
// 预留 调用时设置为 false, 通过多线程的方式提升建链性能.
HcclResult HcclSocketManager::CreateSockets(const std::string &commTag,
    bool isInterLink, const HcclNetDevCtx netDevCtx,
    const std::map<u32, HcclRankLinkInfo> &dstServerMap,
    const std::map<u32, HcclRankLinkInfo> &dstClientMap,
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > &serverSocketsMap,
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > &clientSocketsMap,
    bool isSupportReuse, bool isWaitEstablished)
{
    HCCL_DEBUG("[Create][Sockets]client map size[%u], server map size[%u]",
        dstClientMap.size(), dstServerMap.size());

    HCCL_DEBUG("[Create][Sockets]commTag %s, isInterLink %d, isSupportReuse %d, "\
        "isWaitEstablished %d", commTag.c_str(), isInterLink, isSupportReuse,
        isWaitEstablished);

    HcclResult ret;

    NicType socketType;
    CHK_RET(HcclNetDevGetNicType(netDevCtx, &socketType));
    HcclIpAddress localIp{0};
    CHK_RET(HcclNetDevGetLocalIp(netDevCtx, localIp));

    HCCL_DEBUG("[Create][Sockets]localIp %s", localIp.GetReadableAddress());

    for (auto it = dstServerMap.begin(); it != dstServerMap.end(); ++it) {
        HCCL_DEBUG("[Create][Sockets]dstServerMap rank %u", it->first);
        auto info =  it->second;
        HCCL_DEBUG("[Create][Sockets]dstServerMap userRank %u, devicePhyId %u, ip %s, port %u",
            info.userRank, info.devicePhyId, info.ip.GetReadableAddress(), info.port);
    }
    for (auto it = dstClientMap.begin(); it != dstClientMap.end(); ++it) {
        HCCL_DEBUG("[Create][Sockets]dstClientMap rank %u", it->first);
        auto info =  it->second;
        HCCL_DEBUG("[Create][Sockets]dstClientMap userRank %u, devicePhyId %u, ip %s, port %u",
            info.userRank, info.devicePhyId, info.ip.GetReadableAddress(), info.port);
    }

    // 作为服务端时，先填加白名单
    if (dstClientMap.size() > 0) {
        ret = AddWhiteList(commTag, isInterLink, socketType, localIp, dstClientMap);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][Sockets]Add white list failed. ret[%d]", ret), ret);
    }

    std::map<u32, u32> serverDstRankToUserRank; // 子平面rank 映射 通信域 rank
    std::map<u32, u32> clientDstRankToUserRank; // 子平面rank 映射 通信域 rank

    if (dstServerMap.size() > 0) {
        // 作为客户端，创建 Socket，并向所有的服务端发起建链请求
        ret = CreateSockets(commTag, isInterLink, netDevCtx, socketType, HcclSocketRole::SOCKET_ROLE_CLIENT,
            localIp, dstServerMap, clientSocketsMap, clientDstRankToUserRank, isSupportReuse);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][Sockets]Create sockets failed, local role is client. ret[%d]", ret), ret);
    }

    if (dstClientMap.size() > 0) {
        // 作为服务端，创建 Socket，然后返回
        ret = CreateSockets(commTag, isInterLink, netDevCtx, socketType, HcclSocketRole::SOCKET_ROLE_SERVER,
            localIp, dstClientMap, serverSocketsMap, serverDstRankToUserRank, isSupportReuse);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][Sockets]Create connection failed, local role is server."
                " ret[%d]", ret), ret);
    }

    HCCL_INFO("[Create][Sockets]client socket map size %u, server socket map size %u",
        serverSocketsMap.size(), serverSocketsMap.size());

    // 需要等待连接建立成功时，则会阻塞
    if (isWaitEstablished) {
        // 等待所有作为客户端的链接建立成功
        ret = WaitLinksEstablishCompleted(HcclSocketRole::SOCKET_ROLE_CLIENT, clientSocketsMap);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][Sockets]Wait links establish completed failed, local role is client. ret[%d]", ret);
            PrintErrorConnection(HcclSocketRole::SOCKET_ROLE_CLIENT, clientSocketsMap, clientDstRankToUserRank);
            return ret;
        }

        // 等待所有作为服务端的链接建立成功
        ret = WaitLinksEstablishCompleted(HcclSocketRole::SOCKET_ROLE_SERVER, serverSocketsMap);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][Sockets]Wait links establish completed failed, local role is server. ret[%d]", ret);
            PrintErrorConnection(HcclSocketRole::SOCKET_ROLE_SERVER, serverSocketsMap, serverDstRankToUserRank);
            return ret;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclSocketManager::AddWhiteList(const std::string &commTag,
    const HcclNetDevCtx netDevCtx,
    HcclRankLinkInfo remoteRankInfo)
{
    NicType socketType;
    CHK_RET(HcclNetDevGetNicType(netDevCtx, &socketType));
    HcclIpAddress localIp{0};
    CHK_RET(HcclNetDevGetLocalIp(netDevCtx, localIp));
 
    bool isInterLink{true};
    if (socketType == NicType::VNIC_TYPE) {
        isInterLink = false;
    }
 
    HcclResult ret;
    HcclSocketRole role =
        userRank_ < remoteRankInfo.userRank ? HcclSocketRole::SOCKET_ROLE_SERVER : HcclSocketRole::SOCKET_ROLE_CLIENT;
 
    std::map<u32, HcclRankLinkInfo> remoteMap;
    remoteMap.insert(std::make_pair(remoteRankInfo.userRank, remoteRankInfo));
 
    // 作为服务端时，先填加白名单
    if (role == HcclSocketRole::SOCKET_ROLE_SERVER) {
        ret = AddWhiteList(commTag, isInterLink, socketType, localIp, remoteMap);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][Sockets]Add white list failed. ret[%d]", ret), ret);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclSocketManager::CreateSingleLinkSocket(const std::string &commTag,
    const HcclNetDevCtx netDevCtx,
    HcclRankLinkInfo remoteRankInfo,
    std::vector<std::shared_ptr<HcclSocket> > &connectSockets,
    bool isWaitEstablished,
    bool isSupportReuse)
{
    NicType socketType;
    CHK_RET(HcclNetDevGetNicType(netDevCtx, &socketType));
    HcclIpAddress localIp{0};
    CHK_RET(HcclNetDevGetLocalIp(netDevCtx, localIp));
 
    bool isInterLink{true};
    if (socketType == NicType::VNIC_TYPE) {
        isInterLink = false;
    }
 
    HcclResult ret;
    HcclSocketRole role =
        userRank_ < remoteRankInfo.userRank ? HcclSocketRole::SOCKET_ROLE_SERVER : HcclSocketRole::SOCKET_ROLE_CLIENT;
 
    std::map<u32, HcclRankLinkInfo> remoteMap;
    remoteMap.insert(std::make_pair(remoteRankInfo.userRank, remoteRankInfo));
 
    // 作为服务端时，先填加白名单
    if (role == HcclSocketRole::SOCKET_ROLE_SERVER) {
        ret = AddWhiteList(commTag, isInterLink, socketType, localIp, remoteMap);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][Sockets]Add white list failed. ret[%d]", ret), ret);
    }
 
    std::map<u32, u32> remoteRankToUserRank; // 子平面rank 映射 通信域 rank
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > socketsMap;
    ret = CreateSockets(commTag, isInterLink, netDevCtx, socketType, role,
        localIp, remoteMap, socketsMap, remoteRankToUserRank, isSupportReuse);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Create][Sockets]Create connection failed, local role is server."
            " ret[%d]", ret), ret);
 
    if (isWaitEstablished) {
        ret = WaitLinksEstablishCompleted(role, socketsMap);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Create][Sockets]Wait links establish completed failed, local role is client. ret[%d]", ret);
            PrintErrorConnection(role, socketsMap, remoteRankToUserRank);
            return ret;
        }
    }
 
    connectSockets.clear();
    for (auto iter = socketsMap.begin(); iter != socketsMap.end(); iter++) {
        auto rankSockets = iter->second;
        for (u32 i = 0; i < rankSockets.size(); i++) {
            connectSockets.push_back(rankSockets[i]);
        }
    }
 
    return HCCL_SUCCESS;
}

// public
HcclResult HcclSocketManager::GetListenPortByIp(
    const NICDeployment nicDeployment, const HcclIpAddress &ipAddr, std::set<u32> &listenedPort)
{
    for (auto &res : serverSocketMap_) {
        if (res.first.ip == ipAddr) {
            listenedPort.insert(res.first.listenPort);
        }
    }

    if (listenedPort.size() > 0) {
        return HCCL_SUCCESS;
    }

    return HCCL_E_NOT_FOUND;
}

// private
void HcclSocketManager::TransformSocketStatus(HcclSocketStatus status, std::string &stringStatus) const
{
    switch (status) {
        case HcclSocketStatus::SOCKET_CONNECTING:
            stringStatus = "connecting";
            break;
        case HcclSocketStatus::SOCKET_OK:
            stringStatus = "connected";
            break;
        case HcclSocketStatus::SOCKET_TIMEOUT:
            stringStatus = "time out";
            break;
        case HcclSocketStatus::SOCKET_ERROR:
            stringStatus = "connect failed";
            break;
        case HcclSocketStatus::SOCKET_INIT:
        default:
            stringStatus = "no connect";
            break;
    }
}

// private
void HcclSocketManager::PrintSocketsInfo(const std::string &localRole,
    u32 rank, std::vector<std::shared_ptr<HcclSocket> > ipSockets) const
{
    for (u32 j = 0; j < ipSockets.size(); j++) {
        std::shared_ptr<HcclSocket> tempSocket = ipSockets[j];
        if (tempSocket->GetStatus() != HcclSocketStatus::SOCKET_OK) {
            std::string connectStatus;
            TransformSocketStatus(tempSocket->GetStatus(), connectStatus);
            HCCL_ERROR("   |  %s(%u)   |  %u  |   %s(%u)   |  %u  | %s | %s |  ",
                tempSocket->GetRemoteIp().GetReadableAddress(),
                rank, tempSocket->GetRemotePort(),
                tempSocket->GetLocalIp().GetReadableAddress(), userRank_, tempSocket->GetLocalPort(),
                localRole.c_str(), connectStatus.c_str());
        }
    }
}

// private
void HcclSocketManager::PrintErrorConnectionInfo(HcclSocketRole localRole,
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > &rankSocketsMap,
    std::map<u32, u32> &dstRankToUserRank) const
{
    std::string sRole;
    switch (localRole) {
        case HcclSocketRole::SOCKET_ROLE_SERVER:
            sRole = " server ";
            break;
        case HcclSocketRole::SOCKET_ROLE_CLIENT:
            sRole = " client ";
            break;
        default:
            sRole = "   NA   ";
            break;
    }

    for (auto iter = rankSocketsMap.begin(); iter != rankSocketsMap.end(); iter++) {
        auto rankSockets = iter->second;
        PrintSocketsInfo(sRole, dstRankToUserRank[iter->first], rankSockets);
    }
}

// private
void HcclSocketManager::PrintErrorConnection(HcclSocketRole localRole,
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > &rankSocketsMap,
    std::map<u32, u32> &dstRankToUserRank) const
{
    RPT_INPUT_ERR(true, "EI0006", std::vector<std::string>({"reason"}), \
        std::vector<std::string>({GET_SOCKET_TIMEOUT_REASON}));

    // 原实现中，打印输出了一个num，而实际调用中，这个num都是1，所以当前删除了
    HCCL_ERROR("   _________________________LINK_ERROR_INFO___________________________");
    HCCL_ERROR("   |  comm error, device[%d] ", deviceLogicId_);
    HCCL_ERROR("   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   "
        "|   Status   |");
    HCCL_ERROR("   |--------------------|--------------------|----------|------------|-----------------"
        "|-----------------|");

    PrintErrorConnectionInfo(localRole, rankSocketsMap, dstRankToUserRank);

    HCCL_ERROR("   ___________________________________________________________________  ");
    HCCL_ERROR("the connection failure between this device and target device may be due to the following reasons:");
    HCCL_ERROR("1. the connection between this device and the target device is abnormal.");
    HCCL_ERROR("2. an exception occurred at the target devices.");
    HCCL_ERROR("3. the time difference between the execution of hcom on this device and the target device exceeds the "\
        "timeout threshold. make sure this by keyworld [Entry-]");
    HCCL_ERROR("4. the behavior of executing the calculation graph on this device and the target device is " \
        "inconsistent. ");
    HCCL_ERROR("5. The TLS switch is inconsistent, or the TLS certificate expires. ");
    HCCL_ERROR("6. If src and dst IP address can be pinged, check whether the device IP address conflicts. ");
    HCCL_ERROR("7. Now you can freely specify a port for listening and connecting. If an invalid port is chosen, "
        "it may result in failed listening and connection timeouts");
    return;
}

// private
// 获取SOCKET连接的 ConnLimit，临时写一个，后继看如何判定，后继考虑初始化到Listen的Socket中
u32 HcclSocketManager::GetConnLimit(NicType socketType)
{
    u32 connLimit = 1;
    switch (socketType) {
        case NicType::DEVICE_NIC_TYPE:
            connLimit = NIC_SOCKET_CONN_LIMIT;
            break;
        case NicType::VNIC_TYPE:
            connLimit = VNIC_SOCKET_CONN_LIMIT;
            break;
        case NicType::HOST_NIC_TYPE:
            connLimit = HOST_SOCKET_CONN_LIMIT;
            break;
        default:
            break;
    }
    return connLimit;
}

// 生成统一的 SocketTag, 用于标识Socket, 对于一对链接(Server <--> Client), SocketTag 需要相同
std::string HcclSocketManager::MakeUniqueConnTag(const std::string &commTag, bool isInterLink,
    u32 userRank, u32 indexForLink)
{
    std::string tmpStr = isInterLink ? "_Inter_" : "_Intra_";
    std::string socketTag = commTag + tmpStr + "MultiSocket_" + \
        std::to_string(userRank) + "_" + std::to_string(indexForLink);
    return socketTag;
}

// 保存白名单，后继保存到Listen的Socket对象中
void HcclSocketManager::SaveWhiteListInfo(const std::string &commTag, std::shared_ptr<HcclSocket> &socket,
    const std::vector<SocketWlistInfo> wlistInfos)
{
    // 将Add成功的whiteList, 保存在Socket对象中, 方便后继Del.
    // 理论上, 接口 DelSocketWhiteList, 上层应该没有调用的必要性.
    // socket关闭时, 底层应该会自动清除, 而关闭前, 也没看到调用的必要性.
    // 综上, 即保存这个 wlistInfosMap_ 必要性可能不大.
    std::unique_lock<std::mutex> lock(wlistMapMutex_);
    auto it = wlistInfosMap_.find(commTag);
    if (it == wlistInfosMap_.end()) {
        std::map<std::shared_ptr<HcclSocket>, std::vector<SocketWlistInfo> > nicWlistInfosMap;
        nicWlistInfosMap.insert(std::make_pair(socket, wlistInfos));
        wlistInfosMap_.insert(std::make_pair(commTag, nicWlistInfosMap));
    } else {
        auto &nicWlistInfosMap = it->second;
        auto iter = nicWlistInfosMap.find(socket);
        if (iter == nicWlistInfosMap.end()) {
            nicWlistInfosMap.insert(std::make_pair(socket, wlistInfos));
        } else {
            auto &wlist = iter->second;
            wlist.insert(wlist.end(), wlistInfos.begin(), wlistInfos.end());
        }
    }
}

// private
// 根据相关参数构造 HcclSocket 对象
HcclResult HcclSocketManager::ConstructSockets(const std::string &commTag, bool isInterLink,
    const HcclNetDevCtx netDevCtx, u32 socketsPerLink,
    NicType socketType, u32 remoteUserRank,
    const HcclIpAddress &remoteIp, u32 remotePort,
    const HcclIpAddress &localIp, HcclSocketRole localRole,
    std::vector<std::shared_ptr<HcclSocket> > &socketList)
{
    // 使用Client Rank作为确定标识,保证Client和Server的Tag一致
    u32 clientRank = localRole == HcclSocketRole::SOCKET_ROLE_CLIENT ? userRank_ : remoteUserRank;
    for (u32 i = 0; i < socketsPerLink; i++) {
        std::string socketTag = MakeUniqueConnTag(commTag, isInterLink, clientRank, i);
        std::shared_ptr<HcclSocket> tempSocket;
        EXECEPTION_CATCH((tempSocket = std::make_shared<HcclSocket>(socketTag,
            netDevCtx, remoteIp, remotePort, localRole)), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(tempSocket);
        CHK_RET(tempSocket->Init());
        socketList.push_back(tempSocket);
    }
    return HCCL_SUCCESS;
}

// private
// 保存 HcclSocket 对象
void HcclSocketManager::SaveSockets(const std::string &commTag, u32 remoteRank, const HcclIpAddress &remoteIp,
    std::vector<std::shared_ptr<HcclSocket> > &ipSockets)
{
    // 将 ipSockets 累加到 commSocketsMap_ 中
    HCCL_DEBUG("[Save][Sockets]commTag[%s], remoteRank[%u], remoteIp[%s], localRank[%u], save socket size[%u].",
        commTag.c_str(), remoteRank, remoteIp.GetReadableIP(), userRank_, ipSockets.size());
    
    std::unique_lock<std::mutex> lock(socketsMapMutex_);
    auto it = commSocketsMap_.find(commTag);
    if (it == commSocketsMap_.end()) {
        std::map<u32, std::vector<std::shared_ptr<HcclSocket> > > rankSocketmap {};
        rankSocketmap.insert(std::make_pair(remoteRank, ipSockets));
        commSocketsMap_.insert(std::make_pair(commTag, rankSocketmap));
    } else {
        auto &rankSocketmap = it->second;
        auto iter = rankSocketmap.find(remoteRank);
        if (iter == rankSocketmap.end()) {
            rankSocketmap.insert(std::make_pair(remoteRank, ipSockets));
            HCCL_DEBUG("[Save][Sockets]rankSocketmap size[%u], ipSockets size[%u].",
                rankSocketmap.size(), ipSockets.size());
        } else {
            auto &rankSockets = iter->second;
            // 当前直接插入，不判断本端IP，远端IP是否相同
            for (u32 i = 0; i < ipSockets.size(); i++) {
                rankSockets.push_back(ipSockets[i]);
            }
            HCCL_DEBUG("[Save][Sockets]rankSocketmap size[%u], rankSockets size[%u].",
                rankSocketmap.size(), rankSockets.size());
        }
    }
    HCCL_DEBUG("[Save][Sockets]commSocketsMap_ size[%u].", commSocketsMap_.size());
}

// private
void HcclSocketManager::GetSocketsByRankIP(const std::string &commTag, u32 remoteRank, const HcclIpAddress &remoteIp,
    u32 socketsPerLink, std::vector<std::shared_ptr<HcclSocket> > &ipSockets, u32 &gotLinkNum)
{
    HCCL_DEBUG("[Get][SocketsByRankIP]commTag[%s], remoteRank[%u], remoteIp[%s], localRank[%u], SocketsMap size[%u].",
        commTag.c_str(), remoteRank, remoteIp.GetReadableIP(), userRank_, commSocketsMap_.size());

    gotLinkNum = 0;
    auto it = commSocketsMap_.find(commTag);
    if (it != commSocketsMap_.end()) {
        auto &rankSocketmap = it->second;
        HCCL_INFO("[Get][SocketsByRankIP]rankSocketmap size[%u].", rankSocketmap.size());
        auto iter = rankSocketmap.find(remoteRank);
        if (iter != rankSocketmap.end()) {
            auto &rankSockets = iter->second;
            GetSocketsByRankIP(remoteIp, socketsPerLink, rankSockets, ipSockets, gotLinkNum);
        }
    }
}

void HcclSocketManager::GetSocketsByRankIP(const HcclIpAddress &remoteIp, u32 socketsPerLink,
    std::vector<std::shared_ptr<HcclSocket>> &rankSockets, std::vector<std::shared_ptr<HcclSocket>> &ipSockets,
    u32 &gotLinkNum)
{
    for (u32 idx = 0; idx < rankSockets.size(); idx++) {
        if (rankSockets[idx]->GetRemoteIp() == remoteIp) {
            ipSockets.push_back(rankSockets[idx]);
            gotLinkNum++;
            if (gotLinkNum == socketsPerLink) {
                break;
            }
        }
    }
}

// private
// 同步接口，更新连接状态，并返回连接成功的连接数量
HcclResult HcclSocketManager::WaitLinkEstablish(std::shared_ptr<HcclSocket> socket)
{
    CHK_SMART_PTR_NULL(socket);
    u32 count = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    HCCL_DEBUG("[Wait][LinkEstablish]waiting for sockets link up...");
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("[Wait][LinkEstablish]wait socket establish timeout, role[%u] rank[%u] timeout[%lld]",
                static_cast<u32>(socket->GetLocalRole()), userRank_, timeout);
            socket->SetStatus(HcclSocketStatus::SOCKET_TIMEOUT);
            RPT_INPUT_ERR(true, "EI0006", std::vector<std::string>({"reason"}), \
                std::vector<std::string>({GET_SOCKET_TIMEOUT_REASON}));
            return HCCL_E_TIMEOUT;
        }
        HcclSocketStatus status = socket->GetStatus();
        if (status == HcclSocketStatus::SOCKET_OK) {
            HCCL_DEBUG("[Wait][LinkEstablish]socket is establish. localIp[%s], remoteIp[%s]",
                socket->GetLocalIp().GetReadableIP(), socket->GetRemoteIp().GetReadableIP());
            return HCCL_SUCCESS;
        } else if (status == HcclSocketStatus::SOCKET_CONNECTING) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            // 日志过滤, 50 次才打印一次
            if (count % 50 == 0) {
                HCCL_DEBUG("[Wait][LinkEstablish]socket is connectting ");
            }
            count++;
            
            continue;
        } else if (status == HcclSocketStatus::SOCKET_TIMEOUT) {
            return HCCL_E_TIMEOUT;
        } else {
            socket->SetStatus(HcclSocketStatus::SOCKET_ERROR);
            return HCCL_E_TCP_CONNECT;
        }
    }
    return HCCL_E_TCP_CONNECT;
}

HcclResult HcclSocketManager::WaitLinksEstablishCompleted(HcclSocketRole localRole,
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > &socketsMap, std::map<u32, u32> &dstRankToUserRank)
{
    HcclResult ret = WaitLinksEstablishCompleted(localRole, socketsMap);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Create][Sockets]Wait links establish completed failed, local role is client. ret[%d]", ret);
        PrintErrorConnection(localRole, socketsMap, dstRankToUserRank);
        return ret;
    }
    return HCCL_SUCCESS;
}

// private
// 同步接口，等待连接建立完成
HcclResult HcclSocketManager::WaitLinksEstablishCompleted(HcclSocketRole localRole,
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > &rankSocketsMap)
{
    for (auto iter = rankSocketsMap.begin(); iter != rankSocketsMap.end(); iter++) {
        auto rankSockets = iter->second;
        for (u32 i = 0; i < rankSockets.size(); i++) {
            HcclResult ret = WaitLinkEstablish(rankSockets[i]);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Wait][LinksEstablishCompleted] is failed. ret[%d].",
                ret), ret);
        }
    }
    return HCCL_SUCCESS;
}
}  // namespace hccl
