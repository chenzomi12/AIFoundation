/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_MANAGER_H
#define TRANSPORT_MANAGER_H

#include <mutex>
#include <unordered_map>
#include "base.h"
#include "coll_alg_param.h"
#include "hccl_socket_manager.h"
#include "dispatcher.h"
#include "mem_device_pub.h"
#include "transport_pub.h"
#include "ccl_buffer_manager.h"
#include "externalinput_pub.h"
#include "sal_pub.h"
#include "hccl_hash_utils.h"
#include "common.h"

namespace hccl {

struct TransportData {
    LinkMode linkMode{LinkMode::LINK_RESERVED_MODE};
    std::vector<HcclIpAddress> remoteIpAddr;
    u32 remoteUserrank{INVALID_VALUE_RANKID};
    u32 remoteWorldRank{INVALID_VALUE_RANKID};
    s32 remoteDeviceId{-1};
    DevType deviceType{DevType::DEV_TYPE_COUNT};
    DeviceMem inputMem{DeviceMem()};
    DeviceMem outputMem{DeviceMem()};
    bool supportDataReceivedAck{false};
    u32 remoteSocketPort;

    TransportData(LinkMode linkMode,
            const std::vector<HcclIpAddress> &remoteIpAddr,
            u32 remoteUserrank,
            u32 remoteWorldRank,
            s32 remoteDeviceId,
            DevType deviceType,
            const DeviceMem &inputMem,
            const DeviceMem &outputMem,
            bool supportDataReceivedAck,
            u32 remoteSocketPort)
        : linkMode(linkMode),
        remoteIpAddr(remoteIpAddr),
        remoteUserrank(remoteUserrank),
        remoteWorldRank(remoteWorldRank),
        remoteDeviceId(remoteDeviceId),
        deviceType(deviceType),
        inputMem(inputMem),
        outputMem(outputMem),
        supportDataReceivedAck(supportDataReceivedAck),
        remoteSocketPort(remoteSocketPort) {};

    bool operator==(const TransportData &that) const
    {
        return (linkMode == that.linkMode) &&
            (remoteIpAddr == that.remoteIpAddr) &&
            (remoteUserrank == that.remoteUserrank) &&
            (remoteWorldRank == that.remoteWorldRank) &&
            (remoteDeviceId == that.remoteDeviceId) &&
            (deviceType == that.deviceType) &&
            (inputMem == that.inputMem) &&
            (outputMem == that.outputMem) &&
            (supportDataReceivedAck == that.supportDataReceivedAck) &&
            (remoteSocketPort == that.remoteSocketPort);
    }
};
}

namespace std {

template <> class hash<hccl::TransportData> {
public:
    size_t operator()(const hccl::TransportData &transportData) const
    {
        auto linkMode = hash<s32>{}(static_cast<s32>(transportData.linkMode));
        auto remoteIpAddrFamily = hash<s32>{}(transportData.remoteIpAddr[0].GetFamily());
        auto remoteIpAddr = hash<string>{}(string(transportData.remoteIpAddr[0].GetReadableAddress()));
        auto remoteUserrank = hash<u32>{}(transportData.remoteUserrank);
        auto remoteWorldRank = hash<u32>{}(transportData.remoteWorldRank);
        auto remoteDeviceId = hash<s32>{}(transportData.remoteDeviceId);
        auto deviceType = hash<s32>{}(static_cast<s32>(transportData.deviceType));
        auto inputMemPtr = hash<u64>{}(reinterpret_cast<u64>(transportData.inputMem.ptr()));
        auto inputMemSize = hash<u64>{}(transportData.inputMem.size());
        auto outputMemPtr = hash<u64>{}(reinterpret_cast<u64>(transportData.outputMem.ptr()));
        auto outputMemSize = hash<u64>{}(transportData.outputMem.size());
        auto supportDataReceivedAck = hash<bool>{}(transportData.supportDataReceivedAck);
        auto remoteSocketPort = hash<u32>{}(transportData.remoteSocketPort);

        return hccl::HashCombine({linkMode, remoteIpAddrFamily, remoteIpAddr, remoteUserrank, remoteWorldRank,
            remoteDeviceId, deviceType, inputMemPtr, inputMemSize, outputMemPtr, outputMemSize,
            supportDataReceivedAck, remoteSocketPort});
    }
};
}  // namespace std

namespace hccl {

struct TransportIOMem {
    DeviceMem cclInputMem;
    DeviceMem cclOutputMem;
    DeviceMem paramInputMem;
    DeviceMem paramOutputMem;
    DeviceMem scratchMem;
    DeviceMem aivInputMem;
    DeviceMem aivOutputMem;
};

class TransportManager {
public:
    TransportManager(CCLBufferManager &cclBufferManager,
        const std::unique_ptr<HcclSocketManager> &socketManager_,
        const HcclDispatcher &dispatcher,
        const std::unique_ptr<NotifyPool> &notifyPool,
        const std::vector<RankInfo> &rankInfoList,
        RankId userRank,
        const std::string &identifier,
        s32 deviceLogicId,
        NICDeployment nicDeployment,
        bool isHaveCpuRank,
        const void *transportResourceInfoAddr,
        size_t transportResourceInfoSize,
        bool isUseRankPort,
        bool isUsedRdmaOuter,
        const std::vector<u32> &ranksPort,
        bool useSdidForDeviceId,
        const std::vector<HcclIpAddress> &devIpAddr,
        const HcclIpAddress &hostIp,
        const HcclIpAddress &localVnicIp,
        std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap);

    ~TransportManager();

    HcclResult CreateVirturalTransport(SingleSubCommTransport& singleSubCommTransport);
    HcclResult Alloc(const std::string &tag, const TransportIOMem &transMem, OpCommTransport &opTransportResponse);
    HcclResult IncreAlloc(const std::string &tag, const TransportIOMem &transMem, OpCommTransport &opTransportReq,
        OpCommTransport &opTransportResponse);
    TransportManager(TransportManager const&) = delete;                 // Copy construct
    TransportManager(TransportManager&&) = delete;                      // Move construct
    TransportManager& operator=(TransportManager const&) = delete;      // Copy assign
    TransportManager& operator=(TransportManager &&) = delete;          // Move assign

private:
    HcclResult GetIOMem(const TransportIOMem &transMem,
        const TransportMemType inputMemType, const TransportMemType outputMemType,
        DeviceMem &inputMem,  DeviceMem &outputMem);
    u32 GetRemoteNicPort(u32 remoteRank);
    bool IsSupportInterHccs(const u32 dstRank);
    void UpdateIsInterRdma(const u32 remoteRank, bool &isInterRdma, bool forceRdma);
    u32 GetInterRemotePort(s32 devicePhyId, u32 dstUserRank);
    HcclResult MakeRemoteLinkInfo(const u32 remoteRank, bool isInterRdma,
        u32 socketsPerLink, HcclRankLinkInfo &remoteLinkInfo);
    HcclResult CreateDestSockets(const std::string &newTag, RankId remoteRank, u64 taskNum,
        std::vector<std::shared_ptr<HcclSocket> > &connectSockets, bool &isInterRdma, bool forceRdma = false);
    u32 GetSocketsPerLink(u64 taskNum);
    HcclResult SetMachinePara(const std::string &tag, MachineType machineType, const std::string &serverId, u32 dstRank,
        const bool supportDataReceivedAck, const LinkMode linkMode,
        const std::vector<std::shared_ptr<HcclSocket> > &socketList,
        const DeviceMem &inputMem, const DeviceMem &outputMem, MachinePara &machinePara);
    TransportType GetTransportType(const u32 dstRank, bool isUsedRdma);
    void SetTransportParam(TransportPara &para, MachinePara &machinePara);
    HcclResult TransportInit(const u32 dstRank, MachinePara &machinePara,
        std::shared_ptr<Transport> &link, bool useOneDoorbell, bool isUsedRdma);
    HcclResult CreateLink(const std::string &tag, const ErrContextPub &error_context, const MachineType machineType,
        const std::string &serverId, const u32 remoteRank, const bool supportDataReceivedAck, const LinkMode linkMode,
        const bool enableUseOneDoorbell, const std::string threadStr,
        const std::vector<std::shared_ptr<HcclSocket> > sockets,
        const DeviceMem inputMem, const DeviceMem outputMem, bool isUsedRdma,
        std::shared_ptr<Transport> &link);
    HcclResult ConstructTransTag(const std::string& tag, std::string& transTag, bool isInterRdma);
    HcclResult ExceptionHandle(const std::string &tag, OpCommTransport &opTransportResponse);

    CCLBufferManager &cclBufferManager_;
    const std::unique_ptr<HcclSocketManager> &socketManager_;
    const HcclDispatcher &dispatcher_;
    const std::unique_ptr<NotifyPool> &notifyPool_;
    const std::vector<RankInfo> &rankInfoList_;
    RankId userRank_;
    std::string identifier_;
    s32 deviceLogicId_;
    NICDeployment nicDeployment_;
    bool isHaveCpuRank_{ false };
    const void *transportResourceInfoAddr_;
    size_t transportResourceInfoSize_;
    bool isUseRankPort_{ false };
    bool isUsedRdmaOuter_{ false };
    const std::vector<u32> &ranksPort_;
    bool useSdidForDeviceId_{ false };
    const std::vector<HcclIpAddress> &devIpAddr_;
    const HcclIpAddress &hostIp_;
    const HcclIpAddress &localVnicIp_;
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap_;

    std::unordered_map<TransportData, LINK> transportMap_;
    std::vector<u32> enableP2PDevices_;

    std::vector<std::string> socketTagVec_;
};
}  // namespace hccl


#endif /* TRANSPORT_MANAGER_H */