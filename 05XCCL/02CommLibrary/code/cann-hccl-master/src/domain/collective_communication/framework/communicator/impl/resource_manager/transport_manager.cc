/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_manager.h"
#include "device_capacity.h"
#include "p2p_mgmt_pub.h"
#include <algorithm>

namespace hccl {

TransportManager::TransportManager(CCLBufferManager &cclBufferManager,
                                   const std::unique_ptr<HcclSocketManager> &socketManager,
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
                                   std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap)
    : cclBufferManager_(cclBufferManager), socketManager_(socketManager), dispatcher_(dispatcher),
    notifyPool_(notifyPool), rankInfoList_(rankInfoList), userRank_(userRank), identifier_(identifier),
    deviceLogicId_(deviceLogicId), nicDeployment_(nicDeployment), isHaveCpuRank_(isHaveCpuRank),
    transportResourceInfoAddr_(transportResourceInfoAddr), transportResourceInfoSize_(transportResourceInfoSize),
    isUseRankPort_(isUseRankPort), isUsedRdmaOuter_(isUsedRdmaOuter), ranksPort_(ranksPort),
    useSdidForDeviceId_(useSdidForDeviceId), devIpAddr_(devIpAddr), hostIp_(hostIp), localVnicIp_(localVnicIp),
    netDevCtxMap_(netDevCtxMap)
{
}

TransportManager::~TransportManager()
{
    if (enableP2PDevices_.size() != 0) {
        (void)P2PMgmtPub::DisableP2P(enableP2PDevices_);
        enableP2PDevices_.clear();
    }
}

constexpr u32 EXCEPTION_DELAY_US_COUNT = 100000;
HcclResult TransportManager::ExceptionHandle(const std::string &tag, OpCommTransport &opTransportResponse)
{
    for (auto &levelNSubCommTransport : opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    bool isInterRdma;
                    UpdateIsInterRdma(transportRequest.remoteUserRank, isInterRdma, singleSubCommTransport.isUsedRdma);

                    HcclRankLinkInfo remoteLinkInfo;
                    MakeRemoteLinkInfo(transportRequest.remoteUserRank, isInterRdma, 1, remoteLinkInfo);

                    HcclIpAddress ipAddr;
                    if (isInterRdma || Is310PDevice()) {
                        ipAddr = nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE ?
                            devIpAddr_[0]: hostIp_;
                    } else {
                        ipAddr = localVnicIp_;
                    }

                    std::string newTag;
                    CHK_RET(ConstructTransTag(tag, newTag, isInterRdma));
                    CHK_RET(socketManager_->AddWhiteList(newTag, netDevCtxMap_[ipAddr],
                        remoteLinkInfo));
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::CreateVirturalTransport(SingleSubCommTransport& singleSubCommTransport)
{
    MachinePara machinePara;
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(
        GetExternalInputHcclLinkTimeOut());

    singleSubCommTransport.virtualLinks.clear();
    singleSubCommTransport.virtualLinks.resize(singleSubCommTransport.transportRequests.size());

    for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
        TransportPara para {};
        para.virtualFlag = true;
        para.timeout = kdefaultTimeout;
        para.index = i;
        singleSubCommTransport.virtualLinks[i].reset(new (std::nothrow) Transport(TransportType::TRANS_TYPE_RESERVED,
            para, dispatcher_, notifyPool_, machinePara));
        CHK_PRT_RET(!singleSubCommTransport.virtualLinks[i], HCCL_ERROR("[CreateVirturalTransport]In create link," \
            "new link failed"), HCCL_E_PTR);
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::Alloc(const std::string &tag, const TransportIOMem &transMem,
    OpCommTransport &opTransportResponse)
{
    CHK_RET(notifyPool_->RegisterOp(tag));
    for (auto &levelNSubCommTransport : opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            std::vector<std::unique_ptr<std::thread> > linkThreads; // 建链所需线程
            linkThreads.resize(singleSubCommTransport.transportRequests.size());
            u32 threadsRapplyNum{0};                                // 线程使用计数器

            singleSubCommTransport.links.clear();
            singleSubCommTransport.links.reserve(singleSubCommTransport.transportRequests.size());

            if (singleSubCommTransport.needVirtualLink) {
                // task多线程并行下发，根据当前transport创建vtransport信息
                CHK_RET(CreateVirturalTransport(singleSubCommTransport));
            }

            u32 linkIdx = 0;
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                singleSubCommTransport.links.emplace_back(std::make_shared<Transport>(nullptr));
                if (transportRequest.isValid) {
                    DeviceMem inputMem;
                    DeviceMem outputMem;
                    HCCL_DEBUG("transportRequest.inputMemType[%d] transportRequest.outputMemType[%d]",
                        transportRequest.inputMemType, transportRequest.outputMemType);
                    GetIOMem(transMem, transportRequest.inputMemType, transportRequest.outputMemType,
                        inputMem, outputMem);

                    std::vector<std::shared_ptr<HcclSocket> > connectSockets;
                    bool isInterRdma;
                    HcclResult ret = CreateDestSockets(tag, transportRequest.remoteUserRank, singleSubCommTransport.taskNum,
                        connectSockets, isInterRdma, singleSubCommTransport.isUsedRdma);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc]Create dest sockets failed"), ret);

                    MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank?
                        MachineType::MACHINE_SERVER_TYPE : MachineType::MACHINE_CLIENT_TYPE;

                    std::string threadStr = (isInterRdma? "HcclTerL_" : "HcclIntra_") +
                        std::to_string(threadsRapplyNum);
                    linkThreads[threadsRapplyNum].reset(
                        new (std::nothrow) std::thread(&TransportManager::CreateLink,
                            this, tag, hrtErrMGetErrorContextPub(),
                            machineType, rankInfoList_[userRank_].serverId, transportRequest.remoteUserRank,
                            singleSubCommTransport.supportDataReceivedAck, singleSubCommTransport.linkMode,
                            singleSubCommTransport.enableUseOneDoorbell, threadStr, connectSockets,
                            inputMem, outputMem, singleSubCommTransport.isUsedRdma,
                            std::ref(singleSubCommTransport.links.back())));
                        CHK_SMART_PTR_NULL(linkThreads[threadsRapplyNum]); // 异常时其他线程待处理

                    threadsRapplyNum++;
                }
                linkIdx++;
            }

            for (u32 index = 0; index < linkThreads.size(); index++) {
                if (linkThreads[index] == nullptr) {
                    continue;
                }
                linkThreads[index]->join(); // 等待线程执行完毕
                CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
            }
            linkThreads.clear();

            linkIdx = 0;
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    if (singleSubCommTransport.links[linkIdx] == nullptr) {
                        HCCL_ERROR("[Create]errNo[0x%016llx] transport create fail in thread, local[%d] remote[%d]",
                            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), userRank_, transportRequest.remoteUserRank);
                        (void)ExceptionHandle(tag, opTransportResponse);
                        SaluSleep(EXCEPTION_DELAY_US_COUNT);
                        (void)notifyPool_->UnregisterOp(tag);
                        return HCCL_E_NOT_FOUND;
                    }
                }
                linkIdx++;
            }
            for (auto &tmpTag : socketTagVec_) {
                (void)socketManager_->DestroySockets(tmpTag);
            }
            socketTagVec_.clear();
        }
    }
    CHK_RET(notifyPool_->UnregisterOp(tag));

    return HCCL_SUCCESS;
}

HcclResult TransportManager::IncreAlloc(const std::string &tag, const TransportIOMem &transMem,
    OpCommTransport &opTransportReq, OpCommTransport &opTransportResponse)
{
    CHK_RET(notifyPool_->RegisterOp(tag));
    for (u32 levelIndex = 0; levelIndex < opTransportReq.size(); levelIndex++) {
        for (u32 ringIndex = 0; ringIndex < opTransportReq[levelIndex].size(); ringIndex++) {
            std::vector<std::unique_ptr<std::thread> > linkThreads; // 建链所需线程
            linkThreads.resize(opTransportReq[levelIndex][ringIndex].transportRequests.size());
            u32 threadsRapplyNum{0};                                // 线程使用计数器
            SingleSubCommTransport &reqSingleSubComm = opTransportReq[levelIndex][ringIndex];
            SingleSubCommTransport &respSingleSubComm = opTransportResponse[levelIndex][ringIndex];
            for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size(); rankIndex++) {
                TransportRequest &transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                CHK_PRT_RET(rankIndex >= respSingleSubComm.links.size(),
                    HCCL_ERROR("[IncreAlloc] The remote rank_id[%u] is larger than the existent respSingleSubComm map "\
                    "size[%u]", rankIndex, respSingleSubComm.links.size()), HCCL_E_PARA);
                if (respSingleSubComm.links[rankIndex] != nullptr &&
                    respSingleSubComm.links[rankIndex]->GetLinkType() != hccl::LinkType::LINK_RESERVED) {
                    HCCL_INFO("[IncreAlloc] The link to remote userRank[%u] has existed", transportRequest.remoteUserRank);
                    continue;
                }
                if (transportRequest.isValid) {
                    DeviceMem inputMem;
                    DeviceMem outputMem;
                    GetIOMem(transMem, transportRequest.inputMemType, transportRequest.outputMemType,
                        inputMem, outputMem);

                    std::vector<std::shared_ptr<HcclSocket> > connectSockets;
                    bool isInterRdma;
                    HcclResult ret = CreateDestSockets(tag, transportRequest.remoteUserRank, reqSingleSubComm.taskNum,
                        connectSockets, isInterRdma);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[IncreAlloc]Create dest sockets failed"), ret);

                    MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank?
                        MachineType::MACHINE_SERVER_TYPE : MachineType::MACHINE_CLIENT_TYPE;
                    std::string threadStr = (isInterRdma? "HcclTerL_" : "HcclIntra_") +
                        std::to_string(threadsRapplyNum);
                    linkThreads[threadsRapplyNum].reset(new (std::nothrow) std::thread(&TransportManager::CreateLink,
                            this, tag, hrtErrMGetErrorContextPub(),
                            machineType, rankInfoList_[userRank_].serverId, transportRequest.remoteUserRank,
                            reqSingleSubComm.supportDataReceivedAck, reqSingleSubComm.linkMode,
                            reqSingleSubComm.enableUseOneDoorbell, threadStr, connectSockets, inputMem, outputMem,
                            reqSingleSubComm.isUsedRdma, std::ref(respSingleSubComm.links[rankIndex])));
                        CHK_SMART_PTR_NULL(linkThreads[threadsRapplyNum]); // 异常时其他线程待处理
                    threadsRapplyNum++;
                }
            }
            for (u32 index = 0; index < linkThreads.size(); index++) {
                if (linkThreads[index] != nullptr && linkThreads[index]->joinable()) {
                    linkThreads[index]->join();
                    CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
                }
            }
            linkThreads.clear();

            for (auto &tmpTag : socketTagVec_) {
                (void)socketManager_->DestroySockets(tmpTag);
            }
            socketTagVec_.clear();
        }
    }
    CHK_RET(notifyPool_->UnregisterOp(tag));
    return HCCL_SUCCESS;
}

HcclResult TransportManager::ConstructTransTag(const std::string& tag, std::string& transTag, bool isInterRdma)
{
    transTag = (Is310PDevice() || isHaveCpuRank_) ? tag : identifier_ + "_res_optimize";
    std::string tmpStr = isInterRdma ? "_Inter_" : "_Intra_";
    transTag += tmpStr;
    return HCCL_SUCCESS;
}

HcclResult TransportManager::GetIOMem(const TransportIOMem &transMem,
    const TransportMemType inputMemType, const TransportMemType outputMemType,
    DeviceMem &inputMem,  DeviceMem &outputMem)
{
    if (inputMemType == CCL_INPUT) {
        inputMem = transMem.cclInputMem;
    } else if (inputMemType == SCRATCH) {
        inputMem = transMem.scratchMem;
    } else if (inputMemType == PARAM_INPUT) {
        inputMem = transMem.paramInputMem;
    } else if (inputMemType == AIV_INPUT) {
        inputMem = transMem.aivInputMem;
    } else if (inputMemType == CCL_OUTPUT) {
        inputMem = transMem.cclOutputMem;
    } else {
        HCCL_ERROR("inputMemType is Invalid, inputMem not set");
        return HCCL_E_INTERNAL;
    }

    if (outputMemType == CCL_OUTPUT) {
        outputMem = transMem.cclOutputMem;
    } else if (outputMemType == SCRATCH) {
        outputMem = transMem.scratchMem;
    } else if (outputMemType == PARAM_OUTPUT) {
        outputMem = transMem.paramOutputMem;
    } else if (outputMemType == AIV_OUTPUT) {
        outputMem = transMem.aivOutputMem;
    } else if (outputMemType == CCL_INPUT) {
        outputMem = transMem.cclInputMem;
    } else if (outputMemType == PARAM_INPUT) {
        outputMem = transMem.paramInputMem;
    } else {
        HCCL_ERROR("outputMemType is Invalid, inputMem not set");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

u32 TransportManager::GetRemoteNicPort(u32 remoteRank)
{
    if (isHaveCpuRank_) {
        isUseRankPort_ = true;
    }
    return GetNicPort(rankInfoList_[remoteRank].devicePhyId, ranksPort_,
        rankInfoList_[remoteRank].userRank, isUseRankPort_);
}

HcclResult TransportManager::CreateDestSockets(const std::string &tag, RankId remoteRank, u64 taskNum,
    std::vector<std::shared_ptr<HcclSocket> > &connectSockets, bool &isInterRdma, bool forceRdma)
{
    HcclResult ret = HCCL_SUCCESS;

    UpdateIsInterRdma(remoteRank, isInterRdma, forceRdma);

    u32 socketsPerLink = 1;
    if (isInterRdma) {
        socketsPerLink = GetSocketsPerLink(taskNum);
    }

    HcclRankLinkInfo remoteLinkInfo;
    MakeRemoteLinkInfo(remoteRank, isInterRdma, socketsPerLink, remoteLinkInfo);

    std::string newTag;
    CHK_RET(ConstructTransTag(tag, newTag, isInterRdma));

    if (isInterRdma || Is310PDevice()) {
        HcclNetDevCtx &netDevCtx = nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE ?
            netDevCtxMap_[devIpAddr_[0]]: netDevCtxMap_[hostIp_];
        ret = socketManager_->CreateSingleLinkSocket(newTag, netDevCtx,
            remoteLinkInfo, connectSockets, false, false);
        if (!GetExternalInputHcclIsTcpMode()) {
            std::vector<std::string>::iterator iter = std::find(socketTagVec_.begin(), socketTagVec_.end(), newTag);
            if (iter == socketTagVec_.end()) {
                socketTagVec_.push_back(newTag);
            }
        }
    } else {
        if (rankInfoList_[userRank_].deviceType == DevType::DEV_TYPE_310P3) {
            std::vector<u32> enableP2PDevices;
            enableP2PDevices.push_back(rankInfoList_[remoteRank].devicePhyId);
            HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Create][DestSockets]Enable P2P Failed, src devicePhyId[%d], dst devicePhyId[%d], ret[%u]",
                rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, ret), ret);

            enableP2PDevices_.push_back(rankInfoList_[remoteRank].devicePhyId);
        }
        // server内非异构场景，使能P2P
        if (!isHaveCpuRank_) {
            std::vector<u32> WaitP2PEnabledDevices;
            WaitP2PEnabledDevices.push_back(rankInfoList_[remoteRank].devicePhyId);
            HcclResult ret = P2PMgmtPub::WaitP2PEnabled(WaitP2PEnabledDevices);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Create][DestSockets]Enable P2P Failed, ret[%u]", ret), ret);
        }
        ret = socketManager_->CreateSingleLinkSocket(newTag, netDevCtxMap_[localVnicIp_],
            remoteLinkInfo, connectSockets, false, true);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Create][DestSockets]Create single link sockets failed, "
            "local rank[%u], remote rank[%u], isInterRdma[%d]",
            userRank_, remoteRank, isInterRdma), ret);
    return ret;
}

u32 TransportManager::GetSocketsPerLink(u64 taskNum)
{
    if (GetExternalInputQpsPerConnection() != HCCL_QPS_PER_CONNECTION_DEFAULT &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return 2; // 2：多QP方式下额外创建一个socket用于同步QP状态迁移完成状态
    }
    u32 socketsPerLink = 1;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (taskNum == 0) {
            taskNum = 1;
        }
        socketsPerLink = (taskNum + (HCCP_SQ_TEMPLATE_CAPACITY - 1)) / HCCP_SQ_TEMPLATE_CAPACITY;
    }
    return socketsPerLink;
}

HcclResult TransportManager::CreateLink(const std::string &tag, const ErrContextPub &error_context,
    const MachineType machineType, const std::string &serverId, const u32 remoteRank,
    const bool supportDataReceivedAck, const LinkMode linkMode,
    const bool enableUseOneDoorbell, const std::string threadStr,
    const std::vector<std::shared_ptr<HcclSocket> > sockets,
    const DeviceMem inputMem, const DeviceMem outputMem, bool isUsedRdma,
    std::shared_ptr<Transport> &link)
{
    hrtErrMSetErrorContextPub(error_context);
    // 给当前线程添加名字
    s32 sRet = pthread_setname_np(pthread_self(), threadStr.c_str());
    if (sRet != 0) {
        HCCL_WARNING("err[%d] link[%s] nameSet failed.", sRet, threadStr.c_str());
    }
    link = nullptr;
    CHK_RET(hrtSetDevice(deviceLogicId_));

    MachinePara machinePara;
    CHK_RET(SetMachinePara(tag, machineType, serverId, remoteRank, supportDataReceivedAck, linkMode, sockets,
        inputMem, outputMem, machinePara));
    HCCL_DEBUG("inputMem[%p],outputMem[%p], inputMem size[%llu], outputMem size[%llu]", inputMem.ptr(), outputMem.ptr(),
        inputMem.size(), outputMem.size());
    HCCL_INFO("[creakLink para]rank[%u]-localUserrank[%u]-localIpAddr[%s], linkMode[%d] "
              "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], machineType[%d], serverId[%s], nicDeploy[%d] ",
        userRank_, rankInfoList_[userRank_].worldRank, rankInfoList_[userRank_].serverId.c_str(), machinePara.linkMode,
        remoteRank, rankInfoList_[remoteRank].worldRank, rankInfoList_[remoteRank].serverId.c_str(),
        machinePara.machineType, machinePara.serverId.c_str(), machinePara.nicDeploy);
    // transport初始化
    HcclResult ret = TransportInit(remoteRank, machinePara, link, enableUseOneDoorbell, isUsedRdma);
    if (ret != HCCL_SUCCESS) {
        link = nullptr;
        const std::string  CREATE_LINK_ERR = "[Create][DestLink]Create Dest error! creakLink para:rank[" +
            std::to_string(userRank_) + "]-localUserrank[" + std::to_string(rankInfoList_[userRank_].worldRank) +
            "]-localIpAddr[" + rankInfoList_[userRank_].serverId.c_str() + "], dst_rank[" +
            std::to_string(remoteRank) + "]-remoteUserrank[" + std::to_string(rankInfoList_[remoteRank].worldRank) +
            "]-remote_ip_addr[" + rankInfoList_[remoteRank].serverId.c_str() + "]";

        RPT_INPUT_ERR(true, "EI0009", std::vector<std::string>({"reason"}),
            std::vector<std::string>({CREATE_LINK_ERR}));
        HCCL_ERROR("[Create][DestLink]Transport init error! creakLink para:rank[%u]-localUserrank[%u]-localIpAddr[%s], "
                   "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], machineType[%d], serverId[%s], linkMode[%d], "
                   "tag[%s]",
            userRank_, rankInfoList_[userRank_].worldRank, rankInfoList_[userRank_].serverId.c_str(), remoteRank,
            rankInfoList_[remoteRank].worldRank, rankInfoList_[remoteRank].serverId.c_str(),
            machinePara.machineType, machinePara.serverId.c_str(), machinePara.linkMode,
            machinePara.tag.c_str());
        return ret;
    }
    HCCL_INFO(
        "[creakLink success]:rank[%u]-localUserrank[%u]-localIpAddr[%s], "
        "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], tag[%s]", userRank_,
        rankInfoList_[userRank_].worldRank, rankInfoList_[userRank_].serverId.c_str(), remoteRank,
        rankInfoList_[remoteRank].worldRank, rankInfoList_[remoteRank].serverId.c_str(),
        machinePara.tag.c_str());

    return HCCL_SUCCESS;
}

HcclResult TransportManager::SetMachinePara(const std::string &tag, MachineType machineType,
    const std::string &serverId, u32 dstRank,
    const bool supportDataReceivedAck, const LinkMode linkMode,
    const std::vector<std::shared_ptr<HcclSocket> > &socketList,
    const DeviceMem &inputMem, const DeviceMem &outputMem, MachinePara &machinePara)
{
    machinePara.linkMode = linkMode;
    machinePara.machineType = machineType;
    machinePara.serverId = serverId;
    machinePara.localIpAddr = rankInfoList_[userRank_].nicIp[0];
    machinePara.remoteIpAddr = rankInfoList_[dstRank].nicIp[0];
    machinePara.localUserrank = rankInfoList_[userRank_].userRank;
    machinePara.remoteUserrank = rankInfoList_[dstRank].userRank;
    machinePara.localWorldRank = rankInfoList_[userRank_].worldRank;
    machinePara.remoteWorldRank = rankInfoList_[dstRank].worldRank;
    machinePara.collectiveId = identifier_;
    machinePara.localDeviceId = rankInfoList_[userRank_].devicePhyId;
    machinePara.remoteDeviceId = rankInfoList_[dstRank].devicePhyId;
    machinePara.deviceType = static_cast<DevType>(rankInfoList_[dstRank].deviceType);
    machinePara.inputMem = inputMem;
    machinePara.outputMem = outputMem;
    machinePara.linkAttribute = 0x03; /* 0x03同时支持目的端和源端发起 */
    machinePara.tag = tag;
    // 把原来的两层vector变成一层, 方便后继调用
    if (socketList.size() > 0) {
        std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > socketsMap;
        socketsMap[dstRank] = socketList;
        std::map<u32, u32> dstRankToUserRank;
        dstRankToUserRank[dstRank] = dstRank;
        CHK_RET(socketManager_->WaitLinksEstablishCompleted(socketList[0]->GetLocalRole(),
            socketsMap, dstRankToUserRank));
        machinePara.sockets = socketList;
    }

    machinePara.supportDataReceivedAck = supportDataReceivedAck; /* NeedDataReceivedAck(); */
    machinePara.nicDeploy = nicDeployment_;
    machinePara.localSocketPort = rankInfoList_[userRank_].hostPort;
    machinePara.remoteSocketPort = rankInfoList_[dstRank].hostPort;
    machinePara.deviceLogicId = deviceLogicId_;
    machinePara.udpSport = 0x0; /* 0代表默认不配置 */
    return HCCL_SUCCESS;
}

TransportType TransportManager::GetTransportType(const u32 dstRank, bool isUsedRdma)
{
    TransportType transportType;
    // 判断是否在同一个server
    if (rankInfoList_[userRank_].serverId == rankInfoList_[dstRank].serverId) {
        if (isHaveCpuRank_) {
            transportType = TransportType::TRANS_TYPE_HETEROG_P2P;
        } else {
            LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
            hrtGetPairDeviceLinkType(rankInfoList_[userRank_].devicePhyId, rankInfoList_[dstRank].devicePhyId,
                linkType);
            if (linkType == LinkTypeInServer::SIO_TYPE && GetExternalInputEnableRdmaSdmaConcurrent() && isUsedRdma
                && rankInfoList_[userRank_].deviceType == DevType::DEV_TYPE_910_73) {
                transportType = TransportType::TRANS_TYPE_P2P;
            // Server内判断是否使用rdma
            } else if (isUsedRdma) {
                transportType = TransportType::TRANS_TYPE_IBV_EXP;
            } else {
                transportType = TransportType::TRANS_TYPE_P2P;
            }
        }
    } else { // server间
        if (GetExternalInputHcclIsTcpMode()) {
            transportType = TransportType::TRANS_TYPE_HOST_TCP;
        } else if ((static_cast<DevType>(rankInfoList_[dstRank].deviceType) == DevType::DEV_TYPE_310P3) ||
            (static_cast<DevType>(rankInfoList_[dstRank].deviceType) == DevType::DEV_TYPE_310P1)) {
            transportType = TransportType::TRANS_TYPE_ROCE;
        } else if (isHaveCpuRank_) {
            transportType = TransportType::TRANS_TYPE_HETEROG_ROCE;
        } else if (IsSupportInterHccs(dstRank)) {
            // 超节点内节点间走HCCS通信
            transportType = TransportType::TRANS_TYPE_P2P;
        } else {
            transportType = TransportType::TRANS_TYPE_IBV_EXP;
        }
    }

    HCCL_INFO("SetTransportType: dstRank[%u] transport_type[%d]", dstRank, transportType);
    return transportType;
}

void TransportManager::SetTransportParam(TransportPara &para, MachinePara &machinePara)
{
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(
        GetExternalInputHcclLinkTimeOut());
    para.timeout = kdefaultTimeout;
    para.transportResourceInfoAddr = transportResourceInfoAddr_;
    para.transportResourceInfoSize = transportResourceInfoSize_;
    para.virtualFlag = false;
}

HcclResult TransportManager::TransportInit(const u32 dstRank, MachinePara &machinePara,
    std::shared_ptr<Transport> &link, bool useOneDoorbell, bool isUsedRdma)
{
    // 实例化TransportBase
    TransportPara para{};
    SetTransportParam(para, machinePara);

    TransportType type = GetTransportType(dstRank, isUsedRdma);
    if (type == TransportType::TRANS_TYPE_P2P) {
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_IBV_EXP) {
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_HOST_TCP) {
        para.nicDeploy = nicDeployment_;
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_ROCE) {
        para.selfIp = &machinePara.localIpAddr;
        para.peerIp = &machinePara.remoteIpAddr;
        std::set<u32> listenedPort;
        CHK_SMART_PTR_NULL(socketManager_);
        CHK_RET(socketManager_->GetListenPortByIp(NICDeployment::NIC_DEPLOYMENT_DEVICE, *(para.selfIp),
            listenedPort));
        para.peerPort = *(listenedPort.begin());
        para.selfPort = para.peerPort;
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_HETEROG_P2P) {
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_HETEROG_ROCE) {
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else {
        HCCL_ERROR("[Init][Transport]not supported transport type");
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PRT_RET(!link, HCCL_ERROR("[Init][Transport]In create link, new link failed"), HCCL_E_PTR);

    if (useOneDoorbell) {
        link->EnableUseOneDoorbell();
    }

    CHK_RET(link->Init());

    return HCCL_SUCCESS;
}

bool TransportManager::IsSupportInterHccs(const u32 dstRank)
{
    // 仅判断超节点内, 兼容打平通信域同时有server内和server间, 因此不判断server_id
    bool isInterHccs = GetExternalInputInterHccsDisable() == false &&
        rankInfoList_[userRank_].deviceType == DevType::DEV_TYPE_910_73 &&
        rankInfoList_[userRank_].superPodId.empty() == false &&
        rankInfoList_[userRank_].superPodId == rankInfoList_[dstRank].superPodId;

    HCCL_INFO("[IsSupportInterHccs]rank[%u], superPodId[%s], dstRank[%u], dstSuperPodId[%s], isInterHccs[%d]",
        userRank_, rankInfoList_[userRank_].superPodId.c_str(), dstRank,
        rankInfoList_[dstRank].superPodId.c_str(), isInterHccs);
    return isInterHccs;
}

void TransportManager::UpdateIsInterRdma(const u32 remoteRank, bool &isInterRdma, bool forceRdma) // 待确认判断是否完善
{
    // 超节点内节点间采用HCCS通信的, 放至dstIntraClientVec_, 采用p2p建链
    bool isInterHccs = IsSupportInterHccs(remoteRank);
    bool isConcurrent = GetExternalInputEnableRdmaSdmaConcurrent();
    if (isInterHccs && !isConcurrent) {
        isInterRdma = false;
        return;
    }
    LinkTypeInServer linkType;
    hrtGetPairDeviceLinkType(rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, linkType);
    if (isConcurrent && forceRdma && rankInfoList_[userRank_].deviceType == DevType::DEV_TYPE_910_73) {
        if (linkType == LinkTypeInServer::SIO_TYPE) {
            isInterRdma = false;
        } else {
            isInterRdma = true;
        }
    } else {
        isInterRdma = rankInfoList_[userRank_].serverId != rankInfoList_[remoteRank].serverId ||
            (isUsedRdmaOuter_ && linkType == LinkTypeInServer::PXI_TYPE) ||
            (rankInfoList_[userRank_].serverId == rankInfoList_[remoteRank].serverId && forceRdma);
    }
}

u32 TransportManager::GetInterRemotePort(s32 devicePhyId, u32 dstUserRank)
{
    if (!GetExternalInputHcclDeviceNicDisable() && (!ranksPort_.size() ||
        ranksPort_[dstUserRank] == HCCL_INVALIED_IF_BASE_PORT || (!isUseRankPort_ && !Is310PDevice()))) {
        HCCL_INFO("[GetInterRemotePort] port[%u]", HETEROG_CCL_PORT);
        return HETEROG_CCL_PORT;
    } else if (isUseRankPort_ && ranksPort_.size()) {
        // 此处应使用RankInfo中的userRank而非算法的rank
        return ranksPort_[dstUserRank];
    } else if (GetExternalInputHcclIfBasePort() == HCCL_INVALIED_IF_BASE_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

HcclResult TransportManager::MakeRemoteLinkInfo(const u32 remoteRank, bool isInterRdma,
    u32 socketsPerLink, HcclRankLinkInfo &remoteLinkInfo)
{
    RankInfo dstRankInfo = rankInfoList_[remoteRank];
    remoteLinkInfo.userRank = dstRankInfo.userRank;
    remoteLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
    if (isInterRdma) {
        remoteLinkInfo.ip = dstRankInfo.nicIp[0];
        remoteLinkInfo.port = GetInterRemotePort(remoteLinkInfo.devicePhyId, dstRankInfo.userRank);
        remoteLinkInfo.socketsPerLink = socketsPerLink;
    } else {
        remoteLinkInfo.ip = HcclIpAddress(dstRankInfo.devicePhyId);
        if (useSdidForDeviceId_) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(rankInfoList_[userRank_].devicePhyId,
                DeviceIdType::DEVICE_ID_TYPE_SDID,
                rankInfoList_[remoteRank].superDeviceId,
                remoteLinkInfo.ip));
        } else {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(rankInfoList_[userRank_].devicePhyId,
                DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                rankInfoList_[remoteRank].devicePhyId,
                remoteLinkInfo.ip));
        }
        remoteLinkInfo.port = GetRemoteNicPort(remoteRank); // ?
        remoteLinkInfo.socketsPerLink = socketsPerLink;
    }

    return HCCL_SUCCESS;
}

}  // namespace hccl
