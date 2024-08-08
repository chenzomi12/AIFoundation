/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_alg.h"
#include "hccl_impl.h"
#include "scatter_operator.h"
#include "reduce_operator.h"
#include "all_reduce_operator.h"
#include "reduce_scatter_operator.h"
#include "broadcast_operator.h"
#include "gather_operator.h"
#include "all_gather_operator.h"
#include "send_receive_operator.h"
#include "alltoall_operator.h"
#include "coll_alg_op_registry.h"
#include "topo_matcher.h"
namespace hccl {

HcclAlg::HcclAlg()
{
}

HcclAlg::~HcclAlg()
{
    pimpl_ = nullptr;
}

HcclResult HcclAlg::Init(const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
    std::unique_ptr<WorkspaceResource> &workSpaceRes, CCLBufferManager &cclBufferManager,
    const HcclDispatcher dispatcher, const HcclDispatcher vDispatcher,
    const std::unique_ptr<NotifyPool> &notifyPool, std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
    const std::unique_ptr<QueueNotifyManager> &queueNotifyManager,
    HcclAlgoAttr &algoAttr, HcclTopoAttr &topoAttr, bool isHeterogComm)
{
    CHK_RET(InitAlgoInfo(algoAttr));
    CHK_RET(InitTopoInfoPartOne(topoAttr));
    pimpl_.reset((new (std::nothrow) hcclImpl(dispatcher, vDispatcher, notifyPool, netDevCtxMap, queueNotifyManager,
        workSpaceRes, cclBufferManager, transportResourceInfoAddr, transportResourceInfoSize, algoAttr, topoAttr)));
    CHK_SMART_PTR_NULL(pimpl_);
    CHK_RET(pimpl_->Init(isHeterogComm));
    std::vector<std::vector<std::vector<u32>>> CommPlaneRanks;
    CHK_RET(pimpl_->GetCommPlaneRanks(CommPlaneRanks));
    std::vector<bool> isBridgeVector;
    CHK_RET(pimpl_->GetIsBridgeVector(isBridgeVector));
    CHK_RET(InitTopoInfoPartTwo());
    CHK_RET(InitExternalEnable());
    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
    serverAndsuperPodToRank.clear();
    CHK_RET(pimpl_->GetRankVecInfo(serverAndsuperPodToRank));

    topoMatcher_.reset((new (std::nothrow) TopoMatcher(CommPlaneRanks, isBridgeVector,
                                                       topoInfo_, algoInfo_, externalEnable_,
                                                       serverAndsuperPodToRank)));
    return HCCL_SUCCESS;
}
// 上层保证，以下方法在初始化成功后才会调用，所以未对pimpl_进行保护判断
HcclResult HcclAlg::ReleaseCommInfos()
{
    return pimpl_->ReleaseCommInfos();
}

std::unique_ptr<CollAlgOperator> HcclAlg::GetAlgOperator(const HcclCMDType &opType)
{
    if (!pimpl_) {
        HCCL_ERROR("[HcclAlg][GetAlgOperator] impl ptr is null, get algorithm operator failed.");
        return nullptr;
    }
    if (!topoMatcher_) {
        HCCL_ERROR("[HcclAlg][GetAlgOperator] topoMatcher ptr is null, get algorithm operator failed.");
        return nullptr;
    }
    return CollAlgOpRegistry::Instance()->GetAlgOp(opType, pimpl_, topoMatcher_);
}

HcclResult HcclAlg::AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, Stream stream, HcomCollOpInfo *opInfo)
{
    AllGatherOperator operation(pimpl_, topoMatcher_);
    return operation.AllGather(tag, inputPtr, outputPtr, inputCount, dataType, stream, opInfo);
}

HcclResult HcclAlg::AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    AllGatherOperator operation(pimpl_, topoMatcher_);
    return operation.AllGatherOutPlace(tag, inputPtr, outputPtr, inputCount, dataType, stream, opBaseAtraceInfo);
}

HcclResult HcclAlg::Broadcast(
    const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root, Stream stream)
{
    BroadCastOperator operation(pimpl_, topoMatcher_);
    return operation.Broadcast(tag, ptr, count, dataType, root, stream);
}

HcclResult HcclAlg::BroadcastOutPlace(
    const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root, Stream stream,
    const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    BroadCastOperator operation(pimpl_, topoMatcher_);
    return operation.BroadcastOutPlace(tag, ptr, count, dataType, root, stream);
}

HcclResult HcclAlg::Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, Stream stream)
{
    ScatterOperator operation(pimpl_, topoMatcher_);
    return operation.Scatter(tag, inputPtr, outputPtr, recvCount, dataType, root, stream);
}

HcclResult HcclAlg::ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, Stream stream,
    const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    ScatterOperator operation(pimpl_, topoMatcher_);
    return operation.ScatterOutPlace(tag, inputPtr, outputPtr, recvCount, dataType, root, stream);
}

HcclResult HcclAlg::Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType,
    HcclReduceOp op, u32 root, Stream stream)
{
    ReduceOperator operation(pimpl_, topoMatcher_);
    return operation.Reduce(tag, inputPtr, outputPtr, count, dataType, op, root, stream);
}

HcclResult HcclAlg::ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream,
    const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    ReduceOperator operation(pimpl_, topoMatcher_);
    return operation.ReduceOutPlace(tag, inputPtr, outputPtr, count, dataType, op, root, stream);
}

HcclResult HcclAlg::Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType, u32 destRank,
    Stream stream)
{
    SendReceiveOperator operation(pimpl_, topoMatcher_);
    return operation.Send(tag, inputPtr, count, dataType, destRank, stream);
}

HcclResult HcclAlg::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, Stream stream)
{
    SendReceiveOperator operation(pimpl_, topoMatcher_);
    return operation.SendOutPlace(tag, inputPtr, count, dataType, destRank, stream);
}

HcclResult HcclAlg::Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, Stream stream)
{
    SendReceiveOperator operation(pimpl_, topoMatcher_);
    return operation.Receive(tag, outputPtr, count, dataType, srcRank, stream);
}

HcclResult HcclAlg::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, Stream stream)
{
    SendReceiveOperator operation(pimpl_, topoMatcher_);
    return operation.ReceiveOutPlace(tag, outputPtr, count, dataType, srcRank, stream);
}

HcclResult HcclAlg::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
    HcclDataType dataType, Stream stream)
{
    GatherOperator operation(pimpl_, topoMatcher_);
    return operation.Gather(tag, inputPtr, outputPtr, rootRank, inputCount, dataType, stream);
}

HcclResult HcclAlg::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
    u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize)
{
    AlltoAllOperator operation(pimpl_, topoMatcher_);
    return operation.GetAlltoAllStagedWorkSpaceMemSize(
        sendCounts, sdispls, sendType, recvCounts, rdispls, recvType, memSize);
}

HcclResult HcclAlg::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
{
    AlltoAllOperator operation(pimpl_, topoMatcher_);
    return operation.GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
}

HcclResult HcclAlg::GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize)
{
    AllReduceOperator operation(pimpl_, topoMatcher_);
    return operation.GetAllReduceScratchSize(count, dataType, scratchSize);
}

HcclResult HcclAlg::ClearOpResource(const std::string &tag)
{
    return pimpl_->ClearOpResource(tag);
}

bool HcclAlg::IsExistCommRes(const std::string &tag)
{
    return pimpl_->IsExistCommRes(tag);
}

HcclResult HcclAlg::CreateMutiStreamRes(const std::string &tag, Stream &stream, innerStreamInfo_t &streamInfo,
    AlgType algType, bool isAicpuModeEn)
{
    return pimpl_->CreateMutiStreamRes(tag, stream, streamInfo, algType, isAicpuModeEn);
}

HcclResult HcclAlg::CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
    std::unique_ptr<CommInfo> &commInfo, u32 root, bool isP2p, bool isAicpuModeEn)
{
    return pimpl_->CreateComm(tag, inputMem, outputMem, algType, commInfo, root, isP2p, isAicpuModeEn);
}

HcclResult HcclAlg::CreateComm(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType, u32 root, bool isP2p)
{
    return pimpl_->CreateComm(tag, inputMem, outputMem, algType, root, isP2p);
}

HcclResult HcclAlg::CreateP2PCommQuerry(const std::string &tag, u32 &status)
{
    return pimpl_->CreateP2PCommQuerry(tag, status);
}

HcclResult HcclAlg::CreateP2PCommAsync(const std::string &tag, DeviceMem &mem, u32 peerRank, u32 &status)
{
    return pimpl_->CreateP2PCommAsync(tag, mem, peerRank, status);
}

void HcclAlg::CancelCommRes(const std::string &tag)
{
    pimpl_->CancelCommRes(tag);
}

void HcclAlg::Break()
{
    pimpl_->Break();
}

HcclResult HcclAlg::SetAlgType(AlgType algType, HcclCMDType opType)
{
    return pimpl_->SetAlgType(algType, opType);
}

HcclResult HcclAlg::GetAlgType(AlgType &algType, HcclCMDType opType)
{
    return pimpl_->GetAlgType(algType, opType);
}

std::string HcclAlg::AlgTypeToStr(const AlgType algType)
{
    AlgTypeLevel1 algTypeLevel1 = AlgTypeLevel1(floor(static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH));
    AlgTypeLevel0 algTypeLevel0 = AlgTypeLevel0(static_cast<u32>(algType) -
        (static_cast<u32>(algTypeLevel1) << HCCL_LEVEL_ALGO_WIDTH));
    auto level0Iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algTypeLevel0);
    auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
    std::string algStrLevel0;
    std::string algStrLevel1;
    if (level0Iter == HCCL_ALGO_LEVEL0_NAME_MAP.end()) {
        algStrLevel0 = "invalid algo type";
    } else {
        algStrLevel0 = level0Iter->second;
    }
    if (level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        algStrLevel1 = "invalid algo type";
    } else {
        algStrLevel1 = level1Iter->second;
    }
    std::string algStr = "level0:" + algStrLevel0 + ",level1:" + algStrLevel1;
    return algStr;
}

HcclResult HcclAlg::SupportDeterministicOptim(bool &isDeterministicOptim)
{
    isDeterministicOptim = pimpl_->SupportDeterministicOptim();

    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
    std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    pimpl_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap, ranksPort, isSetHDCModeInfo, isUseRankPort);

    return HCCL_SUCCESS;
}

u8 HcclAlg::GetDeterministicConfig() const
{
    return topoMatcher_->GetDeterministicConfig();
}

HcclResult HcclAlg::SetDeterministicConfig(const u8 deterministic)
{
    CHK_RET(topoMatcher_->SetDeterministicConfig(deterministic));

    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetAlltoAllStatus(DeviceMem &tinySendRecvMem, bool &isAlltoAllZCopyMode)
{
    CHK_RET(pimpl_->GetAlltoAllStatus(tinySendRecvMem, isAlltoAllZCopyMode));

    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitExternalEnable()
{
    externalEnable_.enableRdmaSdmaConcurrent = GetExternalInputEnableRdmaSdmaConcurrent();
    externalEnable_.enableFfts = GetExternalInputHcclEnableFfts();
    externalEnable_.deterministic = GetExternalInputHcclDeterministic();
    externalEnable_.highPerfEnable = GetExternalInputHcclHighPerfEnable();
    externalEnable_.intraRoceSwitch = GetExternalInputIntraRoceSwitch();
    externalEnable_.dumpDebug = GetExternalInputHcclDumpDebug();

    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitTopoInfoPartOne(HcclTopoAttr &topoAttr)
{
    topoInfo_.userRank = topoAttr.userRank;
    topoInfo_.userRankSize = topoAttr.userRankSize;
    topoInfo_.devicePhyId = topoAttr.devicePhyId;
    topoInfo_.deviceLogicId = topoAttr.deviceLogicId;
    topoInfo_.nicList = topoAttr.nicList;
    topoInfo_.isSingleMeshAggregation = topoAttr.isSingleMeshAggregation;
    topoInfo_.deviceNumPerAggregation = topoAttr.deviceNumPerAggregation;
    topoInfo_.devNumInLevel2 = topoAttr.devNumInLevel2;
    topoInfo_.deviceType = topoAttr.deviceType;
    topoInfo_.serverNum = topoAttr.serverNum;
    topoInfo_.meshAggregationRankSize = topoAttr.meshAggregationRankSize;
    topoInfo_.multiModuleDiffDeviceNumMode = topoAttr.multiModuleDiffDeviceNumMode;
    topoInfo_.pairLinkCounter = topoAttr.pairLinkCounter;
    topoInfo_.isDiffDeviceModule = topoAttr.isDiffDeviceModule;
    topoInfo_.realUserRank = topoAttr.realUserRank;
    topoInfo_.moduleNum = topoAttr.moduleNum;

    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitTopoInfoPartTwo()
{
    TopoType topoType;
    CHK_RET(pimpl_->GetTopoType(topoType));
    topoInfo_.topoType = topoType;
    topoInfo_.is310P3Common = pimpl_->Is310P3Common();
    std::unordered_map<u32, bool> isUsedRdmaMap;
    CHK_RET(pimpl_->GetIsUsedRdmaMap(isUsedRdmaMap));
    topoInfo_.isUsedRdmaMap = isUsedRdmaMap;

    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitAlgoInfo(HcclAlgoAttr &algoAttr)
{
    algoInfo_.identifier = algoAttr.identifier;
    algoInfo_.inlineReduceSwitchOn = algoAttr.inlineReduceSwitchOn;
    algoInfo_.isUsedRdmaOuter = algoAttr.isUsedRdmaOuter;

    return HCCL_SUCCESS;
}
}
