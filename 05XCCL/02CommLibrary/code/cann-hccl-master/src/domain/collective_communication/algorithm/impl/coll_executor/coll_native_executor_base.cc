/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_native_executor_base.h"
#include "profiling_manager_pub.h"
namespace hccl {

CollNativeExecutorBase::CollNativeExecutorBase(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollExecutorBase(dispatcher, topoMatcher), topoAttr_(topoMatcher_->GetTopoInfo()),
      algoAttr_(topoMatcher_->GetAlgoInfo())
{
    topoType_ = topoAttr_.topoType;
    is310P3Common_ = topoAttr_.is310P3Common;
}

void CollNativeExecutorBase::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    root_ = param.root;
}

bool CollNativeExecutorBase::CheckNeedRecreateComm(u64 lastScratchMemSize)
{
    return false;
}

// ----------------------资源计算接口----------------------
HcclResult CollNativeExecutorBase::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    (void)ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    bool needAivBuffer = false;
    std::vector<LevelNSubCommTransport> opTransport {
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))
    };

    CHK_RET(CalcScratchMemSize(scratchMemSize));
    CHK_RET(CalcStreamNum(streamNum));
    CHK_RET(CalcNotifyNum(streamNum, notifyNum));
    CHK_RET(GetIfNeedAivBuffer(needAivBuffer));
    CHK_RET(CalcCommInfo(opTransport));

    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, needAivBuffer, opTransport, resourceRequest));
    HCCL_INFO("streamNum[%u], notifyNum[%u], sctrachMemSize[%llu], needAivBuffer[%u]",
        resourceRequest.streamNum, resourceRequest.notifyNum, resourceRequest.scratchMemSize,
        resourceRequest.needAivBuffer);
    // 打印建链诉求
    PrintTransportRequest(resourceRequest);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = 0U;
    HCCL_INFO("[CollNativeExecutorBase][CalcScratchMemSize]tag[%s] scratchMemSize_ is [%llu]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcStreamNum(u32& streamNum)
{
    // 只传递从流数量
    streamNum = 0;
    HCCL_INFO("[CollNativeExecutorBase][CalcStreamNum]tag[%s] streamNum_ is [%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcNotifyNum(u32 streamNum, u32 &notifyNum)
{
    // notify数量是从流的两倍
    notifyNum = 2U * streamNum;
    HCCL_INFO("[CollNativeExecutorBase][CalcNotifyNum]tag[%s] notifyNum_ is [%u]", tag_.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    // 非AIV通信不需要AIV buffer
    needAivBuffer = false;
    HCCL_INFO("[CollNativeExecutorBase][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u]", tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcCommPlaneInfo(const std::string &tag, const CommParaInfo &commParaInfo,
    std::vector<SingleSubCommTransport> &commTransport, TransportMemType inPutMemType,
    TransportMemType outPutMemType)
{
    return topoMatcher_->CalcCommPlaneInfo(tag, commParaInfo, commTransport, inPutMemType, outPutMemType);
}

HcclResult CollNativeExecutorBase::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollNativeExecutorBase][CalcInnerCommInfo]tag[%s]start", tag_.c_str());

    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MAX, root_);
    if (UseInterServerRingAlgo(algType_)) {
        commParaLevel1.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[CollNativeExecutorBase][CalcInnerCommInfo]tag[%s] Calc RingCommInfo", tag_.c_str());
    } else if (UseInterServerNHRAlgo(algType_)) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[CollNativeExecutorBase][CalcInnerCommInfo]tag[%s] Calc NHRCommInfo", tag_.c_str());
    } else if (UseInterServerNHRV1Algo(algType_)) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
        HCCL_INFO("[CollNativeExecutorBase][CalcInnerCommInfo]tag[%s] Calc NHRV1CommInfo", tag_.c_str());
    } else if (UseInterServerNBAlgo(algType_)) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[CollNativeExecutorBase][CalcInnerCommInfo]tag[%s] Calc NBCommInfo", tag_.c_str());
    } else {
        commParaLevel1.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[CollNativeExecutorBase][CalcInnerCommInfo]tag[%s] Calc HDCommInfo", tag_.c_str());
    }
    commParaLevel1.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_LEVEL1], inputType, outputType));
    if (topoMatcher_->GetExternalInputEnableRdmaSdmaConcurrent() && UseInterServerRingAlgo(algType_)) {
        CommParaInfo commParaLevel1Rdma(COMM_LEVEL1_RDMA, CommType::COMM_TAG_RING_INNER);
        commParaLevel1Rdma.forceRdma = true;
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1Rdma, opTransport[COMM_LEVEL1_RDMA], inputType,
        outputType));
    }
    HCCL_INFO("[CollNativeExecutorBase][CalcInnerCommInfo]tag[%s] Calc CommInfo Finish", tag_.c_str());

    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::PrintTransportRequest(AlgResourceRequest& resourceRequest)
{
    for (u32 levelIndex = 0; levelIndex < COMM_LEVEL_RESERVED; levelIndex++) {
        LevelNSubCommTransport &levelTransport = resourceRequest.opTransport[levelIndex];
        u32 ringSize = levelTransport.size();
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            SingleSubCommTransport &subCommTransport = levelTransport[ringIndex];
            u32 rankSize = subCommTransport.transportRequests.size();
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                if (subCommTransport.transportRequests[rankIndex].isValid == true) {
                    HCCL_INFO("[CollNativeExecutorBase][CalcResRequest]" \
                        "levelIndex[%u], ringIndex[%u], rankIndex[%u], userRank[%u], remoteRank[%u]",
                        levelIndex, ringIndex, rankIndex, subCommTransport.transportRequests[rankIndex].localUserRank,
                        subCommTransport.transportRequests[rankIndex].remoteUserRank);
                }
            }
        }
    }
    return HCCL_SUCCESS;
}
// ----------------------算法编排接口----------------------
HcclResult CollNativeExecutorBase::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_WARNING("[CollNativeExecutorBase][KernelRun]Using the default kernel run, nothing is done.");
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::GetStreamInfo(const AlgResourceResponse &algRes)
{
    u32 slaveStreamNum = algRes.streams.size();
    HCCL_INFO("[CollNativeExecutorBase]algRes.streams.size()[%d]", slaveStreamNum);
    if (slaveStreamNum == 0) {
        HCCL_INFO("[CollNativeExecutorBase][GetStreamInfo]tag[%s], slave stream is empty.", tag_.c_str());
    }
    streamInfo_.ringNum = slaveStreamNum + 1;
    streamInfo_.ringStreams = algRes.streams;
    streamInfo_.ringSignal.resize(slaveStreamNum);
    streamInfo_.ringSignalAux.resize(slaveStreamNum);
    for (u32 i = 0; i < slaveStreamNum; i++) {
        streamInfo_.ringSignal[i] = algRes.notifies[2 * i];
        streamInfo_.ringSignalAux[i] = algRes.notifies[2 * i + 1];
    }
    HCCL_DEBUG("[CollNativeExecutorBase][GetStreamInfo]tag[%s], ringNum[%u], ringStream.size()[%u], " \
        "ringSignal.size()[%u], ringSingalAux.size()[%u]", tag_.c_str(), streamInfo_.ringNum,
        streamInfo_.ringStreams.size(), streamInfo_.ringSignal.size(), streamInfo_.ringSignalAux.size());
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::ActiveSlaveStreams(const Stream &stream)
{
    HcclResult ret = HCCL_SUCCESS;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
        for (u32 streamIndex = 0; streamIndex < streamInfo_.ringStreams.size(); streamIndex++) {
            ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                streamInfo_.ringStreams[streamIndex].ptr(), stream.ptr());
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollNativeExecutorBase][ActiveSlaveStreams]tag[%s], stream[%u] active failed,return[%d]",
                tag_.c_str(), streamIndex, ret), ret);
        }
    }
    return ret;
}

HcclResult CollNativeExecutorBase::AddSubStreamToProfiling()
{
    if (((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        hccl::ProfilingManagerPub::GetAddtionInfoState() &&
        hccl::ProfilingManagerPub::GetTaskApiState())) {
        return HCCL_SUCCESS;
    }

    for (u32 streamIndex = 0; streamIndex < streamInfo_.ringStreams.size(); streamIndex++) {
        // profiling加入从环的stream
        HCCL_PROFILER_ADD_STREAM(streamInfo_.ringStreams[streamIndex].ptr(), tag_, streamIndex + 1, algType_);
    }
    return HCCL_SUCCESS;
}


HcclResult CollNativeExecutorBase::CheckCommSize(const CommPlane levelIndex, const u32 expectedSize)
{
    if (algResResp_->opTransportResponse[levelIndex].size() < expectedSize) {
        HCCL_ERROR("[CollNativeExecutorBase][CheckCommSize]tag[%s], levelIndex[%u], " \
            "ring size[%u] is less than expected[%u]",
            tag_.c_str(), levelIndex, algResResp_->opTransportResponse[levelIndex].size(), expectedSize);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

SubCommInfo CollNativeExecutorBase::GetSubCommInfo(const CommPlane levelIndex, const u32 subLevelIndex)
{
    SubCommInfo info;
    SingleSubCommTransport &transportInfo =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[levelIndex][subLevelIndex]);
    info.localRank = transportInfo.userRank2subCommRank[topoAttr_.userRank];
    info.localRankSize = transportInfo.transportRequests.size();
    info.links = transportInfo.links;
    info.virtualLinks = transportInfo.virtualLinks;
    return info;
}

// ----------------------工具接口----------------------

AlgTypeLevel0 CollNativeExecutorBase::GetLevel0AlgType(const AlgType algType) const
{
    const u32 algLevel0 = static_cast<u32>(algType) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1);
    return static_cast<AlgTypeLevel0>(algLevel0);
}

AlgTypeLevel1 CollNativeExecutorBase::GetLevel1AlgType(const AlgType algType) const
{
    const u32 algLevel1 = static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH;
    return static_cast<AlgTypeLevel1>(algLevel1);
}

bool CollNativeExecutorBase::UseInterServerRingAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_RING;
}

bool CollNativeExecutorBase::UseInterServerHDAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_HD;
}

bool CollNativeExecutorBase::UseInterServerNHRAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NHR;
}

bool CollNativeExecutorBase::UseInterServerNHRV1Algo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
}

bool CollNativeExecutorBase::UseInterServerNBAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NB;
}

bool CollNativeExecutorBase::UseLevel2RingAlgo(AlgType algType)
{
    return GetLevel2AlgType(algType) == AlgTypeLevel2::ALG_LEVEL2_RING;
}

bool CollNativeExecutorBase::UseInterServerPipelineAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
}

AlgTypeLevel2 CollNativeExecutorBase::GetLevel2AlgType(const AlgType algType) const
{
    const u32 algLevel2 = static_cast<u32>(algType) >> (HCCL_LEVEL_ALGO_WIDTH * 2);
    return static_cast<AlgTypeLevel2>(algLevel2);
}

HcclResult CollNativeExecutorBase::BuildResourceRequest(u64 scratchMemSize, u32 streamNum, u32 notifyNum,
    bool needAivBuffer, std::vector<LevelNSubCommTransport>& opTransport, AlgResourceRequest& resourceRequest)
{
    resourceRequest.scratchMemSize = scratchMemSize;
    resourceRequest.streamNum = streamNum;
    resourceRequest.notifyNum = notifyNum;
    resourceRequest.needAivBuffer = needAivBuffer;
    resourceRequest.opTransport = opTransport;
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::GetRankByUserRank(CommPlane levelIndex, u32 subLevelIndex, u32 userRank, u32 &rank)
{
    CHK_RET(CheckCommSize(levelIndex, subLevelIndex + 1));
    SingleSubCommTransport &transportInfo =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[levelIndex][subLevelIndex]);
    rank = transportInfo.userRank2subCommRank[userRank];
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::GetUserRankByRank(CommPlane levelIndex, u32 subLevelIndex, u32 rank, u32 &userRank)
{
    CHK_RET(CheckCommSize(levelIndex, subLevelIndex + 1));
    SingleSubCommTransport &transportInfo =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[levelIndex][subLevelIndex]);
    userRank = transportInfo.subCommRank2UserRank[rank];
    return HCCL_SUCCESS;
}
}
