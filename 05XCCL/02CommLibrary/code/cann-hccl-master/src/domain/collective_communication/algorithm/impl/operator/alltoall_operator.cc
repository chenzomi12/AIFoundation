/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_operator.h"
#include "device_capacity.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "all_gather_operator.h"
#include <vector>
#include "allltoall_pipeline_mesh_pairwise_ccl_enough_pub.h"
#include "allltoall_pipeline_mesh_pairwise_ping_pong_pub.h"
#include "coll_alg_exec_registry.h"
#include "coll_alg_op_registry.h"

namespace hccl {

AlltoAllOperator::AlltoAllOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(pImpl, topoMatcher, HcclCMDType::HCCL_CMD_ALLTOALL)
{
    hcclImpl_->GetAlltoAllStatus(tinySendRecvMem_, isAlltoAllZCopyMode_);
}

AlltoAllOperator::~AlltoAllOperator()
{
}

HcclResult AlltoAllOperator::CheckSendRecvParams(
    const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u32 rankSize = allMeshAggregationSendRecvInfo.size();
    for (u32 i = 0; i < rankSize; i++) {
        u32 sendsSize = allMeshAggregationSendRecvInfo[i].sendLength.size();
        u32 recvsSize = allMeshAggregationSendRecvInfo[i].recvLength.size();
        if (rankSize != sendsSize || rankSize != recvsSize) {
            HCCL_ERROR(
                "[AlltoAllV][CheckSendRecvParam] rankSize[%u], sendsSize[%u], recvsSize[%u] are not match Index[%u]",
                rankSize, sendsSize, recvsSize, i);
            return HCCL_E_PARA;
        }
        for (u32 j = 0; j < sendsSize; j++) {
            if (allMeshAggregationSendRecvInfo[i].sendLength[j] != allMeshAggregationSendRecvInfo[j].recvLength[i]) {
                HCCL_ERROR("SendLength[%u][%u]: %llu and recvLength[%u][%u]: %llu are not match", i, j,
                    allMeshAggregationSendRecvInfo[i].sendLength[j], j, i,
                    allMeshAggregationSendRecvInfo[j].recvLength[i]);
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllvcSendRecvInfo(const void *sendCountMatrix, HcclDataType sendType,
    HcclDataType recvType)
{
    allMeshAggregationSendRecvInfo_.clear();
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendCounts.resize(userRankSize_);
        sendRecvInfo.sendDispls.resize(userRankSize_);
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        u64 curSendDispls = 0;
        u64 curSendOffset = 0;

        sendRecvInfo.recvCounts.resize(userRankSize_);
        sendRecvInfo.recvDispls.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        u64 curRecvDispls = 0;
        u64 curRecvOffset = 0;
        // sendCountMatrix[i * userRankSize_ + j] 代表rank i发送到rank j的count参数
        for (u32 j = 0; j < userRankSize_; j++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCountMatrix) + i * userRankSize_ + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[sendType];
            sendRecvInfo.sendCounts[j] = curSendCounts;
            sendRecvInfo.sendDispls[j] = curSendDispls;
            sendRecvInfo.sendLength[j] = curSendLength;
            sendRecvInfo.sendOffset[j] = curSendOffset;
            curSendDispls += curSendCounts;
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(sendCountMatrix) + i + userRankSize_ * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[recvType];
            sendRecvInfo.recvCounts[j] = curRecvCounts;
            sendRecvInfo.recvDispls[j] = curRecvDispls;
            sendRecvInfo.recvLength[j] = curRecvLength;
            sendRecvInfo.recvOffset[j] = curRecvOffset;
            curRecvDispls += curRecvCounts;
            curRecvOffset += curRecvLength;

            HCCL_DEBUG("GetAlltoAllvcSendRecvInfo rank[%u], sendCounts[%llu], sendDispls[%llu] "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[j], sendRecvInfo.sendDispls[j],
                sendRecvInfo.recvCounts[j], sendRecvInfo.recvDispls[j]);
        }
        allMeshAggregationSendRecvInfo_.push_back(sendRecvInfo);
    }
    CHK_RET(CheckSendRecvParams(allMeshAggregationSendRecvInfo_));
    return HCCL_SUCCESS;
}

void AlltoAllOperator::UpdateAlltoAllCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    std::string& copyMode)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 maxSendSize = 0;
        u64 maxRecvSize = 0;
        for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
            for (u32 i = 0; i < userRankSize_; i++) {
                u64 curSendSize = sendRecvInfo.sendLength[i] + sendRecvInfo.sendOffset[i];
                maxSendSize = std::max(maxSendSize, curSendSize);
                u64 curRecvSize = sendRecvInfo.recvLength[i] + sendRecvInfo.recvOffset[i];
                maxRecvSize = std::max(maxRecvSize, curRecvSize);
            }
        }
        bool isAlltoAllZCopyMode = (maxSendSize <= GetExternalInputCCLBuffSize()) &&
                                   (maxRecvSize <= GetExternalInputCCLBuffSize());
        if (isAlltoAllZCopyMode) {
           copyMode = "ZCopy";
        }
        HCCL_INFO("[AlltoAllOperator][UpdateAlltoAllCopyMode] maxSendSize[%llu], maxRecvSize[%llu], "\
            "cclBufferSize[%llu], CopyMode[%s]", maxSendSize, maxRecvSize,
            GetExternalInputCCLBuffSize(), copyMode.c_str());
    } else {
        // 图模式走ZCopy实现
        copyMode = "ZCopy";
    }
}

HcclResult AlltoAllOperator::GetAlltoAllvSendRecvInfo(const OpParam& param, const HostMem &alltoallAddrInfoGathered)
{
    allMeshAggregationSendRecvInfo_.clear();
    u64 stepSize = sizeof(u64) * userRankSize_;
    const u32 addrItemNum = 4;
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + 0 * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvLengthStep * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvOffsetStep * stepSize,
            stepSize));
        allMeshAggregationSendRecvInfo_.push_back(std::move(sendRecvInfo));
    }

    for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo_) {
        for (u32 i = 0; i < userRankSize_; i++) {
            sendRecvInfo.sendCounts.push_back(sendRecvInfo.sendLength[i] / SIZE_TABLE[param.All2AllDataDes.sendType]);
            sendRecvInfo.sendDispls.push_back(sendRecvInfo.sendOffset[i] / SIZE_TABLE[param.All2AllDataDes.sendType]);
            sendRecvInfo.recvCounts.push_back(sendRecvInfo.recvLength[i] / SIZE_TABLE[param.All2AllDataDes.recvType]);
            sendRecvInfo.recvDispls.push_back(sendRecvInfo.recvOffset[i] / SIZE_TABLE[param.All2AllDataDes.recvType]);
            HCCL_INFO("[GetAllMeshAggregationSendRecvInfo] rank[%u], sendCounts[%llu], sendDispls[%llu], "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[i], sendRecvInfo.sendDispls[i],
                sendRecvInfo.recvCounts[i], sendRecvInfo.recvDispls[i]);
            HCCL_INFO("[GetAllMeshAggregationSendRecvInfo] rank[%u], sendLength[%llu], sendOffset[%llu], "\
                "recvLength[%llu], recvOffset[%llu]", i, sendRecvInfo.sendLength[i], sendRecvInfo.sendOffset[i],
                sendRecvInfo.recvLength[i], sendRecvInfo.recvOffset[i]);
        }
    }

    CHK_RET(CheckSendRecvParams(allMeshAggregationSendRecvInfo_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::SelectAlgforAlltoAll(const OpParam& param, std::string& algName, std::string& copyMode)
{

    bool useOneLevelAlgorithm =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE ||
        CollAlgOperator::NAFullmeshSatisfyHighPerfAlltoallMeshCondition(deviceType_, userRankSize_)));
        // 用户配置打平 alltoall

    // NA+pairwise算法不支持A+X跨mesh两卡
    bool isSingleDeviceModuleP2p = (userRankSize_ <= HCCL_ALLTOALLV_P2P_SIZE);

    if (IsSatisfyAlltoallPipelineCondition()) {
        algName = "RunAlltoAllVTwoLevelPipeline";
    } else if (useOneLevelAlgorithm || isAllRankSamePlane_ || isSingleDeviceModuleP2p ||
        multiModuleDiffDeviceNumMode_) {
        algName = "RunAlltoAllVFullMesh";
    } else {
        algName = "RunAlltoAllVStaged";
    }

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        // alltoallv
        CHK_RET(GetAlltoAllvSendRecvInfo(param, hostCollectBuffer_));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || param.opType == HcclCMDType::HCCL_CMD_ALLTOALL){
        // alltoallvc&&alltoall
        CHK_RET(GetAlltoAllvcSendRecvInfo(param.All2AllDataDes.sendCountMatrix, param.All2AllDataDes.sendType,
            param.All2AllDataDes.recvType));
    } else {
        HCCL_ERROR("[AlltoAllOperator][SelectAlgforAlltoAll] get wrong opType");
        return HCCL_E_PARA;
    }
    UpdateAlltoAllCopyMode(allMeshAggregationSendRecvInfo_, copyMode);

    HCCL_INFO("[SelectAlgforAlltoAll] all_to_all SelectAlgforAlltoAll is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;
    std::string copyMode = "BCopy";

    ret = SelectAlgforAlltoAll(param, algName, copyMode);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag + algName + copyMode;
    } else {
        newTag = tag;
    }
    HCCL_INFO("[SelectAlg] all_to_all newTag is [%s]", newTag.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[SelectAlgforAlltoAll][SelectAlg]tag[%s], all_reduce failed, return[%d]", tag.c_str(), ret), ret);
    CHK_RET(SetExcutorExtraInfo(algName));
    return ret;
}

HcclResult AlltoAllOperator::GetAlltoAllvAllAddrInfo(u64 *sendLength, u64 *sendOffset,
    u64 *recvLength, u64 *recvOffset, Stream &stream, std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    const u32 addrItemNum = 4;
    u64 stepSize = sizeof(u64) * userRankSize_;

    std::vector<u64> alltoallAddrInfo(userRankSize_ * addrItemNum, 0);
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;

    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[0], stepSize, sendLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[userRankSize_], stepSize, sendOffset, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvLengthStep * userRankSize_], stepSize, recvLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvOffsetStep * userRankSize_], stepSize, recvOffset, stepSize));


    preMetaInfo->inputData = alltoallAddrInfo;
    preMetaInfo->inputSize = stepSize * addrItemNum;
    preMetaInfo->outputSize = userRankSize_ * stepSize * addrItemNum;

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PrepareAlltoAllAddrInfo(const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    Stream &stream, std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    std::vector<u64> vctSendLength(userRankSize_, 0);
    std::vector<u64> vctSendOffset(userRankSize_, 0);
    std::vector<u64> vctRecvLength(userRankSize_, 0);
    std::vector<u64> vctRecvOffset(userRankSize_, 0);

    for (u32 i = 0; i < userRankSize_; i++) {
        vctSendLength[i] = *(static_cast<const u64 *>(sendCounts) + i) * SIZE_TABLE[sendType];
        vctSendOffset[i] = *(static_cast<const u64 *>(sdispls) + i) * SIZE_TABLE[sendType];
        vctRecvLength[i] = *(static_cast<const u64 *>(recvCounts) + i) * SIZE_TABLE[recvType];
        vctRecvOffset[i] = *(static_cast<const u64 *>(rdispls) + i) * SIZE_TABLE[recvType];

        HCCL_DEBUG("[GetAllMeshAggregationSendRecvInfo] rank[%u], SendLength[%llu], SendOffset[%llu], "\
            "RecvLength[%llu], RecvOffset[%llu]", i, vctSendLength[i], vctSendOffset[i], vctRecvLength[i],
            vctRecvOffset[i]);
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(GetAlltoAllvAllAddrInfo(vctSendLength.data(), vctSendOffset.data(),
            vctRecvLength.data(), vctRecvOffset.data(), stream, preMetaInfo));
    } else {
        HCCL_INFO("Run with Graph, alloc new stream");
        Stream graphStream(StreamType::STREAM_TYPE_ONLINE);
        CHK_RET(GetAlltoAllvAllAddrInfo(vctSendLength.data(), vctSendOffset.data(),
            vctRecvLength.data(), vctRecvOffset.data(), graphStream, preMetaInfo));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PreparePreOpParam(OpParam& preProcessOpParam,
    const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream)
{
    u64 stepSize = sizeof(u64) * userRankSize_;
    u32 perDataSize = SIZE_TABLE[HCCL_DATA_TYPE_UINT64];

    preProcessOpParam.tag = HCCL_ALLTOALL_PARA_ALLGATHER;
    preProcessOpParam.inputPtr = cclBufferManager_.GetInAlltoAllvParaBuffer().ptr();
    preProcessOpParam.inputSize = (preMetaInfo->outputSize / stepSize) * perDataSize;
    preProcessOpParam.outputPtr = cclBufferManager_.GetOutAlltoAllvParaBuffer().ptr();
    preProcessOpParam.outputSize = (preMetaInfo->outputSize / stepSize) * perDataSize * userRankSize_;
    preProcessOpParam.DataDes.count = (preMetaInfo->outputSize / stepSize);
    preProcessOpParam.DataDes.dataType = HCCL_DATA_TYPE_UINT64;
    preProcessOpParam.stream = preProcessStream;
    return HCCL_SUCCESS;
}

bool AlltoAllOperator::JudgeIfNeedPreProcessAndGetParam(const OpParam& param,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_RET(PrepareAlltoAllAddrInfo(param.All2AllDataDes.sendCounts, param.All2AllDataDes.sdispls,
            param.All2AllDataDes.sendType, param.All2AllDataDes.recvCounts, param.All2AllDataDes.rdispls,
            param.All2AllDataDes.recvType, const_cast<Stream&>(param.stream), preMetaInfo));
        preMetaInfo->opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        return true;
    }
    return false;
}

void AlltoAllOperator::SetPreProcessResult(HostMem hostCollectBuffer)
{
    hostCollectBuffer_ = std::move(hostCollectBuffer);
}

HcclResult AlltoAllOperator::SetExcutorExtraInfo(const std::string& algName)
{
    HCCL_DEBUG("[AlltoAllOperator][SetExcutorExtraInfo]algName[%s]", algName.c_str());
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        executor_->SetVirtualDispatcher(vDispatcher_);
        ParallelTaskLoader* parallelTaskLoader = nullptr;
        hcclImpl_->GetParallelTaskLoader(parallelTaskLoader);
        executor_->SetParallelTaskLoader(parallelTaskLoader);
        executor_->SetAlgType(algType_);
    }

    return executor_->SetExcutorExtraInfo(allMeshAggregationSendRecvInfo_);
}

bool AlltoAllOperator::HasMassTasks(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    if (isAlltoAllZCopyMode_) {
        return false;
    }

    u64 maxSendTimes = 0;
    u64 maxRecvTimes = 0;
    const u64 cclBufferSize = cclBufferManager_.GetInCCLbufferSize();
    for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
        u64 sendTimes = 0;
        u64 recvTimes = 0;
        for (u32 i = 0; i < userRankSize_; i++) {
            sendTimes += (sendRecvInfo.sendLength[i] + cclBufferSize - 1) / cclBufferSize;
            recvTimes += (sendRecvInfo.recvLength[i] + cclBufferSize - 1) / cclBufferSize;
        }
        maxSendTimes = (maxSendTimes > sendTimes) ? maxSendTimes : sendTimes;
        maxRecvTimes = (maxRecvTimes > recvTimes) ? maxRecvTimes : recvTimes;
    }
    const u64 massThreshold = 65535; //  65535: 单个ffts+任务中，最多承载64K个task
    const u64 maxTasksPerStep = 10;  // BCOPY中每次和远端通信最多消耗task数
    const u64 maxTasksBaseCost = 50; // BCOPY中除每步和远端通信外，最多消耗的task数
    u64 maxTasks = (maxSendTimes + maxRecvTimes) * maxTasksPerStep + maxTasksBaseCost;
    HCCL_DEBUG("[AlltoAllV] bcopy maxSendTimes[%lu], maxRecvTimes[%lu], maxTasks[%lu], hasMassTask[%u]", maxSendTimes,
        maxRecvTimes, maxTasks, (maxTasks > massThreshold));
    return (maxTasks > massThreshold);
}

bool AlltoAllOperator::IsSatisfyAlltoallPipelineCondition()
{
    bool multiRankPerServer = meshAggregationRankSize_ > 1;
    bool isMultiServer = ((userRankSize_ > meshAggregationRankSize_) &&
        (userRankSize_ % meshAggregationRankSize_) == 0);
    const u32 algLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    bool satisfyAlgType = (static_cast<AlgTypeLevel1>(algLevel1) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE);
    HCCL_DEBUG("[AlltoAllOperator][IsSatisfyAlltoallPipelineCondition]multiRankPerServer %u, "
        "isMultiServer %u, satisfyAlgType, %u, multiModuleDiffDeviceNumMode_ %u", multiRankPerServer,
        isMultiServer, satisfyAlgType, multiModuleDiffDeviceNumMode_);
    return (deviceType_ == DevType::DEV_TYPE_910B && satisfyAlgType && multiRankPerServer &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isMultiServer &&
        !multiModuleDiffDeviceNumMode_);
}

std::vector<u64> AlltoAllOperator::GenerateSendCountMatrix(u64 count, u32 rankSize)
{
    std::vector<u64> sendCountMatrix(rankSize * rankSize, count);
    return sendCountMatrix;
}

HcclResult AlltoAllOperator::GetAllMeshAggregationSendRecvInfo(const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, Stream &stream)
{
    std::vector<u64> vctSendLength(userRankSize_, 0);
    std::vector<u64> vctSendOffset(userRankSize_, 0);
    std::vector<u64> vctRecvLength(userRankSize_, 0);
    std::vector<u64> vctRecvOffset(userRankSize_, 0);
    for (u32 i = 0; i < userRankSize_; i++) {
        vctSendLength[i] = *(static_cast<const u64 *>(sendCounts) + i) * SIZE_TABLE[sendType];
        vctSendOffset[i] = *(static_cast<const u64 *>(sdispls) + i) * SIZE_TABLE[sendType];
        vctRecvLength[i] = *(static_cast<const u64 *>(recvCounts) + i) * SIZE_TABLE[recvType];
        vctRecvOffset[i] = *(static_cast<const u64 *>(rdispls) + i) * SIZE_TABLE[recvType];
        HCCL_DEBUG("[GetAllMeshAggregationSendRecvInfo] rank[%u], SendLength[%llu], SendOffset[%llu], "\
            "RecvLength[%llu], RecvOffset[%llu]", i, vctSendLength[i], vctSendOffset[i], vctRecvLength[i],
            vctRecvOffset[i]);
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(GetAlltoAllvAllSendRecvInfo(vctSendLength.data(), vctSendOffset.data(),
            vctRecvLength.data(), vctRecvOffset.data(), allMeshAggregationSendRecvInfo, stream));
    } else {
        CHK_RET(GetAlltoAllvAllSendRecvInfo(vctSendLength.data(), vctSendOffset.data(),
            vctRecvLength.data(), vctRecvOffset.data(), allMeshAggregationSendRecvInfo));
    }
    for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
        for (u32 i = 0; i < userRankSize_; i++) {
            sendRecvInfo.sendCounts.push_back(sendRecvInfo.sendLength[i] / SIZE_TABLE[sendType]);
            sendRecvInfo.sendDispls.push_back(sendRecvInfo.sendOffset[i] / SIZE_TABLE[sendType]);
            sendRecvInfo.recvCounts.push_back(sendRecvInfo.recvLength[i] / SIZE_TABLE[recvType]);
            sendRecvInfo.recvDispls.push_back(sendRecvInfo.recvOffset[i] / SIZE_TABLE[recvType]);
            HCCL_INFO("[GetAllMeshAggregationSendRecvInfo] rank[%u], sendCounts[%llu], sendDispls[%llu], "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[i], sendRecvInfo.sendDispls[i],
                sendRecvInfo.recvCounts[i], sendRecvInfo.recvDispls[i]);
        }
    }

    CHK_RET(AlltoAllVStagedCalculator::CheckSendRecvParams(allMeshAggregationSendRecvInfo));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllvcAllSendRecvInfo(const void *sendCountMatrix, HcclDataType sendType,
    HcclDataType recvType, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo)
{
    allMeshAggregationSendRecvInfo.clear();
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendCounts.resize(userRankSize_);
        sendRecvInfo.sendDispls.resize(userRankSize_);
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        u64 curSendDispls = 0;
        u64 curSendOffset = 0;

        sendRecvInfo.recvCounts.resize(userRankSize_);
        sendRecvInfo.recvDispls.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        u64 curRecvDispls = 0;
        u64 curRecvOffset = 0;
        // sendCountMatrix[i * userRankSize_ + j] 代表rank i发送到rank j的count参数
        for (u32 j = 0; j < userRankSize_; j++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCountMatrix) + i * userRankSize_ + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[sendType];
            sendRecvInfo.sendCounts[j] = curSendCounts;
            sendRecvInfo.sendDispls[j] = curSendDispls;
            sendRecvInfo.sendLength[j] = curSendLength;
            sendRecvInfo.sendOffset[j] = curSendOffset;
            curSendDispls += curSendCounts;
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(sendCountMatrix) + i + userRankSize_ * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[recvType];
            sendRecvInfo.recvCounts[j] = curRecvCounts;
            sendRecvInfo.recvDispls[j] = curRecvDispls;
            sendRecvInfo.recvLength[j] = curRecvLength;
            sendRecvInfo.recvOffset[j] = curRecvOffset;
            curRecvDispls += curRecvCounts;
            curRecvOffset += curRecvLength;

            HCCL_DEBUG("GetAlltoAllvcAllSendRecvInfo rank[%u], sendCounts[%llu], sendDispls[%llu] "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[j], sendRecvInfo.sendDispls[j],
                sendRecvInfo.recvCounts[j], sendRecvInfo.recvDispls[j]);
        }
        allMeshAggregationSendRecvInfo.push_back(sendRecvInfo);
    }
    CHK_RET(AlltoAllVStagedCalculator::CheckSendRecvParams(allMeshAggregationSendRecvInfo));
    return HCCL_SUCCESS;
}

void AlltoAllOperator::UpdateAlltoAllZCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
                                               const std::string &tag)
{
    bool needRecreateAlltoallComm = false;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 maxSendSize = 0;
        u64 maxRecvSize = 0;
        for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
            for (u32 i = 0; i < userRankSize_; i++) {
                u64 curSendSize = sendRecvInfo.sendLength[i] + sendRecvInfo.sendOffset[i];
                maxSendSize = std::max(maxSendSize, curSendSize);
                u64 curRecvSize = sendRecvInfo.recvLength[i] + sendRecvInfo.recvOffset[i];
                maxRecvSize = std::max(maxRecvSize, curRecvSize);
            }
        }

        const u64 cclBufferSize = cclBufferManager_.GetInCCLbufferSize();

        bool isAlltoAllZCopyMode = (maxSendSize <= cclBufferSize) &&
                                   (maxRecvSize <= cclBufferSize);
        HCCL_INFO("[AlltoAllOperator][UpdateAlltoAllZCopyMode] maxSendSize[%llu], maxRecvSize[%llu], "\
            "cclBufferSize[%llu], preZCopyMode[%d], nextZCopyMode[%d]", maxSendSize, maxRecvSize,
            cclBufferSize, isAlltoAllZCopyMode_, isAlltoAllZCopyMode);
        auto iter = isAlltoAllZCopyModeMap_.find(tag);
        if (iter == isAlltoAllZCopyModeMap_.end()) {
            isAlltoAllZCopyModeMap_[tag] = isAlltoAllZCopyMode;
            needRecreateAlltoallComm = false;
        } else {
            needRecreateAlltoallComm = (isAlltoAllZCopyMode != iter->second);
            isAlltoAllZCopyModeMap_[tag] = isAlltoAllZCopyMode;
        }
        isAlltoAllZCopyMode_ = isAlltoAllZCopyMode;
    } else {
        // 图模式走ZCopy实现
        isAlltoAllZCopyMode_ = true;
    }
    hcclImpl_->UpdateAlltoAllStatus(isAlltoAllZCopyMode_, needRecreateAlltoallComm, isAlltoAllZCopyModeMap_);
}

HcclResult AlltoAllOperator::GetAlltoAllvAllSendRecvInfo(u64 *sendLength, u64 *sendOffset, u64 *recvLength,
    u64 *recvOffset, std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    HCCL_INFO("Run with Graph, alloc new stream");
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    CHK_RET(GetAlltoAllvAllSendRecvInfo(sendLength, sendOffset, recvLength, recvOffset, allMeshAggregationSendRecvInfo,
        stream));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllvAllSendRecvInfo(u64 *sendLength, u64 *sendOffset,
    u64 *recvLength, u64 *recvOffset, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, Stream &stream)
{
    allMeshAggregationSendRecvInfo.clear();
    // 对数据做allgather
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED,
        HCCL_ERROR("[GetAlltoAllvAllSendRecvInfo]Invalid Workflow Mode[%d]", mode),
        HCCL_E_INTERNAL);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    const u32 addrItemNum = 4;
    u64 stepSize = sizeof(u64) * userRankSize_;
    auto inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
    auto outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    if ((inAlltoAllvParaBuffer.ptr() == nullptr) || (outAlltoAllvParaBuffer.ptr() == nullptr)) {
        CHK_RET(
            cclBufferManager_.InitAlltoAllvParaBuffer(stepSize * addrItemNum, stepSize * userRankSize_ * addrItemNum));
        inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
        outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    }
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    if ((inCCLbuffer.ptr() == nullptr) || (outCCLbuffer.ptr() == nullptr)) {
        CHK_RET(cclBufferManager_.CreateCommCCLbuffer());
        inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
        outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    }
    std::vector<u64> alltoallAddrInfo(userRankSize_ * addrItemNum, 0);

    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[0], stepSize, sendLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[userRankSize_], stepSize, sendOffset, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvLengthStep * userRankSize_], stepSize, recvLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvOffsetStep * userRankSize_], stepSize, recvOffset, stepSize));

    CHK_RET(hcclStreamSynchronize(stream.ptr()));
    CHK_RET(hrtMemSyncCopy(inAlltoAllvParaBuffer.ptr(), stepSize * addrItemNum, alltoallAddrInfo.data(),
        stepSize * addrItemNum, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    HCCL_PROFILER_DEL_STREAM(stream.ptr());
    HCCL_PROFILER_ADD_STREAM(stream.ptr(), HCCL_ALLTOALL_PARA_ALLGATHER, 0, algType_);
    CHK_RET(ExchangeSendRecvInfoFromAllGather(HCCL_ALLTOALL_PARA_ALLGATHER, inAlltoAllvParaBuffer.ptr(),
        outAlltoAllvParaBuffer.ptr(), userRankSize_ * addrItemNum, HCCL_DATA_TYPE_UINT64, stream));
    HCCL_PROFILER_DEL_STREAM(stream.ptr());

    SetWorkflowMode(mode);

    HostMem alltoallAddrInfoGathered = HostMem::alloc(userRankSize_ * stepSize * addrItemNum);
    CHK_PTR_NULL(alltoallAddrInfoGathered.ptr());
    CHK_RET(hrtMemSyncCopy(alltoallAddrInfoGathered.ptr(), userRankSize_ * stepSize * addrItemNum,
        outAlltoAllvParaBuffer.ptr(), userRankSize_ * stepSize * addrItemNum,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
    // 非单算子场景，中转内存使用完之后直接释放
    if (mode != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        cclBufferManager_.ReleaseAlltoAllvParaBuffer();
    }
    CHK_RET(FormatAllMeshAggregationSendRecvInfo(alltoallAddrInfoGathered, allMeshAggregationSendRecvInfo));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::ExchangeSendRecvInfoFromAllGather(const std::string &tag, void *inputPtr, void *outputPtr,
    u64 inputCount, HcclDataType dataType, Stream stream)
{
    AllGatherOperator operation(hcclImpl_, topoMatcher_);
    CHK_RET(operation.AllGatherOutPlace(tag, inputPtr, outputPtr, inputCount, dataType, stream));
    CHK_RET(hcclStreamSynchronize(stream.ptr()));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::FormatAllMeshAggregationSendRecvInfo(HostMem &alltoallAddrInfoGathered,
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 stepSize = sizeof(u64) * userRankSize_;
    const u32 addrItemNum = 4;
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + 0 * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvLengthStep * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvOffsetStep * stepSize,
            stepSize));

        allMeshAggregationSendRecvInfo.push_back(std::move(sendRecvInfo));
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
    u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize)
{
    std::vector<u64> sendLength(userRankSize_, 0);
    std::vector<u64> sendOffset(userRankSize_, 0);
    std::vector<u64> recvLength(userRankSize_, 0);
    std::vector<u64> recvOffset(userRankSize_, 0);
    for (u32 i = 0; i < sendLength.size(); i++) {
        sendLength[i] = *(sendCounts + i) * SIZE_TABLE[sendType];
        sendOffset[i] = *(sdispls + i) * SIZE_TABLE[sendType];
        recvLength[i] = *(recvCounts + i) * SIZE_TABLE[recvType];
        recvOffset[i] = *(rdispls + i) * SIZE_TABLE[recvType];
    }

    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    CHK_RET(GetAlltoAllvAllSendRecvInfo(
        sendLength.data(), sendOffset.data(), recvLength.data(), recvOffset.data(), allMeshAggregationSendRecvInfo));
    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo,
        memSize, meshAggregationRankSize_);

    HCCL_INFO("Calculate workSpace MemSize done, memSize[%llu]", memSize);

    // 计算结果
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
{
    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo,
        memSize, meshAggregationRankSize_);

    HCCL_INFO("Calculate workSpace MemSize done, memSize[%llu]", memSize);

    // 计算结果
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PrepareAlltoAllVStaged1(DeviceMem &sendBuf, DeviceMem &recvBuf, DeviceMem &scratchMem,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosIntra,
    Stream &stream, const std::string &tag, std::unique_ptr<AlltoAllVStagedBase> &alltoallOuter)
{
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    // opbase BCopy 不支持fullmesh算法，因此不必做算法选择
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && Buffer拷贝模式
        HCCL_INFO("Running alltoallv Staged Pairwise intra Server");
        alltoallOuter.reset(new (std::nothrow)AlltoAllVStagedPairwise(dispatcher_, stream));
        CHK_SMART_PTR_NULL(alltoallOuter);
        CHK_RET(alltoallOuter->Prepare(sendBuf, scratchMem, inCCLbuffer, outCCLbuffer, sendAddrInfosIntra,
            recvAddrInfosIntra, isAlltoAllZCopyMode_));
    } else {
        bool isOpBaseZCopy = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
                             isAlltoAllZCopyMode_;
        DeviceMem inBuf = isOpBaseZCopy ? inCCLbuffer : sendBuf;
        // 单MeshAggregation下, 分级算法不做第二级, 结果输出到outCCLbuffer_
        DeviceMem outBuf = (isOpBaseZCopy && isSingleMeshAggregation_) ? recvBuf : scratchMem;
        // opbase ZCopy 与 graph，除input buffer差异外，其余行为应保持一致
        if (isOpBaseZCopy) { // 单算子 && ZCopy模式
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCCLbuffer, sendBuf, stream));
        }
        // 互联场景, alltoall暂不支持走fullmesh+pairwise
        if ((GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] ==
            HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE &&
            GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] ==
            HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE) ||
            pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] != 0 ||
            meshAggregationRankSize_ == 1) {
            HCCL_INFO("Running alltoallv Staged Pairwise intra Server");
            alltoallOuter.reset(new (std::nothrow)AlltoAllVStagedPairwise(dispatcher_, stream));
            CHK_SMART_PTR_NULL(alltoallOuter);
            CHK_RET(alltoallOuter->Prepare(inBuf, outBuf, sendAddrInfosIntra,
                recvAddrInfosIntra, isAlltoAllZCopyMode_));
        } else {
            HCCL_INFO("Running alltoallv Staged Mesh intra Server");
            HcclResult ret = hcclImpl_->CreateMutiStreamRes(tag, stream, algType_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AlltoAllOperator][AlltoAllv]errNo[0x%016llx] tag[%s],\
                alltoallv create stream resource", HCCL_ERROR_CODE(ret), tag.c_str()), ret);

            u32 rankSize = meshAggregationRankSize_;
            innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
            CHK_PRT_RET(streamInfo == nullptr,
                HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
                    HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // 从stream 个数 = ranksize -2
                    ret = StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                        streamInfo->ringStreams[streamIndex].ptr(), stream.ptr());
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[AlltoAllOperator][ActiveRingStreams]stream[%u] active failed,return[%d]",
                            streamIndex, ret), ret);
                }
            }

            // 添加从流profiling, 用于维护planID
            CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_ALLTOALL));

            if (GetExternalInputHcclEnableFfts() ||
                streamInfo->ringStreams.size() == 0) {
                alltoallOuter.reset(new (std::nothrow) AlltoAllVStagedMesh(dispatcher_, stream,
                    streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_, streamInfo->ringStreams));
            } else {
                alltoallOuter.reset(new (std::nothrow) AlltoAllVStagedMesh(vDispatcher_, stream,
                    streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_, streamInfo->ringStreams));
            }
            CHK_SMART_PTR_NULL(alltoallOuter);
            CHK_RET(alltoallOuter->Prepare(inBuf, outBuf, sendAddrInfosIntra,
                recvAddrInfosIntra, isAlltoAllZCopyMode_, streamInfo->ringStreams));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PrepareAlltoAllVStaged2(DeviceMem &recvBuf, DeviceMem &scratchMem,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosInter,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosInter,
    Stream &stream, const std::string &tag, std::unique_ptr<AlltoAllVStagedBase> &alltoallInner)
{
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    alltoallInner.reset(new (std::nothrow)AlltoAllVStagedPairwise(dispatcher_, stream));
    CHK_SMART_PTR_NULL(alltoallInner);
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
        CHK_RET(alltoallInner->Prepare(scratchMem, recvBuf, inCCLbuffer, outCCLbuffer, sendAddrInfosInter,
            recvAddrInfosInter, isAlltoAllZCopyMode_));
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_) { // 单算子 && ZCopy模式
        CHK_RET(alltoallInner->Prepare(scratchMem, outCCLbuffer, inCCLbuffer, outCCLbuffer,
            sendAddrInfosInter, recvAddrInfosInter, isAlltoAllZCopyMode_));
    } else {
        CHK_RET(alltoallInner->Prepare(scratchMem, recvBuf, sendAddrInfosInter, recvAddrInfosInter,
            isAlltoAllZCopyMode_));
    }
    return HCCL_SUCCESS;
}

// 计算 alltoall pipeline 910B 的两级流水算法本卡需要的 scratch 大小(图模式需要)
u64 AlltoAllOperator::GetAlltoall2LevelPipelineScratchSize910B(
    u32 rank,
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 scratchSize = 0;
    u32 meshRankStart = (rank / meshAggregationRankSize_) * meshAggregationRankSize_;
    u32 meshRankEnd = meshRankStart + meshAggregationRankSize_ - 1;
    u32 rankIntraMesh = rank - meshRankStart;
    for (u32 sendRank = rankIntraMesh, userRankSize = allMeshAggregationSendRecvInfo.size();
        sendRank < userRankSize; sendRank += meshAggregationRankSize_) {
        const std::vector<u64>& remoteSendLength = allMeshAggregationSendRecvInfo[sendRank].sendLength;
        const std::vector<u64>& remoteSendOffset = allMeshAggregationSendRecvInfo[sendRank].sendOffset;
        scratchSize += (remoteSendOffset[meshRankEnd] + remoteSendLength[meshRankEnd] -
            remoteSendOffset[meshRankStart]);
    }
    return scratchSize;
}

// 计算 alltoall pipeline 910B 的两级流水算法所有卡需要的 scratch 大小的最大值(单算子模式需要)
u64 AlltoAllOperator::GetAlltoall2LevelPipelineMaxScratchSize910B(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 maxScratchSize = 0;
    for (u32 rank = 0, userRankSize = allMeshAggregationSendRecvInfo.size(); rank < userRankSize; rank++) {
        u64 currRankScratchSize = GetAlltoall2LevelPipelineScratchSize910B(rank, allMeshAggregationSendRecvInfo);
        maxScratchSize = (currRankScratchSize > maxScratchSize ? currRankScratchSize : maxScratchSize);
    }
    return maxScratchSize;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALLV, AlltoAllV, AlltoAllOperator);
REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALL, AlltoAll, AlltoAllOperator);
REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALLVC, AlltoAllVC, AlltoAllOperator);

}