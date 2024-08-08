/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <algorithm>

#include "coll_alg_operator.h"
#include "device_capacity.h"
#include "nonuniform_hierarchical_ring_base_pub.h"
#include "coll_executor_base.h"
#include "coll_alg_exec_registry.h"


namespace hccl {
using namespace std;
constexpr float GB2B = 1024 * 1024 * 1024;
constexpr float SECOND2MICROSECOND = 1000000;
constexpr float RHD_FACTOR_TWO = 2.0;
constexpr float RHD_FACTOR_ONE = 1.0;
constexpr float DOUBLE_SUB_HCCLCMD = 2.0; // The hcclCMD can be considered as combination of two hcclCMDs.
constexpr float COPY_TIME_IN_RHD = 1.0;
constexpr double NHR_FACTOR_TWO = 2.0;
constexpr double NHR_FACTOR_THREE = 3.0;
constexpr double NHR_FACTOR_FOUR = 4.0;
constexpr double NHR_SUB_TWO = 2.0;
constexpr float LATENCY = 60; // 静态时延 60 us;
constexpr u64 PIPELINE_MIN_SIZE = 32 * 1024; // 当数据量大于等于32KB时，reduce_scatter和all_gather使能pipeline模式
constexpr u64 PIPELINE_ALLREDUCE_MIN_SIZE = 1024 * 1024; // 当数据量大于等于1MB时，allreduce使能pipeline模式
constexpr u64 PIPELINE_MIN_SIZE_NO_LITE = 2 * 1024 * 1024; // 如不支持RDMALite，当数据量大于等于2MB时，使能pipeline模式

CollAlgOperator::CollAlgOperator(std::unique_ptr<hcclImpl> &pImpl,
                                 std::unique_ptr<TopoMatcher> &topoMatcher, HcclCMDType opType)
    : cclBufferManager_(pImpl->cclBufferManager_), notifyPool_(pImpl->notifyPool_),
      rankInfoList_(pImpl->rankInfoList_), hcclImpl_(pImpl), topoMatcher_(topoMatcher)
{
    hcclImpl_->GetDispatcher(dispatcher_);
    hcclImpl_->GetVirtualDispatcher(vDispatcher_);
    SetTopoAttr();
    SetAlgoAttr();
    hcclImpl_->GetAlgTypeDirect(algType_, opType);
    hcclImpl_->GetAlgoLevel1DefaultSwitch(isAlgoLevel1Default_, opType);
    hcclImpl_->GetTopoType(topoType_);
}

HcclResult CollAlgOperator::SelectAlg(const std::string& tag,
    const OpParam& param, std::string& algName, std::string& newTag)
{
    return HCCL_SUCCESS;
}

bool CollAlgOperator::CheckNeedRecreateComm(const std::string& algName, u64 lastScratchMemSize)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        SetExecutorAttr();
    }
    return executor_->CheckNeedRecreateComm(lastScratchMemSize);
}

HcclResult CollAlgOperator::CalcResRequest(const std::string& algName, const OpParam& param,
    AlgResourceRequest& resourceRequest)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        SetExecutorAttr();
    }
    return executor_->CalcResRequest(param, resourceRequest);
}

HcclResult CollAlgOperator::Orchestrate(const std::string& algName, const OpParam& param,
    const AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAlgOperator][Orchestrate]algName[%s]", algName.c_str());
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][Orchestrate]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        SetExecutorAttr();
    }

    return executor_->Orchestrate(param, algResource);
}

HcclResult CollAlgOperator::CalcIncreLinkRequest(const std::string& algName, const OpParam& param,
    AlgResourceRequest& resourceRequest)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance()->GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[BatchSendRecvOperator][CalcIncreLinkRequest]Fail to find executor for algName[%s]",
            algName.c_str()), HCCL_E_PARA);
    }
    return executor_->CalcIncreLinkRequest(param, resourceRequest);
}

bool CollAlgOperator::JudgeIfNeedPreProcessAndGetParam(const OpParam& param,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    return false;
}

HcclResult CollAlgOperator::PreparePreOpParam(OpParam& preProcessOpParam,
    const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream)
{
    return HCCL_SUCCESS;
}

void CollAlgOperator::SetPreProcessResult(HostMem hostCollectBuffer)
{
    return;
}

void CollAlgOperator::SetTopoAttr()
{
    serverNum_= hcclImpl_->serverNum_;
    moduleNum_ = hcclImpl_->moduleNum_;
    devNumInLevel2_ = hcclImpl_->devNumInLevel2_;
    deviceNumPerServer_ = hcclImpl_->deviceNumPerServer_;
    deviceNumPerAggregation_ = hcclImpl_->deviceNumPerAggregation_;
    multiModuleDiffDeviceNumMode_ = hcclImpl_->multiModuleDiffDeviceNumMode_;

    meshAggregationRankSize_ = hcclImpl_->meshAggregationRankSize_;
    isDiffDeviceModule_ = hcclImpl_->isDiffDeviceModule_;
    isSingleMeshAggregation_= hcclImpl_->isSingleMeshAggregation_;
    meshSinglePlane_= hcclImpl_->meshSinglePlane_;
    isAllRankSamePlane_ = hcclImpl_->isAllRankSamePlane_;
    is310PDuoCard_ = hcclImpl_->is310PDuoCard_;

    userRank_ = hcclImpl_->userRank_;
    realUserRank_ = hcclImpl_->realUserRank_;
    userRankSize_ = hcclImpl_->userRankSize_;

    devicePhyId_ = hcclImpl_->devicePhyId_;
    deviceLogicId_ = hcclImpl_->deviceLogicId_;
    deviceType_ = hcclImpl_->deviceType_;

    nicList_ = hcclImpl_->nicList_;
    pairLinkCounter_ = hcclImpl_->pairLinkCounter_;
    isSupportRdmaLite_ = hcclImpl_->isSupportRdmaLite_;
    return;
}

void CollAlgOperator::SetAlgoAttr()
{
    isHaveCpuRank_ = hcclImpl_->isHaveCpuRank_;
    inlineReduceSwitchOn_ = hcclImpl_->inlineReduceSwitchOn_;
    identifier_ = hcclImpl_->identifier_;
    return;
}

void CollAlgOperator::SetExecutorAttr()
{
    executor_->SetAlgType(algType_);
    executor_->SetVirtualDispatcher(vDispatcher_);
    executor_->SetCCLInBuffer(hcclImpl_->GetInCCLbufferSize());
    ParallelTaskLoader* parallelTaskLoader = nullptr;
    hcclImpl_->GetParallelTaskLoader(parallelTaskLoader);
    executor_->SetParallelTaskLoader(parallelTaskLoader);
    return;
}

std::string CollAlgOperator::GenerateNewTagByAlgTypeLevel1(std::string tag, std::string algTypeLevel1Tag) const
{
    if (algTypeLevel1Tag == "") {
        return tag;
    } else {
        return tag + "_" + algTypeLevel1Tag;
    }
}

HcclResult CollAlgOperator::AppendTag(const AlgTypeLevel1 &algTypeLevel1, std::string &tag)
{
    switch (algTypeLevel1) {
        case AlgTypeLevel1::ALG_LEVEL1_RING:
            tag = "ALG_LEVEL1_RING";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_HD:
            tag = "ALG_LEVEL1_HD";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_NHR:
            tag = "ALG_LEVEL1_NHR";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_PIPELINE:
            tag = "ALG_LEVEL1_PIPELINE";
            break;
        default:
            HCCL_WARNING("[CollAlgOperator][AppendTag] The algTypeLevel1 %d is not supported.", algTypeLevel1);
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::AutoSelectAlgTypeLevel1(HcclCMDType hcclCMDType, u64 countSize, u64 cclBufferSize,
                                                    std::string &algTypeLevel1Tag, bool isInlineReduce,
                                                    bool isRdmaReduce, bool isAivMode)
{
    if (isSingleMeshAggregation_) {
        HCCL_INFO("hccl algorithm: there are %u server(%u module) in level1, no need to choose algo.",
                  serverNum_, moduleNum_);
        return HCCL_SUCCESS;
    }
    // parse algType_ and get algTypeLevel1 and algTypeLevel0
    auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
    // auto algo selection process
    if (isAlgoLevel1Default_) {
        // set algTypeLevel1
        AlgTypeLevel1 algTypeLevel1;
        CHK_RET(
            GetDefaultAlgoLevel1V2(
                hcclCMDType, countSize, cclBufferSize, algTypeLevel1, isInlineReduce, isRdmaReduce, isAivMode));
        auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
        CHK_PRT_RET(iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
            HCCL_ERROR("level1: algType[%u] is invalid.", algTypeLevel1),
            HCCL_E_INTERNAL);
        HCCL_INFO("hccl algorithm: there are %u server(%u module) in level1, using %s algo",
                  serverNum_, moduleNum_, iter->second.c_str());
        algType_ = AlgType(
            (static_cast<u32>(algTypeLevel1) << HCCL_LEVEL_ALGO_WIDTH) + static_cast<u32>(originalAlgTypeLevel0));
        // tag 增加所选的算法
        AppendTag(algTypeLevel1, algTypeLevel1Tag);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoForComm(HcclCMDType hcclCMDType, float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    // 从map中查找对应的计算函数
    auto it = selectFuncMap_.find(hcclCMDType);
    if (it == selectFuncMap_.end()) {
        HCCL_ERROR("[Get][AlgTypeLevel1] The hcclCMDType %d is not supported.", hcclCMDType);
        return HCCL_E_NOT_SUPPORT;
    }
    return (it->second)(delay, curSize, bandWidth, algType);
}

bool CollAlgOperator::IsAlgTypeLevel0Mesh(AlgTypeLevel0 &originalAlgTypeLevel0) const
{
    return originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_2P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_1P_MESH;
}

HcclResult CollAlgOperator::GetDefaultAlgoLevel1V2(HcclCMDType hcclCMDType, u64 curSize, u64 cclBufferSize,
    AlgTypeLevel1 &algType, bool isInlineReduce, bool isRdmaReduce, bool isAivMode)
{
    // pipeline mode is deployed,where there is multi-sever multi-device(insever) now,
    // since RDMA is not reduced by normal serial orchestration of tasks.
    // So pipeline mode is more dominant than normal serial orchestration now.
    auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
    // 对于不支持Rdma Lite的场景，下发性能较差，RS和AG需要一个很大的数据量（AR的一半）才能掩盖下发时间
    u64 pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (((hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && isInlineReduce && isRdmaReduce &&
        topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE) ||
        hcclCMDType == HcclCMDType::HCCL_CMD_ALLGATHER) &&
        deviceNumPerAggregation_ != 1 && curSize >= pipelineMinSize && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0)) {
        algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
        return HCCL_SUCCESS;
    }

    // 对于不支持Rdma Lite的场景，下发性能较差，AllReduce需要一个较大的数据量才能掩盖下发时间
    pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_ALLREDUCE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        // 计算每个slice的大小
        u64 allreduceCurSize = 0;
        allreduceCurSize = curSize / (moduleNum_ * deviceNumPerAggregation_);
        if ((isInlineReduce && isRdmaReduce) && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE &&
            deviceNumPerAggregation_ != 1 && allreduceCurSize >= pipelineMinSize && !isAivMode &&
            IsAlgTypeLevel0Mesh(originalAlgTypeLevel0)) {
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            return HCCL_SUCCESS;
        }
    }
    u64 dataSizePerLoop = curSize > cclBufferSize ? cclBufferSize : curSize;
    float delay = LATENCY; // 静态时延 60 us;
    float bandWidth;
    CHK_RET(GetBandWidthPerNPU(1, userRankSize_, deviceNumPerAggregation_, bandWidth)); // 单位：GB/s
    bandWidth = bandWidth * GB2B; // 单位：B/s
    CHK_RET(SelectAlgoForComm(hcclCMDType, delay, dataSizePerLoop, bandWidth, algType));
    return HCCL_SUCCESS;
}


AlgTypeLevel0 CollAlgOperator::GetLevel0AlgType(const AlgType algType) const
{
    const u32 algLevel0 = static_cast<u32>(algType) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1);
    return static_cast<AlgTypeLevel0>(algLevel0);
}

AlgTypeLevel1 CollAlgOperator::GetLevel1AlgType(const AlgType algType) const
{
    const u32 algLevel1 = static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH;
    return static_cast<AlgTypeLevel1>(algLevel1);
}

AlgTypeLevel2 CollAlgOperator::GetLevel2AlgType(const AlgType algType) const
{
    const u32 algLevel2 = static_cast<u32>(algType) >> (HCCL_LEVEL_ALGO_WIDTH * 2);
    return static_cast<AlgTypeLevel2>(algLevel2);
}

bool CollAlgOperator::UseInterServerRingAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_RING;
}

bool CollAlgOperator::UseInterServerNHRAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NHR;
}

bool CollAlgOperator::UseInterServerHDAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_HD;
}

bool CollAlgOperator::UseInterServerNHRV1Algo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
}

bool CollAlgOperator::UseInterServerNBAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NB;
}

bool CollAlgOperator::UseInterServerPipelineAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
}

bool CollAlgOperator::UseLevel2RingAlgo(AlgType algType)
{
    return GetLevel2AlgType(algType) == AlgTypeLevel2::ALG_LEVEL2_RING;
}

HcclResult CollAlgOperator::SetInterServerHDAlgo(AlgType &algType) const
{
    switch (algType) {
        case AlgType::ALG_8P_RING_PLUS_PIPELINE:
        case AlgType::ALG_8P_RING_PLUS_RING:
        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_8P_RING_PLUS_NB:
            algType = AlgType::ALG_8P_RING_PLUS_HD;
            break;

        case AlgType::ALG_4P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_4P_MESH_PLUS_RING:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NB:
            algType = AlgType::ALG_4P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_2P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_2P_MESH_PLUS_RING:
        case AlgType::ALG_2P_MESH_PLUS_NHR:
        case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_2P_MESH_PLUS_NB:
            algType = AlgType::ALG_2P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_1P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_1P_MESH_PLUS_RING:
        case AlgType::ALG_1P_MESH_PLUS_NHR:
        case AlgType::ALG_1P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_1P_MESH_PLUS_NB:
            algType = AlgType::ALG_1P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_4P_RING_PLUS_PIPELINE:
        case AlgType::ALG_4P_RING_PLUS_RING:
        case AlgType::ALG_4P_RING_PLUS_NHR:
        case AlgType::ALG_4P_RING_PLUS_NHR_V1:
        case AlgType::ALG_4P_RING_PLUS_NB:
            algType = AlgType::ALG_4P_RING_PLUS_HD;
            break;

        case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_HD;
            break;

        case AlgType::ALG_NP_MESH_PLUS_PIPELINE:
        case AlgType::ALG_NP_MESH_PLUS_RING:
        case AlgType::ALG_NP_MESH_PLUS_NHR:
        case AlgType::ALG_NP_MESH_PLUS_NHR_V1:
        case AlgType::ALG_NP_MESH_PLUS_NB:
            algType = AlgType::ALG_NP_MESH_PLUS_HD;
            break;

        case AlgType::ALG_NP_DOUBLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
        case AlgType::ALG_NP_DOUBLE_RING_PLUS_NB:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_HD;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SetInterServerRingAlgo(AlgType &algType) const
{
    switch (algType) {
        case AlgType::ALG_8P_RING_PLUS_HD:
        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_8P_RING_PLUS_NB:
            algType = AlgType::ALG_8P_RING_PLUS_RING;
            break;
        case AlgType::ALG_4P_MESH_PLUS_HD:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NB:
            algType = AlgType::ALG_4P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_2P_MESH_PLUS_HD:
        case AlgType::ALG_2P_MESH_PLUS_NHR:
        case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_2P_MESH_PLUS_NB:
            algType = AlgType::ALG_2P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_1P_MESH_PLUS_HD:
        case AlgType::ALG_1P_MESH_PLUS_NHR:
        case AlgType::ALG_1P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_1P_MESH_PLUS_NB:
            algType = AlgType::ALG_1P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_4P_RING_PLUS_HD:
        case AlgType::ALG_4P_RING_PLUS_NHR:
        case AlgType::ALG_4P_RING_PLUS_NHR_V1:
        case AlgType::ALG_4P_RING_PLUS_NB:
            algType = AlgType::ALG_4P_RING_PLUS_RING;
            break;
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_RING;
            break;
        case AlgType::ALG_NP_MESH_PLUS_HD:
        case AlgType::ALG_NP_MESH_PLUS_NHR:
        case AlgType::ALG_NP_MESH_PLUS_NHR_V1:
        case AlgType::ALG_NP_MESH_PLUS_NB:
            algType = AlgType::ALG_NP_MESH_PLUS_RING;
            break;
        case AlgType::ALG_DOUBLE_RING_PLUS_HD:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_RING;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SetInterServerNHRAlgo(AlgType &algType) const
{
    switch (algType) {
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_NHR;
            break;
        case AlgType::ALG_DOUBLE_RING_PLUS_HD:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_NHR;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForReduceScatter(float delay, u64 recvCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                recvCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD,
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 recvCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForAllGather(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForGather(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForAllReduce(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = NHR_FACTOR_TWO * ceil(log2(moduleNum_)) * delay +
                NHR_FACTOR_TWO * static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                curSize / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = DOUBLE_SUB_HCCLCMD * ceil(log2(moduleNum_)) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForBroadcast(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth
                 * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = scatter + allgather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay +
                 (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForReduce(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = reducescatter + gather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay +
                 (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

bool CollAlgOperator::NAFullmeshSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91073 = (deviceType == DevType::DEV_TYPE_910_73);
    bool oneLevelUseMesh =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH);
    bool isHCCS = !GetExternalInputInterHccsDisable();
    HCCL_DEBUG("[CollAlgOperator][AlltoAllVCOutPlace]isDevice91073 %u oneLevelUseMesh %u isHCCS %u",
        isDevice91073, oneLevelUseMesh, isHCCS);
    CHK_PRT_CONT(!(oneLevelUseMesh && !isDevice91073),
        HCCL_WARNING("[CollAlgOperator][NAFullmeshSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm only "
            "support 91073 device type, use default algorithm type"));
    CHK_PRT_CONT(!(oneLevelUseMesh && !isHCCS),
        HCCL_WARNING("[CollAlgOperator][NAFullmeshSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm depends "
            "on HCCS, use default algorithm type"));
    return (isDevice91073 && oneLevelUseMesh && rankSizeSupport && isHCCS);
}

bool CollAlgOperator::FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91073 = (deviceType == DevType::DEV_TYPE_910_73);
    bool twoLevelIntraUseMesh =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE);
    bool isHCCS = !GetExternalInputInterHccsDisable();
    HCCL_DEBUG("[CollAlgOperator][AlltoAllVCOutPlace]isDevice91073 %u twoLevelIntraUseMesh %u isHCCS %u",
        isDevice91073, twoLevelIntraUseMesh, isHCCS);
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isDevice91073),
        HCCL_WARNING("[CollAlgOperator][FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm only "
            "support 91073 device type, use default algorithm type"));
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isHCCS),
        HCCL_WARNING("[CollAlgOperator][FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm depends "
            "on HCCS, use default algorithm type"));
    return (isDevice91073 && twoLevelIntraUseMesh && rankSizeSupport && isHCCS);
}

}   // namesapce hccl