/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 内部依赖头文件
#include <cmath>
#include <algorithm>
#include "externalinput_pub.h"
#include "device_capacity.h"
#include "stream_active_manager.h"
#include "profiling_manager_pub.h"
#include "heartbeat_pub.h"
#include "hccl_alg.h"
#include "hccl_impl.h"

using namespace std;

namespace hccl {

std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> hcclImpl::inOutPutTempMem_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> hcclImpl::inOutPutTempMemMutex_;
std::array<Referenced, MAX_MODULE_DEVICE_NUM> hcclImpl::instanceRef_;

const std::map<AlgTypeLevel2, std::string> HCCL_ALGO_LEVEL2_NAME_MAP = {
    {AlgTypeLevel2::ALG_LEVEL2_HD, "H-D"},
    {AlgTypeLevel2::ALG_LEVEL2_RING, "ring"},
    {AlgTypeLevel2::ALG_LEVEL2_RESERVED, "null"},
};

constexpr u32 TINY_MEMORY_SIZE = 32; // sendBuff或recvBuff为空时, 使用的DeviceMem大小

const std::set<AlgType> HCCL_ALGO_TYPE_MAP = {
    AlgType::ALG_DEFAULT,
    AlgType::ALG_8P_RING_PLUS_HD,
    AlgType::ALG_4P_MESH_PLUS_HD,
    AlgType::ALG_2P_MESH_PLUS_HD,
    AlgType::ALG_1P_MESH_PLUS_HD,
    AlgType::ALG_4P_RING_PLUS_HD,
    AlgType::ALG_NP_SINGLE_RING_PLUS_HD,
    AlgType::ALG_8P_RING_PLUS_RING,
    AlgType::ALG_4P_MESH_PLUS_RING,
    AlgType::ALG_2P_MESH_PLUS_RING,
    AlgType::ALG_1P_MESH_PLUS_RING,
    AlgType::ALG_4P_RING_PLUS_RING,
    AlgType::ALG_NP_SINGLE_RING_PLUS_RING,
    AlgType::ALG_NP_MESH_PLUS_RING,
    AlgType::ALG_NP_MESH_PLUS_HD,
    AlgType::ALG_8P_RING_PLUS_NHR,
    AlgType::ALG_4P_MESH_PLUS_NHR,
    AlgType::ALG_2P_MESH_PLUS_NHR,
    AlgType::ALG_1P_MESH_PLUS_NHR,
    AlgType::ALG_4P_RING_PLUS_NHR,
    AlgType::ALG_NP_SINGLE_RING_PLUS_NHR,
    AlgType::ALG_NP_MESH_PLUS_NHR,
    AlgType::ALG_WHOLE_NHR,
    AlgType::ALG_8P_RING_PLUS_NHR_V1,
    AlgType::ALG_4P_MESH_PLUS_NHR_V1,
    AlgType::ALG_2P_MESH_PLUS_NHR_V1,
    AlgType::ALG_1P_MESH_PLUS_NHR_V1,
    AlgType::ALG_4P_RING_PLUS_NHR_V1,
    AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1,
    AlgType::ALG_NP_MESH_PLUS_NHR_V1,
    AlgType::ALG_WHOLE_NHR_V1,
    AlgType::ALG_8P_RING_PLUS_NB,
    AlgType::ALG_4P_MESH_PLUS_NB,
    AlgType::ALG_2P_MESH_PLUS_NB,
    AlgType::ALG_1P_MESH_PLUS_NB,
    AlgType::ALG_4P_RING_PLUS_NB,
    AlgType::ALG_NP_SINGLE_RING_PLUS_NB,
    AlgType::ALG_NP_DOUBLE_RING_PLUS_NB,
    AlgType::ALG_NP_MESH_PLUS_NB,
    AlgType::ALG_WHOLE_NB,
    AlgType::ALG_NP_STAR,
    AlgType::ALG_NP_HD,
    AlgType::ALG_DOUBLE_RING_PLUS_RING,
    AlgType::ALG_DOUBLE_RING_PLUS_HD,
    AlgType::ALG_DOUBLE_RING_PLUS_NHR,
    AlgType::ALG_WHOLE_RING_PLUS_PIPELINE,
    AlgType::ALG_8P_RING_PLUS_PIPELINE,
    AlgType::ALG_4P_MESH_PLUS_PIPELINE,
    AlgType::ALG_2P_MESH_PLUS_PIPELINE,
    AlgType::ALG_1P_MESH_PLUS_PIPELINE,
    AlgType::ALG_4P_RING_PLUS_PIPELINE,
    AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE,
    AlgType::ALG_NP_DOUBLE_RING_PLUS_PIPELINE,
    AlgType::ALG_NP_MESH_PLUS_PIPELINE,
    AlgType::ALG_NP_STAR_PLUS_PIPELINE,
};

hcclImpl::hcclImpl(const HcclDispatcher dispatcher, const HcclDispatcher vDispatcher,
    const std::unique_ptr<NotifyPool> &notifyPool, std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
    const std::unique_ptr<QueueNotifyManager> &queueNotifyManager,
    std::unique_ptr<WorkspaceResource> &workSpaceRes, CCLBufferManager &cclBufferManager,
    const void *transportResourceInfoAddr, size_t transportResourceInfoSize, HcclAlgoAttr &algoAttr,
    HcclTopoAttr &topoAttr)
    : topoType_(TopoType::TOPO_TYPE_COMMON), dispatcher_(dispatcher),
      vDispatcher_(vDispatcher), notifyPool_(notifyPool), netDevCtxMap_(netDevCtxMap),
      queueNotifyManager_(queueNotifyManager), workSpaceRes_(workSpaceRes), cclBufferManager_(cclBufferManager),
      transportResourceInfoAddr_(transportResourceInfoAddr), transportResourceInfoSize_(transportResourceInfoSize),
      deterministic_(static_cast<u8>(GetExternalInputHcclDeterministic()))
{
    SetAlgoAttr(algoAttr);
    SetTopoAttr(topoAttr);

    topoAttr_ = std::move(topoAttr);
    algoAttr_ = std::move(algoAttr);

    s32 deviceLogicId = 0;
    if (hrtGetDevice(&deviceLogicId) != HCCL_SUCCESS) {
        HCCL_INFO("start hccl resources build:no get deviceLogicId[%d]", deviceLogicId);
        return;
    }
    if ((static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) || (deviceLogicId < 0)) {
        HCCL_WARNING("start hccl resources build:get fail deviceLogicId[%d]", deviceLogicId);
        return;
    }

    HCCL_INFO("start hccl resources build:get deviceLogicId[%d]", deviceLogicId_);
    instanceRef_[deviceLogicId].Ref();
    if (SalGetBareTgid(&pid_) != HCCL_SUCCESS) {
        HCCL_INFO("get pid fail");
        return;
    }
}

hcclImpl::~hcclImpl()
{
    HCCL_INFO("start hccl resources destruction:deviceLogicId[%d]", deviceLogicId_);

    UnRegisterToHeartBeatP2P();
    UnRegisterToHeartBeat();

    WaitCommThread(commThreadPtrLevel0_);
    WaitCommThread(commThreadPtrLevel1_);
    WaitCommThread(commThreadPtrLevel2_);

    /* 销毁通信域关联资源 */
    for (auto &iter : tagCommInfo_) {
        DestroyOuterComm(iter.first);
        DestroyInnerComm(iter.first);
        DestroyIntraServerComm(iter.first);
        // Workspace资源需要根据tag销毁（临时方案）
        workSpaceRes_->DestroyWorkspaceResource(iter.first);
    }

    cclBufferManager_.ReleaseAlltoAllvParaBuffer();

    for (auto &inner_stream_info : tagStreamInfo_) {
        if (ReleaseSignal(inner_stream_info.second) != HCCL_SUCCESS) {
            HCCL_WARNING("tag[%s],signal is not released successfully", inner_stream_info.first.c_str());
        }
        (void)StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(inner_stream_info.second.ringStreams);
    }

    tagCommInfo_.clear();
    tagStreamInfo_.clear();
    commMeshPtr_.reset();
    commMeshLevel2_.reset();
    commMeshMap_.clear();

    commFactory_ = nullptr;

    if ((static_cast<u32>(deviceLogicId_) >= MAX_MODULE_DEVICE_NUM) || (deviceLogicId_ < 0)) {
        HCCL_WARNING("start hccl resources destruction:get fail deviceLogicId[%d]", deviceLogicId_);
        return;
    }

    if (instanceRef_[deviceLogicId_].Unref() == 0) {
        std::unique_lock<std::mutex> lock(inOutPutTempMemMutex_[deviceLogicId_]);
        inOutPutTempMem_[deviceLogicId_].free();
    }
}

void hcclImpl::SetAlgoAttr(HcclAlgoAttr &algoAttr)
{
    isHaveCpuRank_ = algoAttr.isHaveCpuRank;
    inlineReduceSwitchOn_ = algoAttr.inlineReduceSwitchOn;
    isUsedRdmaOuter_ = algoAttr.isUsedRdmaOuter;
    isUsedInterHccsMode_ = algoAttr.isUsedInterHccsMode;

    identifier_ = algoAttr.identifier;
    collectiveId_ = algoAttr.collectiveId;

    nicDeployment_ = algoAttr.nicDeployment;
    commWorkMode_ = algoAttr.commWorkMode;
    return;
}

void hcclImpl::SetTopoAttr(HcclTopoAttr &topoAttr)
{
    serverNum_= topoAttr.serverNum;
    devNumInLevel2_ = topoAttr.devNumInLevel2;
    moduleNum_ = topoAttr.moduleNum;
    deviceNumPerServer_ = topoAttr.deviceNumPerServer;
    deviceNumPerAggregation_ = topoAttr.deviceNumPerAggregation;
    multiModuleDiffDeviceNumMode_ = topoAttr.multiModuleDiffDeviceNumMode;

    meshAggregationRankSize_ = topoAttr.meshAggregationRankSize;
    isDiffDeviceModule_ = topoAttr.isDiffDeviceModule;
    isSingleMeshAggregation_= topoAttr.isSingleMeshAggregation;
    isAllRankSamePlane_ = topoAttr.isAllRankSamePlane;

    userRank_ = topoAttr.userRank;
    realUserRank_ = topoAttr.realUserRank;
    userRankSize_ = topoAttr.userRankSize;
    rankInfoList_ = topoAttr.rankInfoList;
    hbRankInfoList_ = topoAttr.hbRankInfoList;

    devicePhyId_ = topoAttr.devicePhyId;
    deviceLogicId_ = topoAttr.deviceLogicId;
    useSdidForDeviceId_ = topoAttr.useSdidForDeviceId;
    deviceType_ = topoAttr.deviceType;
    isStandardCard_ = topoAttr.isStandardCard;
    is310PDuoCard_ = topoAttr.is310PDuoCard;

    nicList_ = topoAttr.nicList;
    pairLinkCounter_ = topoAttr.pairLinkCounter;
    pairLinkInfo_ = topoAttr.pairLinkInfo;
    isSupportRdmaLite_ = topoAttr.isSupportRdmaLite;
    return;
}

HcclResult hcclImpl::Init(bool isHeterogComm)
{
    parallelTaskLoader_.reset(static_cast<ParallelTaskLoader *>(new (std::nothrow) ParallelTaskLoader(
        deviceLogicId_, dispatcher_)));
    CHK_SMART_PTR_NULL(parallelTaskLoader_);

    if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) {
        tinySendRecvMem_ = DeviceMem::alloc(TINY_MEMORY_SIZE);
        CHK_PTR_NULL(tinySendRecvMem_.ptr());
    }

    if (!isHeterogComm && commWorkMode_ != WorkMode::HCCL_MODE_AI_CPU) {
        // 获取算法类型
        CHK_RET(SelectAlgType(moduleNum_, deviceType_, algType_));
        // 获取拓扑类型，根据算法类型转化
        CHK_RET(GetTopoTypeByAlgType(algType_[HcclCMDType::HCCL_CMD_ALL], deviceType_, topoType_));
    } else {
        topoType_ = TopoType::TOPO_TYPE_HETEROG;
    }

    commFactory_.reset(new (std::nothrow) CommFactory(identifier_, userRank_, userRankSize_, dispatcher_, notifyPool_,
        netDevCtxMap_, isUsedRdmaOuter_, topoType_, deviceType_, rankInfoList_, nicDeployment_, isHeterogComm,
        isDiffDeviceModule_,
        transportResourceInfoAddr_, transportResourceInfoSize_,
        meshAggregationRankSize_, isHaveCpuRank_, isUsedInterHccsMode_, useSdidForDeviceId_));
    CHK_SMART_PTR_NULL(commFactory_);
    CHK_RET(commFactory_->Init());

    HCCL_INFO("hcclImpl init success.");
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CheckAlgType(const AlgType algType)
{
    if (HCCL_ALGO_TYPE_MAP.count(algType) == 0) {
        HCCL_ERROR("[Check][AlgType]errNo[0x%016llx] algType[%s] is not supported", HCCL_ERROR_CODE(HCCL_E_PARA),
            HcclAlg::AlgTypeToStr(algType).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetTopoTypeByAlgType(const AlgType &algType, const DevType deviceType,
    TopoType &topoType)
{
    CHK_RET(CheckAlgType(algType));

    CHK_RET(CheckDeviceType(deviceType));

    const AlgTypeLevel0 algLevel0 = GetLevel0AlgType(algType);
    switch (algLevel0) {
        case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
            topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_8P_RING:
            topoType = TopoType::TOPO_TYPE_8P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_MESH:
            topoType = TopoType::TOPO_TYPE_4P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_2P_MESH:
            topoType = TopoType::TOPO_TYPE_2P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING:
            topoType = TopoType::TOPO_TYPE_NP_SINGLE_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_1P_MESH:
            topoType = TopoType::TOPO_TYPE_1P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_RING:
            topoType = TopoType::TOPO_TYPE_4P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
            topoType = TopoType::TOPO_TYPE_NP_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING:
        case AlgTypeLevel0::ALG_LEVEL0_RESERVED:
            topoType = TopoType::TOPO_TYPE_COMMON;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_STAR:
            topoType = TopoType::TOPO_TYPE_ES_MESH;
            break;
        default:
            HCCL_ERROR("[hcclImpl][GetTopoTypeByAlgType]errNo[0x%016llx] case: device type[%d](0~1:V910),"
                " algorithm[%s] is not support",
                HCCL_ERROR_CODE(HCCL_E_PARA), deviceType, HcclAlg::AlgTypeToStr(algType).c_str());
            return HCCL_E_PARA;
    }

    HCCL_INFO("[hcclImpl][GetTopoTypeByAlgType]algtype[%s], devicetype[%d],topotype[%d] is selected",
        HcclAlg::AlgTypeToStr(algType).c_str(), deviceType, topoType);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SelectCurrOpAlgType(
    u32 moduleNum, const DevType deviceType, HcclCMDType opType, std::map<HcclCMDType, AlgType>& algType)
{
    AlgTypeLevel0 algType0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    AlgTypeLevel1 algType1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
    AlgTypeLevel2 algType2 = AlgTypeLevel2::ALG_LEVEL2_RESERVED; // 第2层拓扑算法, 待梳理后考虑是否和第0层、第1层算法归一

    if (Is310P3Common()) {
        algType[opType] = AlgType::ALG_DEFAULT;
    } else if (multiModuleDiffDeviceNumMode_) { // 多server不同卡模式，设置为单层拓扑类型
        algType[opType] = AlgType::ALG_DEFAULT;
        isAlgoLevel1Default_[opType] = false;
        if (GetExternalInputHcclAlgoConfig(opType)[HCCL_ALGO_LEVEL_0] != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT ||
            GetExternalInputHcclAlgoConfig(opType)[HCCL_ALGO_LEVEL_1] != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            HCCL_WARNING("multiModuleDiffDeviceNumMode_ is [%d], algorithm type [%d] is selected by force.", \
                         multiModuleDiffDeviceNumMode_, algType[opType]);
        }
    } else if (isHaveCpuRank_) {
        algType[opType] = AlgType::ALG_NP_STAR;
    } else {
        CHK_RET(SetAlgoLevel0(GetExternalInputHcclAlgoConfig(opType)[HCCL_ALGO_LEVEL_0], algType0));
        CHK_RET(SetAlgoLevel1(GetExternalInputHcclAlgoConfig(opType)[HCCL_ALGO_LEVEL_1], moduleNum, algType1, opType));
        CHK_RET(SetAlgoLevel2(GetExternalInputHcclAlgoConfig(opType)[HCCL_ALGO_LEVEL_2], algType2));

        algType[opType] = AlgType((static_cast<u32>(algType1) << HCCL_LEVEL_ALGO_WIDTH) + static_cast<u32>(algType0));

        if (!isStandardCard_ && deviceType != DevType::DEV_TYPE_910B) {
            if (nicList_.size() != DEVICE_EIGHT && deviceNumPerAggregation_ == DEVICE_EIGHT &&
                algType0 != AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
                HCCL_ERROR("[Set][AlgType]nicSize[%zu] error, algType is not 8P ring.", nicList_.size());
                return HCCL_E_PARA;
            }
        }
    }

    auto level0Iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType0);
    CHK_PRT_RET(level0Iter == HCCL_ALGO_LEVEL0_NAME_MAP.end(), HCCL_ERROR("level0: algType0[%u] is invalid.",
        algType0), HCCL_E_INTERNAL);
    auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
    CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
        algType1), HCCL_E_INTERNAL);
    auto level2Iter = HCCL_ALGO_LEVEL2_NAME_MAP.find(algType2);
    CHK_PRT_RET(level2Iter == HCCL_ALGO_LEVEL2_NAME_MAP.end(),
        HCCL_ERROR("level2: algType2[%u] is invalid.", algType2), HCCL_E_INTERNAL);
    HCCL_RUN_INFO("Device Type[%d], average device count[%u], HccsNum[%d], SIONum[%d], HCCS_SW_NUM[%d], " \
        "optype[%u] algorithm type[%d] is selected:level0: using[%s], level1: using[%s], level2: using[%s]",
        deviceType, deviceNumPerAggregation_, pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size(),
        pairLinkInfo_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size(),
        pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size(),
        opType, algType[opType], level0Iter->second.c_str(), level1Iter->second.c_str(), level2Iter->second.c_str());
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SelectAlgType(u32 moduleNum, const DevType deviceType, std::map<HcclCMDType, AlgType>& algType)
{
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        CHK_RET(SelectCurrOpAlgType(moduleNum, deviceType, static_cast<HcclCMDType>(opType), algType));
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetAlgoLevel0(HcclAlgoType algoConfig, AlgTypeLevel0 &algType)
{
    if (isStandardCard_) {
        CHK_RET(SetAlgoLevel0StandardCard(algoConfig, algType));
    } else {
        CHK_RET(SetAlgoLevel0Module(algoConfig, algType));
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetAlgoLevel1(HcclAlgoType algoConfig, u32 moduleNum, AlgTypeLevel1 &algType, HcclCMDType opType)
{
    HcclAlgoType algoConfigShadow = algoConfig;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel1::ALG_LEVEL1_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_INFO("server num[%u]: level1:ring algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_INFO("server num[%u]: level1:nhr algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
            HCCL_INFO("server num[%u]: level1:nhr_v1 algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel1::ALG_LEVEL1_NB;
            HCCL_INFO("server num[%u]: level1:nb algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            HCCL_INFO("server num[%u]: level1:pipeline algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH:
        case HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE:
            HCCL_WARNING("level1:fullmesh algo is not suported. the config is ignored.");
        default:
            algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            break;
    }

    HCCL_DEBUG("[hcclImpl][SetAlgoLevel1] algType[%u], deviceType_[%u], workflowmode[%u]", algType, deviceType_,
        GetWorkflowMode());
    if (algType == AlgTypeLevel1::ALG_LEVEL1_PIPELINE && (deviceType_ != DevType::DEV_TYPE_910B ||
            GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
        algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        HCCL_WARNING("hccl algorithm: there are %u server in level1, config pipeline algo failed.", moduleNum);
    }

    if (algoConfigShadow == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
        if (deviceType_ == DevType::DEV_TYPE_910B) {
            isAlgoLevel1Default_[opType] = true;
        }
        CHK_RET(GetDefaultAlgoLevel1V1(moduleNum, algType));
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetAlgoLevel2(HcclAlgoType algoConfig, AlgTypeLevel2 &algType)
{
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel2::ALG_LEVEL2_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel2::ALG_LEVEL2_RING;
            break;
        default: {
            if (devNumInLevel2_ >=  HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
                // server 数为 8 以上：使用 HD 算法
                algType = AlgTypeLevel2::ALG_LEVEL2_HD;
            } else {
                // server 数为 2 的非整数次幂：使用 RING 算法
                // server 数为 2 的整数次幂：使用 HD 算法
                algType = (((devNumInLevel2_ & (devNumInLevel2_ - 1)) != 0) || (devNumInLevel2_ == 1)) ?
                    AlgTypeLevel2::ALG_LEVEL2_RING :
                    AlgTypeLevel2::ALG_LEVEL2_HD;
            }
            break;
        }
    }
    HCCL_DEBUG("[hcclImpl][SetAlgoLevel2]algType[%u], deviceType_[%u], devNumInLevel2[%u]",
        algType, deviceType_, devNumInLevel2_);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetAlgoLevel0StandardCard(HcclAlgoType algoConfig, AlgTypeLevel0 &algType)
{
    if (algoConfig == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
        algType = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        return HCCL_SUCCESS;
    }

    if (algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT && algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_NA) {
        HCCL_WARNING("level0:%d algo is not suported. the config is ignored.", algoConfig);
    }

    CHK_RET(GetDefaultAlgoLevel0StandardCard(algType));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetAlgoLevel0Module(HcclAlgoType algoConfig, AlgTypeLevel0 &algType)
{
    if (algoConfig == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
        algType = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        return HCCL_SUCCESS;
    }

    if (algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT && algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_NA) {
        HCCL_WARNING("level0:%d algo is not suported. the config is ignored.", algoConfig);
    }

    CHK_RET(GetDefaultAlgoLevel0Module(algType));
    return HCCL_SUCCESS;
}

bool hcclImpl::IsHCCSSWNumEqualToTwiceSIONum()
{
    u32 hccsSWNum = pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)];
    u32 sioNum = pairLinkCounter_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)];
    HCCL_DEBUG(
        "In pairLinkCounter_, the hccsSWNum is [%lu], the sioNum is [%lu], the deviceNumPerAggregation_ is [%lu]",
        hccsSWNum, sioNum, deviceNumPerAggregation_);
    if (hccsSWNum == 0) {
        return false;
    }
    if (sioNum == 0) {
        return false;
    }
    // The following 2 means that the device has no HCCS_SW link with itself and its companion linked by same SIO link.
    return (hccsSWNum == ((deviceNumPerAggregation_ - 2) * deviceNumPerAggregation_)) &&
           (sioNum == deviceNumPerAggregation_);
}

HcclResult hcclImpl::GetDefaultAlgoLevel0Module(AlgTypeLevel0 &algType)
{
    if (deviceNumPerAggregation_ == DEVICE_EIGHT) {
        algType = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    } else if (deviceNumPerAggregation_ == DEVICE_FOUR) {
        algType = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    } else if (deviceNumPerAggregation_ == DEVICE_TWO) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else if (deviceNumPerAggregation_ == DEVICE_ONE) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else {
        algType = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
    }

    if ((pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] ==
        deviceNumPerAggregation_ * (deviceNumPerAggregation_ - 1) ||
        pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] ==
        FACTOR_NUM_TWO * deviceNumPerAggregation_ * (deviceNumPerAggregation_ - 1)) &&
        deviceType_ == DevType::DEV_TYPE_910B) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
        HCCL_DEBUG("[GetDefaultAlgoLevel0Module] AlgTypeLevel0 is set to ALG_LEVEL0_NP_MESH (HCCS links is enabled).");
    }

    if (deviceType_ == DevType::DEV_TYPE_910_73) {
        algType = IsHCCSSWNumEqualToTwiceSIONum() ? AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING :
                                                    AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
        if ((algType == AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING) && GetExternalInputEnableRdmaSdmaConcurrent()) {
            SetRdmaSdmaConcurrentDisable();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetDefaultAlgoLevel0StandardCard(AlgTypeLevel0 &algType) const
{
    if (deviceNumPerAggregation_ == DEVICE_TWO) {
        if ((deviceType_ == DevType::DEV_TYPE_910B)) {
            algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
        } else {
            algType = AlgTypeLevel0::ALG_LEVEL0_2P_MESH;
        }
    } else if (deviceNumPerAggregation_ > DEVICE_TWO && deviceNumPerAggregation_ <= DEVICE_EIGHT) {
        // 随标卡支持rank数变更
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else if (deviceNumPerAggregation_ == DEVICE_ONE) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else {
        HCCL_ERROR("in standaed card[num %u] there is no supported algo.", deviceNumPerAggregation_);
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[GetDefaultAlgoLevel0StandardCard] AlgTypeLevel0 is set to [%u].", algType);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetAlgType(AlgType &algType, HcclCMDType opType)
{
    opType = (algType_.find(opType) == algType_.end() ? HcclCMDType::HCCL_CMD_INVALID : opType);
    algType = algType_[opType];
    CHK_RET(CheckAlgType(algType));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetAlgTypeDirect(AlgType &algType, HcclCMDType opType)
{
    opType = (algType_.find(opType) == algType_.end() ? HcclCMDType::HCCL_CMD_INVALID : opType);
    algType = algType_[opType];
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetDefaultAlgoLevel1V1(u32 moduleNum, AlgTypeLevel1 &algType) const
{
    if (moduleNum >=  HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
        // server 数为 8 以上：使用 HD 算法
        algType = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else {
        // server 数为 2 的非整数次幂：使用 RING 算法
        // server 数为 2 的整数次幂：使用 HD 算法
        algType = (((moduleNum & (moduleNum - 1)) != 0) || (moduleNum == 1)) ?
            AlgTypeLevel1::ALG_LEVEL1_RING :
            AlgTypeLevel1::ALG_LEVEL1_HD;
    }
    HCCL_INFO("[hcclImpl][GetDefaultAlgoLevel1V1] algType[%u], moduleNum[%u]", algType, moduleNum);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::ReleaseCommInfos()
{
    auto iter = tagCommInfo_.begin();
    while (iter != tagCommInfo_.end()) {
        for (auto& comm : iter->second.commInner) {
            if (comm != nullptr) {
                CHK_RET(comm->DeInit());
            }
        }
        iter++;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::ActiveRingStreams(const std::string& tag, Stream &stream)
{
    HcclResult ret = HCCL_SUCCESS;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
        //  激活stream
        std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
        auto iterRank = tagStreamInfo_.find(tag);
        CHK_PRT_RET(iterRank == tagStreamInfo_.end(),
            HCCL_ERROR("[HcclImpl][ActiveRingStreams]errNo[0x%016llx] tag[%s] can't find in stream info",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_NOT_FOUND);
        mutiStreamLock.unlock();
        innerStreamInfo_t streamInfo = (iterRank->second);
        for (u32 streamIndex = 0; streamIndex < streamInfo.ringStreams.size(); streamIndex++) {
            ret = StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                streamInfo.ringStreams[streamIndex].ptr(), stream.ptr());
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[HcclImpl][ActiveRingStreams]stream[%u] active failed,return[%d]", streamIndex, ret), ret);
        }
    }
    return HCCL_SUCCESS;
}

bool hcclImpl::SupportDeterministicOptim() const
{
    bool support = isSingleMeshAggregation_ &&
                   deviceNumPerAggregation_ > DEVICE_TWO &&
                   deviceType_ == DevType::DEV_TYPE_910B &&
                   deterministic_ == DETERMINISTIC_CONFIG_ENABLE &&
                   GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB;
    return support;
}

HcclResult hcclImpl::CreateAlltoAllVCommMem(DeviceMem& inputMem, DeviceMem& outputMem) const
{
    std::unique_lock<std::mutex> lock(inOutPutTempMemMutex_[deviceLogicId_]);
    if (inputMem.ptr() == nullptr) {
        if (inOutPutTempMem_[deviceLogicId_].ptr() == nullptr) {
            inOutPutTempMem_[deviceLogicId_] = DeviceMem::alloc(tinyMemSizeForTransportCreation);
            CHK_PTR_NULL(inOutPutTempMem_[deviceLogicId_].ptr());
        }
        inputMem = inOutPutTempMem_[deviceLogicId_].range(0, inOutPutTempMem_[deviceLogicId_].size());
    }
    if (outputMem.ptr() == nullptr) {
        if (inOutPutTempMem_[deviceLogicId_].ptr() == nullptr) {
            inOutPutTempMem_[deviceLogicId_] = DeviceMem::alloc(tinyMemSizeForTransportCreation);
            CHK_PTR_NULL(inOutPutTempMem_[deviceLogicId_].ptr());
        }
        outputMem = inOutPutTempMem_[deviceLogicId_].range(0, inOutPutTempMem_[deviceLogicId_].size());
    }
    lock.unlock();
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateCommForNoScratchAlltoall(
    const std::string &tag, DeviceMem &sendBuf, DeviceMem &recvBuf, DeviceMem scratchMem)
{
    DeviceMem inputMem = sendBuf;
    DeviceMem outputMem = recvBuf;
    auto it = tagCommInfo_.find(tag);
    if (it == tagCommInfo_.end()) {
        CommInfo commInfo;
        std::unique_ptr<CommBase> curCommOuter;
        std::unique_ptr<CommBase> curCommInner;
        auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
        auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CommParaInfo commParaMesh0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, outCCLbuffer, outCCLbuffer, commParaMesh0, commInfo.commOuter));

            CommParaInfo commParaMesh1(COMM_MESH_L1, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, outCCLbuffer, commParaMesh1, commInfo.commInner));
        } else {
            CommParaInfo commParaMesh0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, scratchMem, outputMem, commParaMesh0, commInfo.commOuter));

            CommParaInfo commParaMesh1(COMM_MESH_L1, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inputMem, scratchMem, commParaMesh1, commInfo.commInner));
        }
        tagCommInfo_.insert(std::pair<string, CommInfo>(tag, std::move(commInfo)));
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateCommForAlltoallVStaged(
    const std::string &tag, DeviceMem &sendBuf, DeviceMem &recvBuf, DeviceMem &scratchMem, bool alltoallReadOnly)
{
    // 将网卡初始化判断，提到上层调用，减少无必要的循环依赖。
    DeviceMem inputMem = sendBuf;
    DeviceMem outputMem = recvBuf;
    CHK_RET(CreateAlltoAllVCommMem(inputMem, outputMem));
    auto it = tagCommInfo_.find(tag);
    if (it == tagCommInfo_.end() || needRecreateAlltoallComm_) {
        tagCommInfo_.erase(tag);
        tagStreamInfo_.erase(tag);
        CommInfo commInfo;
        std::unique_ptr<CommBase> curCommOuter;
        std::unique_ptr<CommBase> curCommInner;
        auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
        auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
            !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
            CommParaInfo commParaMesh0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, outCCLbuffer, commParaMesh0, commInfo.commOuter));

            CommParaInfo commParaMesh1(COMM_MESH_L1, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, outCCLbuffer, commParaMesh1, commInfo.commInner));

            CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, outCCLbuffer, commParaLevel2, commInfo.commLevel2));
        } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
            isAlltoAllZCopyMode_) { // 单算子 && ZCopy模式
            if (isSingleMeshAggregation_) {
                CommParaInfo commParaMesh0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
                CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, outCCLbuffer, commParaMesh0,
                    commInfo.commOuter));
            } else {
                CommParaInfo commParaMesh0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
                CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, (alltoallReadOnly ? outCCLbuffer : scratchMem),
                    commParaMesh0, commInfo.commOuter));

                CommParaInfo commParaMesh1(COMM_MESH_L1, CommType::COMM_TAG_MESH);
                CHK_RET(commFactory_->CreateCommPlane(tag, scratchMem, outCCLbuffer, commParaMesh1,
                    commInfo.commInner));
            }
            CommParaInfo commParaInfo(COMM_LEVEL2, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, outCCLbuffer, commParaInfo, commInfo.commLevel2));
        } else {
            CommParaInfo commParaMesh0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inputMem, scratchMem, commParaMesh0, commInfo.commOuter));

            CommParaInfo commParaMesh1(COMM_MESH_L1, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, scratchMem, outputMem, commParaMesh1, commInfo.commInner));

            CommParaInfo commParaInfo(COMM_LEVEL2, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inputMem, outputMem, commParaInfo, commInfo.commLevel2));
        }
        tagCommInfo_.insert(std::pair<string, CommInfo>(tag, std::move(commInfo)));
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::BuildAlltoAllVScratchMem(const std::string &tag, u64 workSpaceMemSize)
{
    std::unique_lock<std::mutex> lock(inOutPutTempMemMutex_[deviceLogicId_]);
    if (workSpaceMemSize == 0 && inOutPutTempMem_[deviceLogicId_].ptr() == nullptr) {
        inOutPutTempMem_[deviceLogicId_] = DeviceMem::alloc(tinyMemSizeForTransportCreation);
        CHK_PTR_NULL(inOutPutTempMem_[deviceLogicId_].ptr());
    }
    lock.unlock();
    DeviceMem tmpMem;
    if (scratchMemMap_.find(tag) == scratchMemMap_.end()) {
        if (workSpaceMemSize == 0) {
            HCCL_DEBUG("[BuildAlltoAllVScratchMem] workSpaceMemSize is zero!");
            tmpMem = inOutPutTempMem_[deviceLogicId_].range(0, inOutPutTempMem_[deviceLogicId_].size());
            scratchMemMap_.insert(std::pair<std::string, DeviceMem>(tag, tmpMem));
        } else {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                tmpMem = DeviceMem::alloc(max(workSpaceMemSize, cclBufferManager_.GetInCCLbufferSize()));
            } else {
                tmpMem = workSpaceRes_->AllocDeviceMem(tag, workSpaceMemSize);
            }
            scratchMemMap_.insert(std::pair<std::string, DeviceMem>(tag, std::move(tmpMem)));
        }
    } else {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            u64 curSize = scratchMemMap_[tag].size();
            if (curSize >= workSpaceMemSize) {
                return HCCL_SUCCESS;
            }
            scratchMemMap_.erase(tag);
            needRecreateAlltoallComm_ |= isAlltoAllZCopyMode_;
            u64 nextworkSpaceMemSize = max(workSpaceMemSize, cclBufferManager_.GetInCCLbufferSize());
            HCCL_INFO("[Rebuild][workSpaceMem] workSpaceMem expand, cur size[%llu], next size[%llu]",
                curSize, nextworkSpaceMemSize);
            tmpMem = DeviceMem::alloc(nextworkSpaceMemSize);
            scratchMemMap_.insert(std::pair<std::string, DeviceMem>(tag, std::move(tmpMem)));
        }
    }
    HCCL_INFO("[BuildAlltoAllVScratchMem] tmpMem ptr[%p], size[%llu]", tmpMem.ptr(), tmpMem.size());
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateP2pComm(const std::string &tag, CommInfo &commInfo,
    DeviceMem &inOutMem, u32 peerUserRank)
{
    CommParaInfo commP2P(COMM_COMBINE, CommType::COMM_TAG_P2P);
    commP2P.peerUserRank = peerUserRank;
    CHK_RET(commFactory_->CreateCommPlane(tag, inOutMem, inOutMem, commP2P, commInfo.commP2P));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::WaitCommThread(std::unique_ptr<std::thread> &ThreadPtr) const
{
    // 若线程指针为空，为此线程从未被拉起使能，不返回异常日志
    if (ThreadPtr != nullptr && ThreadPtr->joinable()) {
        ThreadPtr->join(); // 等待线程执行完毕
        CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::InitMultiStreamResource(const std::string &tag, innerStreamInfo_t &streamInfo, AlgType algType,
    bool isAicpuModeEn, bool isBatchSendRecv, u32 ringNum)
{
    if (!isBatchSendRecv) {
        switch (algType) {
            case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
            case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
            case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
                if (deviceType_ == DevType::DEV_TYPE_910_73) {
                    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                        streamInfo.ringNum
                            = OUTER_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
                    } else {
                        streamInfo.ringNum
                            = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
                    }
                }
                break;
            case AlgType::ALG_DOUBLE_RING_PLUS_RING:
            case AlgType::ALG_DOUBLE_RING_PLUS_HD:
            case AlgType::ALG_DOUBLE_RING_PLUS_NHR:
                // 当前这两种AlgType只支持910_73场景
                if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    streamInfo.ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
                } else {
                    streamInfo.ringNum
                        = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
                }
                break;
            case AlgType::ALG_8P_RING_PLUS_HD:
            case AlgType::ALG_8P_RING_PLUS_RING:
            case AlgType::ALG_8P_RING_PLUS_NHR:
            case AlgType::ALG_8P_RING_PLUS_NHR_V1:
            case AlgType::ALG_8P_RING_PLUS_NB:
            case AlgType::ALG_8P_RING_PLUS_PIPELINE:
                streamInfo.ringNum = OUTER_PLANE_NUM_IN_8PRING;
                break;
            case AlgType::ALG_NP_MESH_PLUS_RING:
            case AlgType::ALG_NP_MESH_PLUS_HD:
            case AlgType::ALG_NP_MESH_PLUS_NHR:
            case AlgType::ALG_NP_MESH_PLUS_NHR_V1:
            case AlgType::ALG_NP_MESH_PLUS_NB:
            case AlgType::ALG_2P_MESH_PLUS_RING:
            case AlgType::ALG_2P_MESH_PLUS_HD:
            case AlgType::ALG_2P_MESH_PLUS_NHR:
            case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
            case AlgType::ALG_2P_MESH_PLUS_NB:
            case AlgType::ALG_WHOLE_RING_PLUS_PIPELINE:
            case AlgType::ALG_4P_MESH_PLUS_PIPELINE:
            case AlgType::ALG_2P_MESH_PLUS_PIPELINE:
            case AlgType::ALG_1P_MESH_PLUS_PIPELINE:
            case AlgType::ALG_4P_RING_PLUS_PIPELINE:
            case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
            case AlgType::ALG_NP_DOUBLE_RING_PLUS_PIPELINE:
            case AlgType::ALG_NP_MESH_PLUS_PIPELINE:
            case AlgType::ALG_NP_STAR_PLUS_PIPELINE:
                if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
                    (deviceType_ == DevType::DEV_TYPE_910B) && isSingleMeshAggregation_) {
                    streamInfo.ringNum = deviceNumPerAggregation_;
                } else if ((deviceType_ == DevType::DEV_TYPE_910_73) && (isAicpuModeEn == true)) {
                    streamInfo.ringNum = deviceNumPerAggregation_;
                } else if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
                    (deviceType_ == DevType::DEV_TYPE_910B) && UseInterServerPipelineAlgo(algType)) {
                    streamInfo.ringNum = deviceNumPerAggregation_ + 1; /* pipeline ring场景下性能优化 */
                } else {
                    streamInfo.ringNum = deviceNumPerAggregation_ - 1;
                }
                break;
            case AlgType::ALG_4P_MESH_PLUS_HD:
            case AlgType::ALG_4P_MESH_PLUS_RING:
            case AlgType::ALG_4P_MESH_PLUS_NHR:
            case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
            case AlgType::ALG_4P_MESH_PLUS_NB:
                streamInfo.ringNum = OUTER_PLANE_NUM_IN_4PMESH;
                break;
            default:
                break;
        }
    } else {
        // 批量send/recv需要2条流
        streamInfo.ringNum = 2;
    }

    if (GetExternalInputEnableRdmaSdmaConcurrent() && deviceType_ == DevType::DEV_TYPE_910_73) {
        streamInfo.ringNum += RDMA_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }

    if (piplineSliceNum_ > 0) {
        streamInfo.ringNum++; // 流水并行算法, Server间需要额外一条从流
    }
    streamInfo.ringNum = std::max(streamInfo.ringNum, ringNum);
    HCCL_INFO("algType:[%u] InitMultiStreamResource streamInfo.ringNum %u", algType, streamInfo.ringNum);
    if (streamInfo.ringNum > 1) {
        u32 resNum = streamInfo.ringNum - 1;
        streamInfo.ringStreams.resize(resNum);    // 只有主环以外会用,减去主环1
        streamInfo.ringSignal.resize(resNum);     // 只有主环以外会用,减去主环1
        streamInfo.ringSignalAux.resize(resNum);  // 只有主环以外会用,减去主环1
        streamInfo.ringThreadsManage.resize(resNum);
        streamInfo.tidInfo.resize(resNum);

        for (auto &signal : streamInfo.ringSignal) {
            signal = nullptr;
        }
        for (auto &signal : streamInfo.ringSignalAux) {
            signal = nullptr;
        }

        u32 notifyNum = resNum * 2; // 2:Signal + SignalAux
        std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
        CHK_RET(queueNotifyManager_->Alloc(tag, notifyNum, notifys));
        for (u32 i = 0; i < resNum; i++) {
            streamInfo.ringSignal[i] = notifys[2 * i];
            streamInfo.ringSignalAux[i] = notifys[2 * i + 1];
        }
        for (u32 ringIndex = 0; ringIndex < resNum; ringIndex++) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                streamInfo.ringThreadsManage[ringIndex].reset(new (std::nothrow) ThreadManage(deviceLogicId_,
                                                                                               userRank_, dispatcher_));
                CHK_SMART_PTR_NULL(streamInfo.ringThreadsManage[ringIndex]);
                HcclResult ret = streamInfo.ringThreadsManage[ringIndex]->Init();
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Init][MultiRingResource]ringIndex[%u] ThreadManage failed,return[%d]",
                        ringIndex, ret), ret);
                streamInfo.tidInfo[ringIndex] = streamInfo.ringThreadsManage[ringIndex]->GetTid();
                HCCL_INFO("ringThreadsManage Init success[%u]", ringIndex);
            }
        }
    }
    if (isAicpuModeEn == true) {
        HCCL_INFO("aicpu resource num[%u]", streamInfo.ringNum);
        streamInfo.ringDeviceStreams.resize(streamInfo.ringNum);

        if (streamInfo.ringNum > 1) {
            u32 resNum = streamInfo.ringNum - 1;
            streamInfo.ringDeviceSignal.resize(resNum);
            streamInfo.ringDeviceSignalAux.resize(resNum);

            for (auto &signal : streamInfo.ringDeviceSignal) {
                signal = nullptr;
            }

            for (auto &signal : streamInfo.ringDeviceSignalAux) {
                signal = nullptr;
            }

            u32 notifyNum = resNum * 2; // 2:Signal + SignalAux
            std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
            CHK_RET(queueNotifyManager_->Alloc(tag, notifyNum, notifys, NotifyLoadType::DEVICE_NOTIFY));
            for (u32 i = 0; i < resNum; i++) {
                streamInfo.ringDeviceSignal[i] = notifys[2 * i];
                streamInfo.ringDeviceSignalAux[i] = notifys[2 * i + 1];
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::ReplaceCommInfoByTag(const std::string &tag, std::unique_ptr<CommInfo> &commInfo)
{
    std::unique_lock<std::mutex> replLock(commLock_);
    tagCommInfo_.erase(tag);
    tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(*commInfo)));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::PrepareCommRes(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
    Stream stream, u32 root, bool isP2p, bool isHaveCpuRank, bool isBatchSendRecv, bool meshSinglePlane, bool aivMode,
    std::set<u32> batchSendRecvtargetRanks)
{
    CHK_PRT_RET(IsExistCommRes(tag),
        HCCL_DEBUG("[HcclImpl][PrepareCommRes] tag[%s] comm has existed, do nothing", tag.c_str()),
        HCCL_SUCCESS);

    HcclUs startut = TIME_NOW();
    HCCL_INFO("[HcclImpl][PrepareCommRes] tag[%s], inputMem ptr[%p] size[%llu], outputMem ptr[%p] size[%llu], "
              "algType[%s], root[%u], isP2p[%d], isHaveCpuRank[%d], meshSinglePlane[%d], aivMode[%d]",
              tag.c_str(), inputMem.ptr(), inputMem.size(), outputMem.ptr(), outputMem.size(),
              HcclAlg::AlgTypeToStr(algType).c_str(), root, isP2p, isHaveCpuRank, meshSinglePlane, aivMode);
    CHK_RET(notifyPool_->RegisterOp(tag));

    HcclResult ret = HCCL_SUCCESS;

    do {
        // 创建通信域
        ret = CreateComm(tag, inputMem, outputMem, algType, root, isP2p, isBatchSendRecv, meshSinglePlane, aivMode,
            batchSendRecvtargetRanks);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclImpl][PrepareCommRes]errNo[0x%016llx] tag[%s], create comm failed",
            HCCL_ERROR_CODE(ret), tag.c_str()),);

        if (!isHaveCpuRank) {
            // send recv 算子不申请从流
            if (algType != AlgType::ALG_RESERVED && !Is310P3Common()) {
                ret = CreateMutiStreamRes(tag, stream, algType);
                CHK_PRT_BREAK(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[HcclImpl][PrepareCommRes]errNo[0x%016llx] tag[%s], init stream resource failed",
                    HCCL_ERROR_CODE(ret), tag.c_str()),);
            }
            if (isUseRankPort_) {
                HeartbeatPub::SetRankPortInfo(deviceLogicId_, isUseRankPort_, ranksPort_);
            }
            if (isBatchSendRecv) {
                HCCL_INFO("[HcclImpl][PrepareCommRes] BatchSendRecv skip RegisterToHeartBeat.");
            } else if (algType != AlgType::ALG_RESERVED) {
                ret = RegisterToHeartBeat();
            } else {
                ret = RegisterToHeartBeat(root, tag);
            }

            CHK_PRT_BREAK(ret != HCCL_SUCCESS,
                HCCL_ERROR("[HcclImpl][PrepareCommRes]errNo[0x%016llx] tag[%s], register heartbeat failed",
                HCCL_ERROR_CODE(ret), tag.c_str()),);
        }
    } while (0);

    if (ret != HCCL_SUCCESS) {
        s32 streamId = 0;
        (void)hrtGetStreamId(stream.ptr(), streamId);
        HCCL_ERROR("[HcclImpl][PrepareCommRes] failed, tag[%s], inputMem ptr[%p] size[%llu], outputMem ptr[%p] "\
            "size[%llu], algType[%s], streamId[%d], root[%u], isP2p[%d], isHaveCpuRank[%d], return[0x%016llx]",
            tag.c_str(), inputMem.ptr(), inputMem.size(), outputMem.ptr(), outputMem.size(),
            HcclAlg::AlgTypeToStr(algType).c_str(), streamId, root, isP2p, isHaveCpuRank, HCCL_ERROR_CODE(ret));
        (void)notifyPool_->UnregisterOp(tag);
        if (!isBatchSendRecv) {
            UnRegisterToHeartBeat();
        }
        return ret;
    }

    CHK_RET(notifyPool_->UnregisterOp(tag));

    HCCL_INFO("resource creation success, take time [%lld]us, tag[%s]",
        DURATION_US(TIME_NOW() - startut), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
    std::unique_ptr<CommInfo> &commInfo, u32 root, bool isP2p, bool isAicpuModeEn, bool isBatchSendRecv,
    bool meshSinglePlane, bool aivMode, std::set<u32> batchSendRecvtargetRanks)
{
    // Comm资源的唯一性，由上层调用保证
    // tag 多线程并行调度时唯一标识，不能为空
    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[Create][Comm]errNo[0x%016llx] tag is empty", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    commInfo.reset(new (std::nothrow) CommInfo);
    CHK_SMART_PTR_NULL(commInfo);

    DeviceMem inputMemComm = cclBufferManager_.GetCommRegMem(inputMem, MemAttr::IN_CCL_BUFFER, aivMode);
    DeviceMem outputMemComm = cclBufferManager_.GetCommRegMem(outputMem, MemAttr::OUT_CCL_BUFFER, aivMode);

    if (isP2p) {
        CHK_RET(CreateP2pComm(tag, *commInfo, inputMemComm, root));
    } else if (isAicpuModeEn && deviceType_ == DevType::DEV_TYPE_910_73) {
        // level0 mesh通信域
        std::vector<std::unique_ptr<CommBase> > commMeshL0;
        CommParaInfo commCombinePara(COMM_MESH_L0, CommType::COMM_TAG_MESH);
        commCombinePara.isAicpuModeEn = isAicpuModeEn;
        CHK_RET(commFactory_->CreateCommPlane(tag, inputMemComm, outputMemComm, commCombinePara, commInfo->commOuter));
    } else {
        CHK_RET(CreateCommByAlg(tag, algType, *commInfo, inputMemComm, outputMemComm, root, isAicpuModeEn,
            meshSinglePlane));
    }

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, AlgType algType,
    u32 root, bool isP2p, bool isBatchSendRecv, bool meshSinglePlane, bool aivMode,
    std::set<u32> batchSendRecvtargetRanks)
{
    // tag 多线程并行调度时唯一标识，不能为空
    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[Create][Comm]errNo[0x%016llx] tag is empty", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    // 作下重复的判断，在Gather等逻辑梳理清楚后，再清理
    CHK_PRT_RET(IsExistCommRes(tag),
        HCCL_DEBUG("[HcclImpl][CreateComm] tag[%s] comm has existed, do nothing", tag.c_str()),
        HCCL_SUCCESS);

    std::unique_ptr<CommInfo> commInfo = nullptr;
    HcclResult ret = CreateComm(tag, inputMem, outputMem, algType, commInfo, root, isP2p, false, isBatchSendRecv,
        meshSinglePlane, aivMode, batchSendRecvtargetRanks);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[hcclImpl][CreateComm]create comminfo by tag[%s] failed. return[%d]", tag.c_str(), ret), ret);

    // 根据上下层逻辑，这里其实只是Save/Insert。
    CHK_RET(ReplaceCommInfoByTag(tag, commInfo));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateP2PCommAsync(const std::string &tag, DeviceMem &mem, u32 peerRank, u32& status)
{
    // tag 多线程并行调度时唯一标识，不能为空
    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[Create][P2PCommAsync]tag is empty"), HCCL_E_PARA);

    std::unique_ptr<CommInfo> commInfo;
    commInfo.reset(new (std::nothrow) CommInfo);
    CHK_SMART_PTR_NULL(commInfo);

    // 新tag创建一组comm实例
    if (!IsExistCommRes(tag)) {
        commInfo->commP2P = commFactory_->CreateCommP2PAsync(tag, mem, mem, peerRank, status);
        HcclResult ret = ReplaceCommInfoByTag(tag, commInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][Comm]replace comminfo by tag[%s] failed. return[%d]", tag.c_str(), ret), ret);
    } else {
        status = HETEROG_P2P_SUCCESS;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateP2PCommQuerry(const std::string &tag, u32& status)
{
    std::unique_lock<std::mutex> replLock(commLock_);
    auto iter = tagCommInfo_.find(tag);
    if (iter != tagCommInfo_.end()) {
        CHK_RET(commFactory_->CreateCommP2PQuerry(iter->second.commP2P, status));
    } else {
        status = HETEROG_P2P_FAILED;
        HCCL_ERROR("[CreateP2PCommQuerry]querry tag[%s] comm info failed", tag.c_str());
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("CreateP2PCommQuerry %u", status);
    if (status == HETEROG_P2P_SUCCESS) {
        HCCL_INFO("CreateP2PCommQuerry connect complete.");
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetCommTypeInLevel0(const AlgType algType, const TopoType topoType, CommType &commType)
{
    if (Is310P3Common()) {
        if (algType == AlgType::ALG_NP_HD) {
            commType = CommType::COMM_TAG_HALVING_DOUBLING;
        } else {
            commType = CommType::COMM_TAG_RING_INNER;
        }
        HCCL_DEBUG("[Get][CommTypeForLevel0]The algType is %s, topoType is %d, while commType is %d",
            HcclAlg::AlgTypeToStr(algType).c_str(), topoType, commType);
        return HCCL_SUCCESS;
    }

    bool isMesh = ((topoType_ == TopoType::TOPO_TYPE_4P_MESH) || (topoType_ == TopoType::TOPO_TYPE_2P_MESH) ||
                   (topoType_ == TopoType::TOPO_TYPE_1P_MESH) || (topoType_ == TopoType::TOPO_TYPE_NP_MESH));

    // 根据算法类型创建内层拓扑
    if (algType == AlgType::ALG_NP_STAR) {
        commType =  CommType::COMM_TAG_STAR;
    } else if (isMesh) {
        commType = CommType::COMM_TAG_MESH;
    } else {
        commType = CommType::COMM_TAG_RING_INNER;
    }
    HCCL_DEBUG("[Get][CommTypeForLevel0]The algType is %s, topoType is %d, while commType is %d",
        HcclAlg::AlgTypeToStr(algType).c_str(), topoType, commType);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetCommTypeInLevel1(const AlgType algType, CommType &commType)
{
    // 根据算法类型创建内层拓扑
    switch (algType) {
        case AlgType::ALG_DEFAULT: {
            commType = CommType::COMM_TAG_RING_COMBINED;
            break;
        }

        case AlgType::ALG_DOUBLE_RING_PLUS_HD:
        case AlgType::ALG_NP_MESH_PLUS_HD:
        case AlgType::ALG_8P_RING_PLUS_HD:
        case AlgType::ALG_4P_MESH_PLUS_HD:
        case AlgType::ALG_2P_MESH_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_1P_MESH_PLUS_HD:
        case AlgType::ALG_4P_RING_PLUS_HD:{
            commType =  CommType::COMM_TAG_HALVING_DOUBLING;
            break;
        }

        /* pipeline ring场景下性能优化 */
        case AlgType::ALG_WHOLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_8P_RING_PLUS_PIPELINE:
        case AlgType::ALG_4P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_2P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_1P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_4P_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_DOUBLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_MESH_PLUS_PIPELINE:
        case AlgType::ALG_NP_STAR_PLUS_PIPELINE:
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
        case AlgType::ALG_NP_MESH_PLUS_RING:
        case AlgType::ALG_8P_RING_PLUS_RING:
        case AlgType::ALG_4P_MESH_PLUS_RING:
        case AlgType::ALG_2P_MESH_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_1P_MESH_PLUS_RING:
        case AlgType::ALG_4P_RING_PLUS_RING:{
            commType = CommType::COMM_TAG_RING_INNER;
            break;
        }

        case AlgType::ALG_NP_STAR:{
            commType = CommType::COMM_TAG_STAR;
            break;
        }

        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_2P_MESH_PLUS_NHR:
        case AlgType::ALG_1P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_RING_PLUS_NHR:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
        case AlgType::ALG_DOUBLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_MESH_PLUS_NHR: {
            commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
            break;
        }

        case AlgType::ALG_WHOLE_NHR: {
            commType = CommType::COMM_TAG_WHOLE_NHR;
            break;
        }

        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_1P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_MESH_PLUS_NHR_V1: {
            commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
            break;
        }

        case AlgType::ALG_WHOLE_NHR_V1: {
            commType = CommType::COMM_TAG_WHOLE_NHR_V1;
            break;
        }

        case AlgType::ALG_8P_RING_PLUS_NB:
        case AlgType::ALG_4P_MESH_PLUS_NB:
        case AlgType::ALG_2P_MESH_PLUS_NB:
        case AlgType::ALG_1P_MESH_PLUS_NB:
        case AlgType::ALG_4P_RING_PLUS_NB:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
        case AlgType::ALG_NP_MESH_PLUS_NB: {
            commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
            break;
        }

        case AlgType::ALG_WHOLE_NB: {
            commType = CommType::COMM_TAG_WHOLE_NB;
            break;
        }

        default:
            HCCL_ERROR("[Get][CommTypeInLevel1]algType[%s] is not support", HcclAlg::AlgTypeToStr(algType).c_str());
            return HCCL_E_PARA;
    }
    HCCL_DEBUG("[Get][CommTypeInLevel1]The algType is %s, while commType is %d",
        HcclAlg::AlgTypeToStr(algType).c_str(), commType);
    return HCCL_SUCCESS;
}

CommPlane hcclImpl::GetCommPlaneInLevel1(CommType &commType)
{
    CommPlane commPlane;
    switch (commType) {
        case CommType::COMM_TAG_RING_COMBINED:
        case CommType::COMM_TAG_MESH_COMBINED: {
            commPlane = COMM_COMBINE;
            break;
        }

        case CommType::COMM_TAG_WHOLE_NB:
        case CommType::COMM_TAG_WHOLE_NHR:
        case CommType::COMM_TAG_WHOLE_NHR_V1: {
            commPlane = COMM_COMBINE_ORDER;
            break;
        }

        default: {
            commPlane = COMM_LEVEL1;
            break;
        }
    }
    HCCL_DEBUG("[Get][CommPlaneInLevel1]The commType is %d, commPlane is %d", commType, commPlane);
    return commPlane;
}

HcclResult hcclImpl::CreateCommByAlg(const std::string &tag, const AlgType algType, CommInfo &commInfo,
    DeviceMem &inputMem, DeviceMem &outputMem, u32 root, bool isAicpuModeEn, bool meshSinglePlane)
{
    CHK_RET(CheckAlgType(algType));
    CHK_RET(commFactory_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPort_, isSetHDCModeInfo_, isUseRankPort_));

    HcclResult commThreadWaitResultLevel0       = HCCL_SUCCESS;
    HcclResult commThreadWaitResultLevel0Rdma   = HCCL_SUCCESS;
    HcclResult commThreadWaitResultLevel1       = HCCL_SUCCESS;
    HcclResult commThreadWaitResultLevel1Rdma   = HCCL_SUCCESS;
    HcclResult commThreadWaitResultLevel2       = HCCL_SUCCESS;

    /* Level0通信域 */
    CommType commTypeInLevel0;
    HcclResult commThreadResultLevel0 = HCCL_SUCCESS;
    HcclResult commThreadResultLevel0Rdma = HCCL_SUCCESS;
    CHK_RET(GetCommTypeInLevel0(algType, topoType_, commTypeInLevel0));
    bool isUsedRdma = false;
    if (GetExternalInputEnableRdmaSdmaConcurrent() && deviceType_ == DevType::DEV_TYPE_910_73) {
        HCCL_INFO("commInfo create commOuterRdma/commInnerRdma for EnableRdmaSdma start");
        isUsedRdma = true;
    }

    if (Is310P3Common()) {
        if (isAicpuModeEn) {
            commTypeInLevel0 = CommType::COMM_TAG_MESH;
        }
        // level0 通信域
        CommParaInfo commParaLevel0(COMM_LEVEL0, commTypeInLevel0);
        commParaLevel0.isAicpuModeEn = isAicpuModeEn;
        std::vector<std::unique_ptr<CommBase> > commVec;
        CHK_RET(commFactory_->CreateCommPlane(tag, inputMem, outputMem, commParaLevel0, commVec));

        CHK_PRT_RET(commVec.empty() || !commVec[0],
            HCCL_ERROR("[Create][CommIntraServer]errNo[0x%016llx] tag[%s], created commIntraServer fail.",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_NOT_FOUND);
        commInfo.commIntraServer = std::move(commVec[0]);
        return HCCL_SUCCESS;
    }
    CommParaInfo commInfoLevel0(COMM_LEVEL0, commTypeInLevel0, root, INVALID_VALUE_RANKID,
        isAicpuModeEn, meshSinglePlane);
    // defalut、whole_nhr和whole_nb算法不创建外层拓扑
    if (algType != AlgType::ALG_DEFAULT && algType != AlgType::ALG_WHOLE_NHR && algType != AlgType::ALG_WHOLE_NHR_V1 &&
        algType != AlgType::ALG_WHOLE_NB) {
        commThreadPtrLevel0_.reset(new (std::nothrow) std::thread(&hcclImpl::CreateCommThread, this,
            hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem), std::ref(outputMem),
            std::ref(commInfoLevel0), std::ref(commInfo.commOuter), std::ref(commThreadResultLevel0)));
        CHK_PRT_RET(!commThreadPtrLevel0_, HCCL_ERROR("[Create][CommByAlg]commTypeInLevel0[%d] threads reset failed.",
            commInfoLevel0.commType), HCCL_E_INTERNAL);
        commThreadWaitResultLevel0 = WaitCommThread(commThreadPtrLevel0_);
        if (isUsedRdma) {
            commInfoLevel0.forceRdma = isUsedRdma;
            commThreadPtrLevel0Rdma_.reset(new (std::nothrow) std::thread(&hcclImpl::CreateCommThread, this,
                hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem), std::ref(outputMem),
                std::ref(commInfoLevel0), std::ref(commInfo.commOuterRdma),
                std::ref(commThreadResultLevel0Rdma)));
            CHK_PRT_RET(!commThreadPtrLevel0Rdma_, HCCL_ERROR("[Create][CommByAlg]commTypeInLevel0[%d]" \
                " commOuterRdma threads reset failed.", commInfoLevel0.commType), HCCL_E_INTERNAL);
            commThreadWaitResultLevel0Rdma = WaitCommThread(commThreadPtrLevel0Rdma_);
        }
    }

    /* Level1通信域 */
    HcclResult commThreadResultLevel1 = HCCL_SUCCESS;
    HcclResult commThreadResultLevel1Rdma = HCCL_SUCCESS;
    CommType commTypeInLevel1;
    CHK_RET(GetCommTypeInLevel1(algType, commTypeInLevel1));
    CommPlane commPlaneInLevel1 = GetCommPlaneInLevel1(commTypeInLevel1);
    CommParaInfo commInfoLevel1(commPlaneInLevel1, commTypeInLevel1, root, INVALID_VALUE_RANKID, isAicpuModeEn);
    if (commTypeInLevel1 != CommType::COMM_TAG_STAR) {
        commThreadPtrLevel1_.reset(new (std::nothrow) std::thread(&hcclImpl::CreateCommThread, this,
            hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem), std::ref(outputMem),
            std::ref(commInfoLevel1), std::ref(commInfo.commInner), std::ref(commThreadResultLevel1)));
        CHK_PRT_RET(!commThreadPtrLevel1_, HCCL_ERROR("[Create][CommByAlg]commTypeInLevel1[%d] threads reset failed.",
            commInfoLevel1.commType), HCCL_E_INTERNAL);
        commThreadWaitResultLevel1 = WaitCommThread(commThreadPtrLevel1_);
 
        if (isUsedRdma) {
            commInfoLevel1.forceRdma = isUsedRdma;
            commThreadPtrLevel1Rdma_.reset(new (std::nothrow) std::thread(&hcclImpl::CreateCommThread, this,
                hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem), std::ref(outputMem),
                std::ref(commInfoLevel1), std::ref(commInfo.commInnerRdma),
                std::ref(commThreadResultLevel1Rdma)));
            CHK_PRT_RET(!commThreadPtrLevel1Rdma_, HCCL_ERROR("[Create][CommByAlg]commTypeInLevel1[%d]" \
                " commInnerRdma threads reset failed.", commInfoLevel1.commType), HCCL_E_INTERNAL);
                commThreadWaitResultLevel1Rdma = WaitCommThread(commThreadPtrLevel1Rdma_);
        }
    }

    /* Level2通信域 */
    HcclResult commThreadResultLevel2 = HCCL_SUCCESS;
    CommParaInfo commInfoLevel2(COMM_LEVEL2, CommType::COMM_TAG_RING_INNER);
    commThreadPtrLevel2_.reset(new (std::nothrow) std::thread(&hcclImpl::CreateCommThread, this,
        hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem), std::ref(outputMem),
        std::ref(commInfoLevel2), std::ref(commInfo.commLevel2), std::ref(commThreadResultLevel2)));
    CHK_PRT_RET(!commThreadPtrLevel2_, HCCL_ERROR("[Create][CommByAlg]commTypeInLevel2[%d] threads reset failed.",
        commInfoLevel2.commType), HCCL_E_INTERNAL);
    commThreadWaitResultLevel2 = WaitCommThread(commThreadPtrLevel2_);

    CHK_PRT_RET(commThreadWaitResultLevel0 || commThreadWaitResultLevel1 || commThreadWaitResultLevel2 ||
    commThreadWaitResultLevel0Rdma || commThreadWaitResultLevel1Rdma,
        HCCL_ERROR("[Create][CommByAlg]wait thread failed.Level0[%d] Level1[%d] Level2[%d] Level0rdma[%d]" \
            " Level1rdma[%d]", commThreadWaitResultLevel0, commThreadWaitResultLevel1, commThreadWaitResultLevel2,
            commThreadWaitResultLevel0Rdma, commThreadWaitResultLevel1Rdma), HCCL_E_INTERNAL);

    CHK_PRT_RET(commThreadResultLevel0 || commThreadResultLevel1 || commThreadResultLevel2 ||
    commThreadResultLevel0Rdma || commThreadResultLevel1Rdma,
        HCCL_ERROR("[Create][CommByAlg]CreateComm failed. result: Level0[%d] Level1[%d] Level2[%d]" \
            " Level0rdma[%d] Level1rdma[%d].", commThreadResultLevel0, commThreadResultLevel1, commThreadResultLevel2,
            commThreadResultLevel0Rdma, commThreadResultLevel1Rdma), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateCommThread(const ErrContextPub &error_context, const std::string &tag,
    DeviceMem &inputMem, DeviceMem &outputMem, const CommParaInfo &commParaInfo,
    std::vector<std::unique_ptr<CommBase> > &commVec, HcclResult &retOut)
{
    hrtErrMSetErrorContextPub(error_context);
    retOut = hrtSetDevice(deviceLogicId_);
    CHK_PRT_RET(retOut != HCCL_SUCCESS, HCCL_ERROR("[Create][CommThread]set device[%d] failed", deviceLogicId_),
        retOut);

    retOut = commFactory_->CreateCommPlane(tag, inputMem, outputMem, commParaInfo, commVec);
    CHK_PRT_RET(retOut != HCCL_SUCCESS,
        HCCL_ERROR("[Create][CommThread]tag[%s], create comm level[%d] commType[%d] fail",
        tag.c_str(), commParaInfo.commPlane, commParaInfo.commType), retOut);

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateMutiStreamRes(const std::string &tag, Stream &stream, innerStreamInfo_t &streamInfo,
    AlgType algType, bool isAicpuModeEn, bool isBatchSendRecv, u32 ringNum)
{
    /* 多环资源初始化 */
    HcclResult ret = InitMultiStreamResource(tag, streamInfo, algType, isAicpuModeEn, isBatchSendRecv, ringNum);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Create][MutiStreamRes]tag[%s] init muti ring resource failed, return[%d]",
            tag.c_str(), ret), ret);

    CHK_RET(hccl::ProfilingManagerPub::CallMsprofReportMultiThreadInfo(streamInfo.tidInfo));

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        // GE OffloadStreamManager中set的流都是从流
        CHK_RET(workSpaceRes_->RegisterMaster(tag, stream));
        streamInfo.ringStreams = workSpaceRes_->AllocSlaveStreams(tag, streamInfo.ringNum - 1);
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(opBaseStreamManager_.RegisterMaster(stream));
        streamInfo.ringStreams =
            opBaseStreamManager_.AllocSlaves(StreamType::STREAM_TYPE_ONLINE, streamInfo.ringNum - 1);

        if (isAicpuModeEn == true) {
            if (auxRingStreamsDev_.empty()) {
                auxRingStreamsDev_.reserve(MAX_SUBSTREAM_NUM + 1);
                HCCL_DEBUG("CreateMutiStreamRes: reserve auxRingStreamsDev_[%d]", MAX_SUBSTREAM_NUM);
            }
            if (auxRingStreamsDev_.size() < streamInfo.ringNum) {
                HCCL_DEBUG(
                    "CreateMutiStreamRes:tag[%s], auxRingStreamsDev_.size[%d], less than [%d], need create new streams",
                    tag.c_str(), auxRingStreamsDev_.size(), streamInfo.ringNum);
                CHK_PRT_RET(streamInfo.ringNum > MAX_SUBSTREAM_NUM + 1,
                    HCCL_ERROR(
                        "[Create][MutiStreamRes]tag[%s] streamInfo.ringNum[%d] is larger than MAX_SUBSTREAM_NUM+1[%d].",
                        tag.c_str(), streamInfo.ringNum, MAX_SUBSTREAM_NUM + 1),
                    HCCL_E_INTERNAL);
                u32 ringNum = auxRingStreamsDev_.size();
                for (u32 ringIndex = ringNum; ringIndex < streamInfo.ringNum; ringIndex++) {
                    auxRingStreamsDev_.emplace_back(Stream(StreamType::STREAM_TYPE_DEVICE));
                    // 给device侧申请的流不需要setmode，否则rts会捕获流成员Flags为1024的异常
                }
            }
            for (u32 ringIndex = 0; ringIndex < streamInfo.ringNum; ringIndex++) {
                streamInfo.ringDeviceStreams[ringIndex] = auxRingStreamsDev_[ringIndex];
                CHK_SMART_PTR_NULL(streamInfo.ringDeviceStreams[ringIndex]);
            }
        }
    } else {
        HCCL_ERROR("[Create][MutiStreamRes]WorkflowMode[%d] invalid", GetWorkflowMode());
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET((streamInfo.ringStreams.size() != streamInfo.ringNum - 1),
        HCCL_ERROR("[Create][MutiStreamRes]tag[%s] get slave stream failed, " \
        "expect to get size [%d], but only alloc [%d].",
        tag.c_str(), streamInfo.ringNum - 1, streamInfo.ringStreams.size()), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateMutiStreamRes(const std::string &tag, Stream &stream, AlgType algType, bool isBatchSendRecv,
    u32 ringNum)
{
    std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
    CHK_PRT_RET(tagStreamInfo_.find(tag) != tagStreamInfo_.end(),
        HCCL_DEBUG("[Create][MutiStreamRes]tag[%s] is already exit, do nothing", tag.c_str()), HCCL_SUCCESS);

    innerStreamInfo_t streamInfo;
    CHK_RET(CreateMutiStreamRes(tag, stream, streamInfo, algType, false, isBatchSendRecv, ringNum));

    // 构建线程和内部流维护关系
    tagStreamInfo_.insert(std::pair<std::string, InnerStreamInfo>(tag, std::move(streamInfo)));
    mutiStreamLock.unlock();
    HCCL_INFO("[Create][MutiStreamRes]tag[%s], ringNum[%u]", tag.c_str(), streamInfo.ringNum);
    return HCCL_SUCCESS;
}

void hcclImpl::DestroyInnerComm(const std::string &tag)
{
    // vector成员是智能指针, 自动destroy
    tagCommInfo_t::iterator itr = tagCommInfo_.find(tag);
    if (itr != tagCommInfo_.end()) {
        itr->second.commInner.clear();
    }
}

void hcclImpl::DestroyOuterComm(const std::string &tag)
{
    // vector成员是智能指针, 自动destroy
    tagCommInfo_t::iterator itr = tagCommInfo_.find(tag);
    if (itr != tagCommInfo_.end()) {
        itr->second.commOuter.clear();
    }
}

void hcclImpl::DestroyIntraServerComm(const std::string &tag)
{
    tagCommInfo_t::iterator itr = tagCommInfo_.find(tag);
    if (itr != tagCommInfo_.end()) {
        itr->second.commIntraServer.reset();
    }
}

HcclResult hcclImpl::ReleaseSignal(innerStreamInfo_t &innerStream)
{
    for (auto &signal : innerStream.ringSignal) {
        if (signal != nullptr) {
            signal = nullptr;
        }
    }

    for (auto &signal : innerStream.ringSignalAux) {
        if (signal != nullptr) {
            signal = nullptr;
        }
    }

    for (auto &signal : innerStream.ringDeviceSignal) {
        if (signal != nullptr) {
            signal = nullptr;
        }
    }

    for (auto &signal : innerStream.ringDeviceSignalAux) {
        if (signal != nullptr) {
            signal = nullptr;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::RunExecutor(std::unique_ptr<CommBase> &commCombine, std::unique_ptr<ExecutorBase> &executor,
    DeviceMem &inputMem, DeviceMem &outputMem, u64 count, HcclDataType dataType,
    HcclReduceOp op, u32 root, Stream &stream) const
{
    CHK_SMART_PTR_NULL(executor);
    CHK_SMART_PTR_NULL(commCombine);

    CHK_RET(executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op, root));

    CHK_RET(commCombine->RunExecutor(executor));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::ClearOpResource(const std::string &tag)
{
    // 链接资源释放
    commMeshMap_.erase(tag);
    tagCommInfo_.erase(tag);
    UnRegisterToHeartBeat(tag);
    // stream解绑定
    auto iterStream = tagStreamInfo_.find(tag);
    if (iterStream != tagStreamInfo_.end()) {
        CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(iterStream->second.ringStreams));
    }
    tagStreamInfo_.erase(tag);
    // scratchMemMap_清理
    scratchMemMap_.erase(tag);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateCommForAlltoAllFullMesh(const std::string &tag, DeviceMem &sendBuf, DeviceMem &recvBuf)
{
     // A+X单机双module启用下，未使能RDMA不能进行一层pairWise。
    bool isDifModule = serverNum_ == 1 && isDiffDeviceModule_ && userRankSize_ > HCCL_ALLTOALLV_P2P_SIZE;
    CHK_PRT_RET(isDifModule && !isUsedRdmaOuter_,
        HCCL_ERROR("[CreateComm][AlltoAllFullMesh] not support dual modules in a single server" \
                   " when RDMA disabled "), HCCL_E_NOT_SUPPORT);

    // 将网卡初始化判断，提到上层调用，减少无必要的循环依赖。
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (commMeshPtr_ == nullptr || needRecreateAlltoallComm_) {
            auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
            auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();

            // level0 - level1 全连接通信域
            std::vector<std::unique_ptr<CommBase> > commLevelCombine;
            CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, outCCLbuffer, commCombinePara, commLevelCombine));
            commMeshPtr_ = std::move(commLevelCombine[0]);

            // level2 层通信域
            std::vector<std::unique_ptr<CommBase> > commLevel2;
            CommParaInfo commLevel2Para(COMM_LEVEL2, CommType::COMM_TAG_MESH);
            CHK_RET(commFactory_->CreateCommPlane(tag, inCCLbuffer, outCCLbuffer, commLevel2Para, commLevel2));
        }
    } else {
        DeviceMem inputMem = sendBuf;
        DeviceMem outputMem = recvBuf;
        CHK_RET(CreateAlltoAllVCommMem(inputMem, outputMem));

        // level0 - level1 全连接通信域
        std::vector<std::unique_ptr<CommBase> > commLevelCombine;
        CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
        CHK_RET(commFactory_->CreateCommPlane(tag, inputMem, outputMem, commCombinePara, commLevelCombine));
        commMeshMap_.insert(make_pair(tag, std::move(commLevelCombine[0])));

        // level2 层通信域
        std::vector<std::unique_ptr<CommBase> > commLevel2;
        CommParaInfo commLevel2Para(COMM_LEVEL2, CommType::COMM_TAG_MESH);
        CHK_RET(commFactory_->CreateCommPlane(tag, inputMem, outputMem, commLevel2Para, commLevel2));
    }

    return HCCL_SUCCESS;
}

void hcclImpl::UnRegisterToHeartBeatP2P()
{
    std::unique_lock<std::mutex> commLock(commLock_);
    for (auto iter = tagCommInfo_.begin(); iter != tagCommInfo_.end(); iter++) {
        if (iter->second.commP2P.size() > 0) {
            UnRegisterToHeartBeat(iter->first);
        }
    }
}

HcclResult hcclImpl::ParallelTaskLoaderProcess(const std::string &tag, Stream &stream, SubCommInfo &outerCommInfo,
    std::vector<Stream> &ringStreams)
{
    u32 streamIndex;
    std::vector<Stream *> streamsPtr;
    streamsPtr.resize(ringStreams.size() + 1);

    for (streamIndex = 0; streamIndex < ringStreams.size(); streamIndex++) { // StreamInfo_.ringStreams
        streamsPtr[streamIndex] = &ringStreams[streamIndex];
    }
    streamsPtr[streamIndex] = &stream;

    HCCL_INFO("[ParallelTaskLoaderProcess]main stream[%p], streams size[%u]", stream.ptr(), streamsPtr.size());

    // 准备多线程启动参数
    CHK_RET(parallelTaskLoader_->Prepare(streamsPtr, outerCommInfo));

    // 启动多线程处理
    CHK_RET(parallelTaskLoader_->StartTaskLoad());

    // 等待多线程处理结果
    CHK_RET(parallelTaskLoader_->WaitTaskLoadFinish());

    // 销毁通信域
    CHK_RET(parallelTaskLoader_->ClearTagCommInfo());
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::AddSubStreamToProfiling(const std::string &tag, HcclCMDType opType)
{
    if (((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        hccl::ProfilingManagerPub::GetAddtionInfoState() &&
        hccl::ProfilingManagerPub::GetTaskApiState())) {
        return HCCL_SUCCESS;
    }

    if (tagStreamInfo_.empty()) {
        return HCCL_SUCCESS;
    }
    innerStreamInfo_t &streamInfo = tagStreamInfo_[tag];
    for (u32 streamIndex = 0; streamIndex < streamInfo.ringStreams.size(); streamIndex++) {
        // profiling加入从环的stream
        HCCL_PROFILER_ADD_STREAM(streamInfo.ringStreams[streamIndex].ptr(), tag, streamIndex + 1, algType_[opType]);
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetAlgType(AlgType algType, HcclCMDType opType)
{
    CHK_RET(CheckAlgType(algType));
    algType_[opType] = algType;
    return HCCL_SUCCESS;
}

AlgTypeLevel0 hcclImpl::GetLevel0AlgType(const AlgType algType) const
{
    if (algType != AlgType::ALG_NP_STAR) {
        const u32 algLevel0 = static_cast<u32>(algType) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1);
        return static_cast<AlgTypeLevel0>(algLevel0);
    }

    return AlgTypeLevel0::ALG_LEVEL0_NP_STAR;
}

AlgTypeLevel1 hcclImpl::GetLevel1AlgType(const AlgType algType) const
{
    const u32 algLevel1 = (static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1);
    return static_cast<AlgTypeLevel1>(algLevel1);
}

bool hcclImpl::UseInterServerPipelineAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
}

HcclResult hcclImpl::RegisterToHeartBeat()
{
    return HeartbeatPub::RegisterToHeartBeat(deviceLogicId_, userRank_, deviceType_,
        hbRankInfoList_, collectiveId_, isUsedRdmaOuter_);
}

HcclResult hcclImpl::RegisterToHeartBeat(u32 peerRankId, const std::string& tag)
{
    return HeartbeatPub::RegisterToHeartBeat(deviceLogicId_, userRank_, deviceType_,
        hbRankInfoList_, peerRankId, tag, isUsedRdmaOuter_);
}

void hcclImpl::UnRegisterToHeartBeat()
{
    HeartbeatPub::UnRegisterToHeartBeat(deviceLogicId_, deviceType_, collectiveId_);
}

void hcclImpl::UnRegisterToHeartBeat(const std::string& tag)
{
    HeartbeatPub::UnRegisterToHeartBeat(deviceLogicId_, deviceType_, tag);
}

HcclResult hcclImpl::GetTopoType(TopoType &topoType)
{
    topoType = topoType_;
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetAlgoLevel1DefaultSwitch(bool &isAlgoLevel1Default, HcclCMDType opType)
{
    isAlgoLevel1Default = isAlgoLevel1Default_[opType];
    return HCCL_SUCCESS;
}

u32 hcclImpl::GetSubRootForScatter(const u32 root)
{
    return commFactory_->GetSubRootForScatter(root);
}

u32 hcclImpl::GetSubRootUserRank(const u32 userRank, const u32 rootUserRank)
{
    return commFactory_->GetSubRootUserRank(userRank, rootUserRank);
}

u32 hcclImpl::GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank)
{
    return commFactory_->GetSubRootUserRankWithSuperPod(userRank, rootUserRank);
}

HcclResult hcclImpl::GetCommInfo(CommInfo *&currComm, const std::string &tag)
{
    std::unique_lock<std::mutex> commLock(commLock_);
    currComm = &tagCommInfo_[tag];
    commLock.unlock();
    return HCCL_SUCCESS;
}

std::unique_ptr<CommBase>& hcclImpl::GetCommMesh()
{
    return commMeshPtr_;
}

std::unique_ptr<CommBase>& hcclImpl::GetCommMeshByTag(const std::string &tag)
{
    return commMeshMap_[tag];
}

HcclResult hcclImpl::SetScratchMem(DeviceMem &scratchMem, const std::string &tag, u64 allocMemSize)
{
    std::unique_lock<std::mutex> lock(scratchMemLock_);
    auto iter = scratchMemMap_.find(tag);
    if (iter == scratchMemMap_.end()) { /* 查找tag对应的scratch Mem,没找到申请记录新的mem */
        DeviceMem workSpaceScratchMem = workSpaceRes_->AllocDeviceMem(tag, allocMemSize);
        // map表保存整段内存
        scratchMemMap_.insert(std::pair<std::string, DeviceMem>(tag, std::move(workSpaceScratchMem)));
        scratchMem = scratchMemMap_[tag];
    } else {
        scratchMem = iter->second;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetScratchMem(DeviceMem &scratchMem, const std::string &tag)
{
    std::unique_lock<std::mutex> lock(scratchMemLock_);
    scratchMem = scratchMemMap_[tag];
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetNicSendSize(const std::string &tag, std::vector<u64> &sizeList)
{
    std::unique_lock<std::mutex> lock(nicSendSizeListLock_);
    nicSendSizeList_[tag] = sizeList;
    return HCCL_SUCCESS;
}

innerStreamInfo_t* hcclImpl::GetStreamInfo(const std::string &tag)
{
    std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
    auto iterRank = tagStreamInfo_.find(tag);
    if (iterRank == tagStreamInfo_.end()) {
        HCCL_ERROR("[hcclImpl][GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str());
        return nullptr;
    }
    return &iterRank->second;
}

// 这里的streamNum表示的是总的流数量
HcclResult hcclImpl::GetStreamThreadManage(const std::string &tag, u32 streamNum,
    std::vector<std::shared_ptr<ThreadManage>> &threadManager)
{
    std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
    auto iterRank = tagStreamInfo_.find(tag);
    if (iterRank == tagStreamInfo_.end()) {
        innerStreamInfo_t streamInfo;
        streamInfo.ringThreadsManage.resize(streamNum - 1);
        streamInfo.tidInfo.resize(streamNum - 1);
        for (u32 ringIndex = 0; ringIndex < streamNum - 1; ringIndex++) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                streamInfo.ringThreadsManage[ringIndex].reset(new (std::nothrow) ThreadManage(deviceLogicId_,
                                                                                                userRank_,
                                                                                                dispatcher_));
                CHK_SMART_PTR_NULL(streamInfo.ringThreadsManage[ringIndex]);
                HcclResult ret = streamInfo.ringThreadsManage[ringIndex]->Init();
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Init][MultiRingResource]ringIndex[%u] ThreadManage failed,return[%d]",
                        ringIndex, ret), ret);
                streamInfo.tidInfo[ringIndex] = streamInfo.ringThreadsManage[ringIndex]->GetTid();
                HCCL_INFO("ringThreadsManage Init success[%u]", ringIndex);
            }
        }
        tagStreamInfo_.insert(std::pair<std::string, InnerStreamInfo>(tag, std::move(streamInfo)));
        HCCL_INFO("[GetStreamThreadManage]tag[%s]create ThreadManage success. streamNum[%d]", tag.c_str(), streamNum);
    } else {
        threadManager = iterRank->second.ringThreadsManage;
        HCCL_INFO("[GetStreamThreadManage]tag[%s]get ThreadManage success. streamNum[%d]", tag.c_str(), streamNum);
        return HCCL_SUCCESS;
    }
    iterRank = tagStreamInfo_.find(tag);
    threadManager = iterRank->second.ringThreadsManage;
    return HCCL_SUCCESS;
}

innerStreamInfo_t* hcclImpl::GetStreamInfoWithoutCheck(const std::string &tag)
{
    std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
    return &tagStreamInfo_[tag];
}

HcclResult hcclImpl::SetPipelineSliceNum(u64 piplineSliceNum)
{
    piplineSliceNum_ = piplineSliceNum;
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
    const HcomCollOpInfo &opInfo)
{
    return workSpaceRes_->CreateOpBasedResources(opType, tag, opInfo);
}

u32 hcclImpl::GetInnerCommRank(const u32 ringIdx)
{
    return commFactory_->GetInnerCommRank(ringIdx);
}

HcclResult hcclImpl::GetAlltoAllStatus(DeviceMem &tinySendRecvMem, bool &isAlltoAllZCopyMode)
{
    tinySendRecvMem = tinySendRecvMem_;
    isAlltoAllZCopyMode = isAlltoAllZCopyMode_;

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::UpdateAlltoAllStatus(bool &isAlltoAllZCopyMode, bool &needRecreateAlltoallComm,
    std::map<std::string, bool> &isAlltoAllZCopyModeMap)
{
    isAlltoAllZCopyMode_ = isAlltoAllZCopyMode;
    needRecreateAlltoallComm_ = needRecreateAlltoallComm;
    isAlltoAllZCopyModeMap_ = isAlltoAllZCopyModeMap;
    return HCCL_SUCCESS;
}

u64 hcclImpl::GetOtherRankAllocScratchSize(
    u32 rank,
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 workspaceMemSize = 0;
    u32 meshAggregationIndex = rank / meshAggregationRankSize_;
    u32 meshAggregationRankBegin = meshAggregationIndex * meshAggregationRankSize_;
    for (u32 infoIndex = rank % meshAggregationRankSize_; infoIndex < userRankSize_;
        infoIndex += meshAggregationRankSize_) {
        for (u32 k = meshAggregationRankBegin; k < meshAggregationRankBegin + meshAggregationRankSize_; k++) {
            workspaceMemSize += allMeshAggregationSendRecvInfo[k].sendLength[infoIndex];
        }
    }
    HCCL_DEBUG("[hcclImpl][GetOtherRankScratchSize] rank [%u] workspaceMemSize[%llu]", rank,
        workspaceMemSize);
    if (workspaceMemSize == 0) {
        workspaceMemSize = inOutPutTempMem_[deviceLogicId_].size();
    } else {
        workspaceMemSize = std::max(workspaceMemSize, cclBufferManager_.GetInCCLbufferSize());
    }
    return workspaceMemSize;
}

void hcclImpl::CheckStagedAlltoAllNeedRecreateComm(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    const std::string &tag)
{
    if (allMeshAggregationSendRecvInfo.size() % meshAggregationRankSize_ != 0 ||
        allMeshAggregationSendRecvInfo.size() == 0) {
        HCCL_ERROR("Invalid Send Recv Info Size[%u]", allMeshAggregationSendRecvInfo.size());
        return;
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        bool anyRankReallocScratchMem = false;
        if (allRankAlltoallScratchMemSize_.find(tag) == allRankAlltoallScratchMemSize_.end()) {
            allRankAlltoallScratchMemSize_[tag] = std::unordered_map<u32, u64>{};
        }
        std::unordered_map<u32, u64>& currScratchMap = allRankAlltoallScratchMemSize_[tag];
        for (u32 rank = 0; rank < userRankSize_; rank++) {
            if (currScratchMap.find(rank) == currScratchMap.end()) {
                currScratchMap[rank] = 0;
            }
            u64 scratchSizeCurrRank = GetOtherRankAllocScratchSize(rank, allMeshAggregationSendRecvInfo);
            u64 lastScratchSize = currScratchMap[rank];
            bool thisRankReallocScratch = scratchSizeCurrRank > lastScratchSize;
            anyRankReallocScratchMem |= thisRankReallocScratch;
            currScratchMap[rank] = std::max(scratchSizeCurrRank, lastScratchSize);
            HCCL_DEBUG("[HcclImpl][CheckStagedAlltoAllNeedRecreateComm] local rank %u other rank %u"
                "last scratch %llu, curr scratch %llu tag %s", userRank_, rank, lastScratchSize,
                scratchSizeCurrRank, tag.c_str());
        }
        needRecreateAlltoallComm_ |= (isAlltoAllZCopyMode_ && anyRankReallocScratchMem);
        HCCL_DEBUG("[HcclImpl][CheckStagedAlltoAllNeedRecreateComm] %s recreate alltoall comm tag %s",
            needRecreateAlltoallComm_ ? "need" : "not need", tag.c_str());
    }
}

HcclResult hcclImpl::PrepareInnerCommInfo(u32 &segmentIdx, u32 &commIndex, u64 &hdSize,
                                          const SubCommInfo &commInfo,
                                          const std::vector<std::vector<Slice> > &multRingsSliceZero,
                                          const std::string &tag)
{
    segmentIdx = devicePhyId_;
    commIndex = devicePhyId_;
    CHK_PRT_RET(multRingsSliceZero.empty(), HCCL_ERROR("[Prepare][InnerCommInfo]sicle map is empty"), HCCL_E_PARA);
    if (multRingsSliceZero.size() > 1) {
        std::vector<u32>::iterator iterNic = std::find(nicList_.begin(), nicList_.end(), devicePhyId_);
        if (iterNic != nicList_.end()) {                          // 如果当前rank为通信网口
            u32 nicIdx = distance(nicList_.begin(), iterNic);
            std::unique_lock<std::mutex> lock(nicSendSizeListLock_);
            auto iter = nicSendSizeList_.find(tag);
            CHK_PRT_RET(iter == nicSendSizeList_.end(), HCCL_ERROR("[Prepare][InnerCommInfo]find tag[%s] in "\
                "nicSendSizeList_ failed", tag.c_str()), HCCL_E_INTERNAL);
            CHK_PRT_RET(nicIdx >= iter->second.size(), HCCL_ERROR("[Prepare][InnerCommInfo]tag[%s] nicIdx[%u] "\
                "invaild, expect less than %zu", tag.c_str(), nicIdx, iter->second.size()), HCCL_E_INTERNAL);
            hdSize = iter->second[nicIdx];                    // 通过nicSendSizeList_得到该网口传输数据量
            u32 ringRanks = multRingsSliceZero[0].size(); // 获取单个 ring 上设备的数量
            segmentIdx = ringRanks / nicList_.size() * nicIdx; // 通过网口位置得到该网口传输数据的起始位置
            // 910A只有8卡场景，所以commIdx等于devicePhyId_，不由nicIdx决定
            // 910_73场景的ring环内的设备物理ID是是由nicList决定的，需要segmentIdx(由nicIdx决定)更新
            if (deviceType_ == DevType::DEV_TYPE_910_73) {
                commIndex = segmentIdx;
            }
        } else {                                                  // 如果当前rank不是通信网口，则不发送数据
            hdSize = 0;
        }
    } else if (multRingsSliceZero.size() == 1) {
        segmentIdx = commInfo.localRank; // 针对0、4device下
        CHK_PRT_RET(segmentIdx >= multRingsSliceZero[0].size(), HCCL_ERROR("[Prepare][InnerCommInfo]index is out of "\
            "range. Idx[%u] Slice size[%llu]", segmentIdx, multRingsSliceZero[0].size()), HCCL_E_PARA);
        hdSize = multRingsSliceZero[0][segmentIdx].size;
        commIndex = segmentIdx;
    } else {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::PrepareInnerCommInfo(u32 &segmentIdx, u32 &commIndex, u64 &hdSize,
                                          std::vector<std::unique_ptr<CommBase> > &commOuter,
                                          const std::vector<std::vector<Slice> > &multRingsSliceZero,
                                          const std::string &tag)
{
    segmentIdx = devicePhyId_;
    commIndex = devicePhyId_;
    CHK_PRT_RET(multRingsSliceZero.empty(), HCCL_ERROR("[Prepare][InnerCommInfo]sicle map is empty"), HCCL_E_PARA);
    if (multRingsSliceZero.size() > 1) {
        std::vector<u32>::iterator iterNic = std::find(nicList_.begin(), nicList_.end(), devicePhyId_);
        if (iterNic != nicList_.end()) {                          // 如果当前rank为通信网口
            u32 nicIdx = distance(nicList_.begin(), iterNic);
            std::unique_lock<std::mutex> lock(nicSendSizeListLock_);
            auto iter = nicSendSizeList_.find(tag);
            CHK_PRT_RET(iter == nicSendSizeList_.end(), HCCL_ERROR("[Prepare][InnerCommInfo]find tag[%s] in "\
                "nicSendSizeList_ failed", tag.c_str()), HCCL_E_INTERNAL);
            CHK_PRT_RET(nicIdx >= iter->second.size(), HCCL_ERROR("[Prepare][InnerCommInfo]tag[%s] nicIdx[%u] "\
                "invaild, expect less than %zu", tag.c_str(), nicIdx, iter->second.size()), HCCL_E_INTERNAL);
            hdSize = iter->second[nicIdx];                    // 通过nicSendSizeList_得到该网口传输数据量
            u32 ringRanks = multRingsSliceZero[0].size(); // 获取单个 ring 上设备的数量
            segmentIdx = ringRanks / nicList_.size() * nicIdx; // 通过网口位置得到该网口传输数据的起始位置
            // 910A只有8卡场景，所以commIdx等于devicePhyId_，不由nicIdx决定
            // 910_73场景的ring环内的设备物理ID是是由nicList决定的，需要segmentIdx(由nicIdx决定)更新
            if (deviceType_ == DevType::DEV_TYPE_910_73) {
                commIndex = segmentIdx;
            }
        } else {                                                  // 如果当前rank不是通信网口，则不发送数据
            hdSize = 0;
        }
    } else if (multRingsSliceZero.size() == 1) {
        CHK_PRT_RET(commOuter.empty(), HCCL_ERROR("[Prepare][InnerCommInfo]comm outer is empty"), HCCL_E_PARA);
        segmentIdx = commOuter[0]->Rank(); // 针对0、4device下
        CHK_PRT_RET(segmentIdx >= multRingsSliceZero[0].size(), HCCL_ERROR("[Prepare][InnerCommInfo]index is out of "\
            "range. Idx[%u] Slice size[%llu]", segmentIdx, multRingsSliceZero[0].size()), HCCL_E_PARA);
        hdSize = multRingsSliceZero[0][segmentIdx].size;
        commIndex = segmentIdx;
    } else {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetRingNics(const std::string &tag, const std::vector<std::vector<u32>> &ringNics)
{
    std::unique_lock<std::mutex> lock(ringNicListLock_);
    ringNicList_[tag] = ringNics;
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetRingNics(const std::string &tag, std::vector<std::vector<u32>> &ringNics)
{
    std::unique_lock<std::mutex> lock(ringNicListLock_);
    auto iterRingNic = ringNicList_.find(tag);
    if (iterRingNic == ringNicList_.end()) {
        ringNics = {{0, 1, 2, 3, 4, 5, 6, 7}};
    } else {
        ringNics = iterRingNic->second;
    }
    return HCCL_SUCCESS;
}

void hcclImpl::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
    std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    rankDevicePhyIdNicInfoMap_ = rankDevicePhyIdNicInfoMap;
    ranksPort_ = ranksPort;
    isSetHDCModeInfo_ = isSetHDCModeInfo;
    isUseRankPort_ = isUseRankPort;
}

u64 hcclImpl::GetInCCLbufferSize() const
{
    return cclBufferManager_.GetInCCLbufferSize();
}

HcclResult hcclImpl::GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>> &CommPlaneRanks)
{
    CHK_RET(commFactory_->GetCommPlaneRanks(CommPlaneRanks));
    return HCCL_SUCCESS;
}
HcclResult hcclImpl::GetIsBridgeVector(std::vector<bool> &isBridgeVector)
{
    CHK_RET(commFactory_->GetIsBridgeVector(isBridgeVector));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetIsUsedRdmaMap(std::unordered_map<u32, bool> &isUsedRdmaMap)
{
    CHK_RET(commFactory_->GetIsUsedRdmaMap(isUsedRdmaMap));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetDispatcher(HcclDispatcher &dispatcher)
{
    dispatcher = dispatcher_;
    return HCCL_SUCCESS;
}
HcclResult hcclImpl::GetVirtualDispatcher(HcclDispatcher &vdispatcher)
{
    vdispatcher = vDispatcher_;
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetParallelTaskLoader(ParallelTaskLoader* &parallelTaskLoader)
{
    parallelTaskLoader = parallelTaskLoader_.get();
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetRankVecInfo(std::vector<std::vector<std::vector<u32>>> &serverAndsuperPodToRank)
{
    CHK_RET(commFactory_->GetRankVecInfo(serverAndsuperPodToRank));
    return HCCL_SUCCESS;
}

}
// namespace hccl
