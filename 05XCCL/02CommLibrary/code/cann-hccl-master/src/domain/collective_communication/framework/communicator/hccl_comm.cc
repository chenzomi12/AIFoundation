/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <algorithm>
#include <arpa/inet.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <hccl/hccl_types.h>
#include "device_capacity.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"

namespace hccl {
RankTable_t g_hcclDefaultRankTable;

hcclComm::hcclComm(u64 inCCLbufferSize, u64 outCCLbufferSize, std::string identifier)
    : barrierSendBuf(nullptr), barrierRecvBuf(nullptr),
      inCCLbufferSize_(inCCLbufferSize), outCCLbufferSize_(outCCLbufferSize),
      deviceType_(DevType::DEV_TYPE_COUNT), isFirstBarrier_(true), identifier_(identifier), isHeterogComm_(false),
      isResetDevice_(false), isSpecialType_(false), communicator_(nullptr)
{
    indirectInCCLbuffer_ = DeviceMem();
    indirectOutCCLbuffer_ = DeviceMem();
    barrierInMemory_ = DeviceMem();
    barrierOutMemory_ = DeviceMem();
}

hcclComm::~hcclComm()
{
    RealeaseBarrierMemory();
    communicator_ = nullptr;
}

HcclResult hcclComm::ReleaseSubComms() const
{
    CHK_SMART_PTR_NULL(communicator_);

    CHK_RET(communicator_->ReleaseCommInfos());

    return HCCL_SUCCESS;
}

void hcclComm::ReleaseCommCCLbuffer() const
{
    if (!communicator_) {
        return;
    }
    communicator_->ReleaseCommCCLbuffer();
}

void hcclComm::RealeaseBarrierMemory()
{
    barrierInMemory_.free();
    barrierOutMemory_.free();
}

void hcclComm::UpdateIsHaveCpuRank(const RankTable_t &rankTable)
{
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (rankTable.rankList[i].deviceInfo.devicePhyId == HOST_DEVICE_ID) {
            isHaveCpuRank_ = true;
        }
    }
}

void hcclComm::UpdateIsHaveCpuRank(const std::vector<RankInfo> &rankList)
{
    for (u32 i = 0; i < rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (rankList[i].devicePhyId == HOST_DEVICE_ID) {
            isHaveCpuRank_ = true;
        }
    }
}

HcclResult hcclComm::init(HcclCommParams &params, const RankTable_t &rankTable)
{
    UpdateIsHaveCpuRank(rankTable);
    isHeterogComm_ = params.isHeterogComm;

    HCCL_INFO("hcclComm init workmode [%d]", params.commWorkMode);
    if (params.commWorkMode == WorkMode::HCCL_MODE_AI_CPU) {
        isSpecialType_ = true;
    }
    CHK_RET(InitImpl(params.deviceType));

    /* 强行将最后一个字符置0, 确保其可以做字符串操作 */
    params.id.internal[HCCL_ROOT_INFO_BYTES - 1] = '\0';

    /* 入参判断 */
    if (params.rank >= params.totalRanks) {
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] rank[%u] out of range[0, %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            params.rank, params.totalRanks - 1);
        return HCCL_E_PARA;
    }
    params.identifier = identifier_;

    CHK_RET(communicator_->AtomicInitSet());                  /* 初始化竞争, 只允许被初始化一次 */
    HcclResult ret = communicator_->Init(params, rankTable);  /* 初始化实例, 失败则重新开放初始化竞争 */
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] hccl initialize failed", HCCL_ERROR_CODE(ret));
        communicator_->AtomicInitClear();
        return ret;
    }

    CHK_RET(communicator_->InitCCLbuffer(inCCLbufferSize_, outCCLbufferSize_));

    HCCL_RUN_INFO("hcclCommInitInfo:commId[%s], rank[%u], totalRanks[%u], serverId[%s], deviceType[%d]," \
        "logicDevId[%d], identifier[%s]", params.id.internal, params.rank, params.totalRanks, params.serverId.c_str(),
        params.deviceType, params.logicDevId, params.identifier.c_str());
    return HCCL_SUCCESS;
}

HcclResult hcclComm::init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
                          WorldGroupInfo &groupCommonData)
{
    UpdateIsHaveCpuRank(rankList);
    /* 强行将最后一个字符置0, 确保其可以做字符串操作 */
    params.id.internal[HCCL_ROOT_INFO_BYTES - 1] = '\0';

    HCCL_INFO("rootInfo[%s], rank[%u], totalRanks[%u], chip[%d], logicDevId[%d]", params.id.internal,
        params.rank, params.totalRanks, params.deviceType, params.logicDevId);

    HCCL_INFO("rootInfo init group workmode[%d]", params.commWorkMode);
    if (params.commWorkMode == WorkMode::HCCL_MODE_AI_CPU) {
        isSpecialType_ = true;
    }
    CHK_RET(InitImpl(params.deviceType));

    /* 入参判断 */
    if (params.rank >= params.totalRanks) {
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] rank[%u] out of range[0, %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            params.rank, params.totalRanks - 1);
        return HCCL_E_PARA;
    }

    params.identifier = identifier_;

    CHK_RET(communicator_->CheckDeviceType(params.deviceType));                /* 芯片类型检查 */
    CHK_RET(communicator_->AtomicInitSet());                                 /* 初始化竞争, 只允许被初始化一次 */
    HcclResult ret = communicator_->Init(params, rankList, groupCommonData); /* 初始化实例, 失败则重新开放初始化竞争 */
    if (ret != HCCL_SUCCESS) {
        communicator_->AtomicInitClear();
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] hccl initialize failed", HCCL_ERROR_CODE(ret));
        return ret;
    }
    return ret;
}

HcclResult hcclComm::CreateGroup(const std::string &group, const u32 &groupRank, const u32 &userRank,
                                 const std::vector<u32> &groupRanks, std::shared_ptr<hcclComm> &groupComm)
{
    // 增加输出日志关键字
    HCCL_INFO("HCCL_KEY_INFO: group[%s], groupRank[%u], userRank[%u], groupComm[%p]", group.c_str(),
        groupRank, userRank, groupComm.get());

    // 入参有消息校验
    if (group.length() == 0) {
        HCCL_ERROR("[Create][Group]errNo[0x%016llx] group name lenth is 0", HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if (groupRank >= groupRanks.size()) {
        HCCL_ERROR("[Create][Group]errNo[0x%016llx] group rank[%u] out of range [0,%llu])",
            HCCL_ERROR_CODE(HCCL_E_PARA), groupRank, groupRanks.size() - 1);
        return HCCL_E_PARA;
    }

    HcclRootInfo id;
    CHK_RET(GetUniqueId(&id));

    HcclCommParams params;
    params.rank = groupRank;
    params.userRank = userRank;
    params.totalRanks = groupRanks.size();
    params.isHeterogComm = isHeterogComm_;
    s32 iret = snprintf_s(params.id.internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1, "%s%s%s",
                          id.internal, "-", group.c_str());

    CHK_PRT_RET((iret == -1), HCCL_ERROR("[Create][Group]errNo[0x%016llx] get group unique id falied",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);

    WorldGroupInfo groupCommonData;

    CHK_RET(communicator_->GetGroupCommonData(groupCommonData));
    params.logicDevId = groupCommonData.deviceLogicId;
    params.profilingInitiated = groupCommonData.profilingInitiated;
    params.deviceType = deviceType_;
    params.hcomGroupNicInit = communicator_->GetNicInitialized();
    std::vector<RankInfo> rankList;

    CHK_RET(communicator_->GetGroupRanksInfo(groupRanks, rankList));

    groupComm.reset(new (std::nothrow) hccl::hcclComm(0, 0, group));
    CHK_SMART_PTR_NULL(groupComm);
    CHK_RET(groupComm->init(params, rankList, groupCommonData));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::DestroyGroup(const std::string &group) const
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("start destroy group[%s]", group.c_str());
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetAlgType(AlgType &algType, HcclCMDType opType)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("algType[%s]", HcclAlg::AlgTypeToStr(algType).c_str());
    return communicator_->GetAlgType(algType, opType);
}

HcclResult hcclComm::AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
                               HcclDataType dataType, HcclRtStream stream, HcomCollOpInfo *opInfo)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], count[%llu], data_type[%s]", tag.c_str(), inputPtr,
        GetDataTypeEnumStr(dataType).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AllGather]errNo[0x%016llx] all gather tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(inputCount));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->AllGather(tag, inputPtr, outputPtr, inputCount, dataType, stream, opInfo));

    return HCCL_SUCCESS;
}
HcclResult hcclComm::AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
        tag.c_str(), inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str());

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->AllGatherOutPlace(tag, inputPtr, outputPtr, inputCount, dataType, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, SyncMode syncMode)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
               tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
               GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AllReduce]errNo[0x%016llx] all reduce tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckReductionOp(op));
    CHK_RET(communicator_->AllReduce(tag, inputPtr, outputPtr, count, dataType, op, stream, syncMode));

    return HCCL_SUCCESS;
}


HcclResult hcclComm::AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, SyncMode syncMode)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]", tag.c_str(),
        inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->AllReduceOutPlace(tag, inputPtr, outputPtr, count, dataType, op, stream, syncMode));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                               const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                               rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(rdispls);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AlltoAllV]errNo[0x%016llx] alltoallv tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckDataType(sendType, false));

    CHK_RET(communicator_->CheckDataType(recvType, false));

    CHK_RET(communicator_->AlltoAllV(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType,
        stream, tag));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(sendType, false));
    CHK_RET(communicator_->CheckDataType(recvType, false));

    CHK_RET(communicator_->AlltoAllVOutPlace(
        sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, stream, tag));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendCountMatrix);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AlltoAllVC]errNo[0x%016llx] alltoallvc tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckDataType(sendType, false));

    CHK_RET(communicator_->CheckDataType(recvType, false));

    CHK_RET(communicator_->AlltoAllVC(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(sendType, false));
    CHK_RET(communicator_->CheckDataType(recvType, false));

    CHK_RET(communicator_->AlltoAllVCOutPlace(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType, const void *recvBuf,
    u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_PTR_NULL(communicator_);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AlltoAll]errNo[0x%016llx] alltoall tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckDataType(sendType, false));

    CHK_RET(communicator_->CheckDataType(recvType, false));

    CHK_RET(communicator_->AlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, stream, tag));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
                               HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO:tag[%s], ptr[%p], count[%llu], data_type[%s], root[%u]",
               tag.c_str(), ptr, count, GetDataTypeEnumStr(dataType).c_str(), root);

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(ptr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][Broadcast]errNo[0x%016llx] broadcast tag length is 0",
            HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    /* * 初始化检查 */
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(root));
    CHK_RET(communicator_->Broadcast(tag, ptr, count, dataType, root, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
    HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO:tag[%s], ptr[%p], count[%llu], data_type[%s], root[%u]", tag.c_str(), ptr, count,
        GetDataTypeEnumStr(dataType).c_str(), root);

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(root));
    CHK_RET(communicator_->BroadcastOutPlace(tag, ptr, count, dataType, root, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], recvCount[%llu], data_type[%s], root[%u]",
        tag.c_str(), inputPtr, outputPtr, recvCount, GetDataTypeEnumStr(dataType).c_str(), root);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][Scatter]errNo[0x%016llx] scatter tag length is 0",
            HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    CHK_RET(communicator_->CheckCount(recvCount));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(root));
    CHK_RET(communicator_->ScatterOutPlace(tag, inputPtr, outputPtr, recvCount, dataType, root, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                   HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], "
               "op[%s]", tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
               GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][ReduceScatter]errNo[0x%016llx] reduceScatter tag length is"\
            "0", HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckReductionOp(op));
    CHK_RET(communicator_->ReduceScatter(tag, inputPtr, outputPtr, count, dataType, op, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
        tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->ReduceScatterOutPlace(tag, inputPtr, outputPtr, count, dataType, op, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                            HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",\
               tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
               GetReduceOpEnumStr(op).c_str(), root);

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][Reduce]errNo[0x%016llx] reduce tag length is 0",
            HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckReductionOp(op));
    CHK_RET(communicator_->CheckUserRank(root));
    CHK_RET(communicator_->Reduce(tag, inputPtr, outputPtr, count, dataType, op, root, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",
        tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        root);

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckUserRank(root));
    CHK_RET(communicator_->ReduceOutPlace(tag, inputPtr, outputPtr, count, dataType, op, root, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ProcessSendRecvTasks(const std::string &tag, std::vector<struct HcclSendRecvItemDef*> &orderedList,
    u32 itemNum, u32 startIndex, rtStream_t stream)
{
    CHK_RET(communicator_->ProcessSendRecvTasks(tag, orderedList, itemNum, startIndex, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
                          u32 destRank, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], count[%llu], data_type[%s], destRank[%u]",
        tag.c_str(), inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank);

    /* 入参检查 */
    CHK_PTR_NULL(inputPtr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][Send]errNo[0x%016llx] send tag length is 0",
            HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(destRank));
    CHK_RET(communicator_->Send(tag, inputPtr, count, dataType, destRank, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], count[%llu], data_type[%s], destRank[%u],",
        tag.c_str(), inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank);

    /* 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(destRank));
    CHK_RET(communicator_->SendOutPlace(tag, inputPtr, count, dataType, destRank, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], output_ptr[%p], count[%llu], data_type[%s], srcRank[%u]",
        tag.c_str(), outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank);

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(srcRank));
    CHK_RET(communicator_->ReceiveOutPlace(tag, outputPtr, count, dataType, srcRank, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
                             u32 srcRank, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], output_ptr[%p], count[%llu], data_type[%s], srcRank[%u]",
               tag.c_str(), outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank);

    /* * 入参检查 */
    CHK_PTR_NULL(outputPtr);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][Receive]errNo[0x%016llx] receive tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(srcRank));
    CHK_RET(communicator_->Receive(tag, outputPtr, count, dataType, srcRank, stream));

    return HCCL_SUCCESS;
}

// 目前支持按tag对资源释放、解绑定
HcclResult hcclComm::ClearOpResource(const std::string &tag)
{
    CHK_RET(communicator_->ClearOpResource(tag));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetUniqueId(HcclRootInfo *uniqueId)
{
    CHK_PTR_NULL(uniqueId);

    std::string uniqueIdGot = HcclCommunicator::GetUniqueId();
    s32 ret = snprintf_s(uniqueId->internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1,
                         "%s%s", "hccl-", uniqueIdGot.c_str());
    CHK_PRT_RET((ret == -1), HCCL_ERROR("[Get][UniqueId]errNo[0x%016llx] get unique id failed,uniqueId[%p]",
        HCCL_ERROR_CODE(ret), uniqueId), HCCL_E_MEMORY);

    return HCCL_SUCCESS;
}

HcclResult hcclComm::CreateCommCCLbuffer() const
{
    CHK_RET(communicator_->CreateCommCCLbuffer());

    return HCCL_SUCCESS;
}

HcclResult hcclComm::CreateIndirectCCLbuf()
{
    indirectInCCLbuffer_ = DeviceMem::alloc(sizeof(uintptr_t), true);
    CHK_SMART_PTR_NULL(indirectInCCLbuffer_);
    indirectOutCCLbuffer_ = DeviceMem::alloc(sizeof(uintptr_t), true);
    CHK_SMART_PTR_NULL(indirectOutCCLbuffer_);

    return HCCL_SUCCESS;
}

void hcclComm::ReleaseIndirectCCLbuf()
{
    indirectInCCLbuffer_.free();
    indirectOutCCLbuffer_.free();
}

HcclResult hcclComm::GetIndirectInCCLbuf(void* &ptr, u64 &size)
{
    ptr = indirectInCCLbuffer_.ptr();
    size = sizeof(uintptr_t);
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetIndirectOutCCLbuf(void* &ptr, u64 &size)
{
    ptr = indirectOutCCLbuffer_.ptr();
    size = sizeof(uintptr_t);
    return HCCL_SUCCESS;
}
std::string hcclComm::GetIdentifier()
{
    return identifier_;
}

HcclResult hcclComm::CommCheckErrorCqe(HcclResult &result)
{
    CHK_RET(communicator_->GetCqeError(result));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitImpl(DevType deviceType)
{
    HCCL_INFO("InitImpl Implementation isHeterogComm_[%d] isHaveCpuRank_[%d] deviceType[%d] isSpecialType_[%d]",
        isHeterogComm_,
        isHaveCpuRank_,
        deviceType,
        isSpecialType_);

    communicator_.reset(new (std::nothrow) HcclCommunicator());
    CHK_SMART_PTR_NULL(communicator_);
    deviceType_ = deviceType;

    return HCCL_SUCCESS;
}

HcclResult hcclComm::CreateBarrierMemory()
{
    if (isFirstBarrier_) {
        // 申请device内存
        barrierInMemory_ = DeviceMem::alloc(HCCL_BARRIER_DEFAULT_COUNT * sizeof(float));
        barrierOutMemory_ = DeviceMem::alloc(HCCL_BARRIER_DEFAULT_COUNT * sizeof(float));
        CHK_PRT_RET(!barrierInMemory_, HCCL_ERROR("[Create][BarrierMemory]create barrier input memory fail"),
            HCCL_E_PTR);
        CHK_PRT_RET(!barrierOutMemory_, HCCL_ERROR("[Create][BarrierMemory]create barrier output memory fail"),
            HCCL_E_PTR);

        barrierSendBuf = static_cast<void *>(barrierInMemory_.ptr());
        barrierRecvBuf = static_cast<void *>(barrierOutMemory_.ptr());

        // device内存清0
        // 申请host内存，并将初始值设置为0
        HostMem barrierHostMem = HostMem::alloc(HCCL_BARRIER_DEFAULT_COUNT * sizeof(float));
        CHK_SMART_PTR_NULL(barrierHostMem);
        s32 sRet = memset_s(barrierHostMem.ptr(), barrierHostMem.size(), 0, barrierHostMem.size());
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Create][BarrierMemory]mem set failed.errorno[%d]", sRet), HCCL_E_MEMORY);

        CHK_RET(hrtMemSyncCopy(barrierSendBuf, barrierInMemory_.size(), barrierHostMem.ptr(), barrierHostMem.size(),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        CHK_RET(hrtMemSyncCopy(barrierRecvBuf, barrierOutMemory_.size(), barrierHostMem.ptr(), barrierHostMem.size(),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        isFirstBarrier_ = false;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetInCCLbuffer(void* &buffer, u64 &size)
{
    CHK_RET(communicator_->GetInCCLbuffer(buffer, size));

    return HCCL_SUCCESS;
}
HcclResult hcclComm::GetOutCCLbuffer(void* &buffer, u64 &size)
{
    CHK_RET(communicator_->GetOutCCLbuffer(buffer, size));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetUserRank(u32 &userRank)
{
    userRank = communicator_->GetUserRank();

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetGroupRank(u32 &userRank)
{
    userRank = communicator_->GetGroupRank();

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetRankSize(u32 &rankSize)
{
    rankSize = communicator_->GetRankSize();

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize, HcclCMDType optype) const
{
    return communicator_->GetWorkspaceSubStreamNum(streamNum, dataSize, optype);
}
HcclResult hcclComm::GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
    u32 &rankSize, u64 &size)
{
    return communicator_->GetWorkspaceMemSize(opType, count, dataType, rankSize, size, deviceType_);
}

HcclResult hcclComm::GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize) const
{
    return communicator_->GetAllReduceScratchSize(count, dataType, scratchSize);
}

HcclResult hcclComm::SetQosCfg(const u32 qosCfg)
{
    return communicator_->SetQosCfg(qosCfg);
}

HcclResult hcclComm::ResetQosCfg()
{
    return communicator_->ResetQosCfg();
}

HcclResult hcclComm::GetQosCfg(u32& qosCfg)
{
    return communicator_->GetQosCfg(qosCfg);
}

// 设定 workspace 资源
HcclResult hcclComm::SetWorkspaceResource(const std::string &tag, void *memPtr, u64 maxSize,
                                          std::vector<rtStream_t> &stream)
{
    return communicator_->SetWorkspaceResource(tag, memPtr, maxSize, stream);
}

HcclResult hcclComm::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
    const HcomCollOpInfo &opInfo)
{
    return communicator_->CreateOpBasedResources(opType, tag, opInfo);
}

HcclResult hcclComm::GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation)
{
    return communicator_->GetDeviceNumPerAggregation(deviceNumPerAggregation);
}

HcclResult hcclComm::GetBandWidthPerNPU(u32 level, float &bandWidth)
{
    return communicator_->GetBandWidthPerNPU(level, bandWidth);
}

HcclResult hcclComm::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
    u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize) const
{
    CHK_RET(communicator_->GetAlltoAllStagedWorkSpaceMemSize(
        sendCounts, sdispls, sendType, recvCounts, rdispls, recvType, memSize));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    u64 &memSize) const
{
    CHK_RET(communicator_->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr)
{
    CHK_RET(communicator_->SetGlobalWorkSpace(globalWorkSpaceAddr));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo)
{
    CHK_RET(communicator_->GetandClearOverFlowTasks(hcclDumpInfo));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::SupportDeterministicOptim(bool &isDeterministicOptim)
{
    CHK_RET(communicator_->SupportDeterministicOptim(isDeterministicOptim));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetHccsLinkNum(u32 &numHccsLink)
{
    return communicator_->GetHccsLinkNum(numHccsLink);
}

HcclResult hcclComm::GetDeviceId(s32 &deviceId)
{
    CHK_RET(communicator_->GetDeviceId(deviceId));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetDevType(DevType &devType)
{
    devType = deviceType_;
    return HCCL_SUCCESS;
}

HcclResult hcclComm::IsStandardCard(bool &isStandardCard)
{
        isStandardCard = communicator_->IsStandardCard();

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Is310PDuoCard(bool &is310PDuoCard)
{
    is310PDuoCard = communicator_->Is310PDuoCard();
    return HCCL_SUCCESS;
}

HcclResult hcclComm::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
    HcclDataType dataType, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
               tag.c_str(), inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][Gather]errNo[0x%016llx] gather tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(inputCount));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->Gather(tag, inputPtr, outputPtr, rootRank, inputCount, dataType, stream));

    return HCCL_SUCCESS;
}

bool hcclComm::IsNeedResetDevice()
{
    return isResetDevice_;
}

HcclResult hcclComm::ResetDeviceEnable()
{
    isResetDevice_ = true;
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SaveOpbaseKeyTraceInfo(std::string &logInfo)
{
    CHK_PRT(communicator_->SaveOpbaseKeyTraceInfo(logInfo));

    return HCCL_SUCCESS;
}

bool hcclComm::GetCommResource(const std::string &tag, void **commContext)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: GetCommResource commContext[%p]", commContext);

    return (communicator_->GetCommResource(tag, commContext));
}

HcclResult hcclComm::CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
    void **commContext)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: CreateCommResource commContext[%p], isOpbaseMode[%u]", commContext, isOpbaseMode);

    CHK_RET(communicator_->CreateCommResource(tag, aiCpuStream, isOpbaseMode, commContext));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetAicpuOpStreamNotify(HcclRtStream *opStream, void** aicpuNotify)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: GetAicpuOpStreamNotify commContext[%p]", opStream);

    CHK_RET(communicator_->GetAicpuOpStreamNotify(opStream, aicpuNotify));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: Mc2AiCpuStreamAllocAndGet streamMode[%u]", streamMode);

    CHK_RET(communicator_->Mc2AiCpuStreamAllocAndGet(streamMode, aiCpuStream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize)
{
    HCCL_INFO("HCCL_KEY_INFO: GetTopoDesc topoDescs[%p] topoSize[%u]", topoDescs, topoSize);

    CHK_RET(communicator_->GetTopoDesc(topoDescs, topoSize));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReStartVnic(const HcclCommParams &params, const RankTable_t &rankTable)
{
    CHK_RET(communicator_->ReStartVnic(params, rankTable));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetDeterministicConfig(const u8 deterministic)
{
    CHK_RET(communicator_->SetDeterministicConfig(deterministic));
    return HCCL_SUCCESS;
}

u64 hcclComm::GetConfigInCCLbufferSize()
{
    return inCCLbufferSize_;
}

u64 hcclComm::GetConfigOutCCLbufferSize()
{
    return outCCLbufferSize_;
}
}  // namespace hccl
