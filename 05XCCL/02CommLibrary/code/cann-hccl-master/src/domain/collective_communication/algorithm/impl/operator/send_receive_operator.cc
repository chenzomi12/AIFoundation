/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_receive_operator.h"
#include "rank_consistent.h"
#include "executor_impl.h"

#define BATCH_SEND_RECV_TAG "_targetRanksHash_"

namespace hccl {
SendReceiveOperator::SendReceiveOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(pImpl, topoMatcher, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV)
{
}

SendReceiveOperator::~SendReceiveOperator()
{
}

HcclResult SendReceiveOperator::SendRun(const std::string &tag, DeviceMem &inputPtr, u64 count, HcclDataType dataType,
    u32 destUserRank, Stream stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    if (currComm->commP2P.size() <= 0) {
        HCCL_ERROR("[SendReceiveOperator][SendRun]errNo[0x%016llx] tag[%s],send op p2p comm is empty",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), tag.c_str());
        return HCCL_E_INTERNAL;
    }
    std::unique_ptr<CommBase> &commP2P = currComm->commP2P[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commP2P);

    // 获取到目的userrank的link
    u32 destRank = 0;
    HcclResult ret = commP2P->GetRankByUserRank(destUserRank, destRank);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[SendReceiveOperator][SendRun]destUserRank[%u],get destRank[%u]", destUserRank, destRank), ret);

    std::shared_ptr<Transport> &destLink = commP2P->GetTransportByRank(destRank);
    CHK_SMART_PTR_NULL(destLink);

    SendReceive executor(dispatcher_, destLink);
    CHK_RET(executor.SendPrepare(inputPtr, destRank, stream));
    CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(executor.SendRunAsync());
    return HCCL_SUCCESS;
}

HcclResult SendReceiveOperator::SendCommon(const std::string &tag, DeviceMem &inputPtr, u64 count,
    HcclDataType dataType, u32 destUserRank, Stream stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    if (currComm->commP2P[0] == nullptr) {
        HCCL_ERROR("[hcclImpl][SendCommon]errNo[0x%016llx] tag[%s],send op p2p comm is NULL",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), tag.c_str());
        return HCCL_E_INTERNAL;
    }

    std::unique_ptr<CommBase> &commP2P = currComm->commP2P[0];
    CHK_SMART_PTR_NULL(commP2P);

    // 获取到目的userrank的link
    u32 destRank = 0;
    HcclResult ret = commP2P->GetRankByUserRank(destUserRank, destRank);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[hcclImpl][SendCommon]send op destUserRank[%u],get destRank[%u]",
            destUserRank, destRank), ret);

    std::shared_ptr<Transport> &destLink = commP2P->GetTransportByRank(destRank);
    CHK_SMART_PTR_NULL(destLink);

    // 使用SendReceive类，执行send操作
    std::unique_ptr<SendReceive> executor;
    executor.reset(new (std::nothrow) SendReceive(dispatcher_, destLink));
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->SendPrepare(inputPtr, destRank, stream));

    CHK_RET(executor->RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                                       stream));

    CHK_RET(executor->SendRunAsync()); /* 执行send操作 */

    return HCCL_SUCCESS;
}

HcclResult SendReceiveOperator::Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, Stream stream)
{
    /* ------------集合通信资源准备------------ */
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 length = static_cast<u64>(count) * perDataSize;
    DeviceMem inputMem(inputPtr, length);

    CHK_RET(hcclImpl_->PrepareCommRes(tag, inputMem, inputMem, AlgType::ALG_RESERVED, stream, destRank, true, false));

    /*  ------------执行算法-------------- */
    if (Is310P3Common()) {
        CHK_RET(SendCommon(tag, inputMem, count, dataType, destRank, stream));
    } else {
        CHK_RET(SendRun(tag, inputMem, count, dataType, destRank, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult SendReceiveOperator::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, Stream stream)
{
    void *commInputPtr = nullptr;
    u64 commInputSize;
    HcclResult ret;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));

    u32 unitSize = SIZE_TABLE[dataType];
    auto meta = HcclOpMetaInfo::GetOneForSend();
    u8 *curInputPtr = static_cast<u8 *>(inputPtr);
    CHK_PTR_NULL(curInputPtr);

    u64 inputOffset = 0;
    u64 countLeft = count;
    while (countLeft > 0) {
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        curInputPtr += inputOffset;
        HCCL_DEBUG("SendOutPlace:inputOffset[%llu]", inputOffset);
        u64 curCount = ((countLeft * unitSize) > commInputSize) ? (commInputSize / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte

        HCCL_DEBUG("SendOutPlace:curInputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr, curCount, curSize);
        DeviceMem inCommMem(commInputPtr, curSize);
        DeviceMem inMem(curInputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, stream));

        /* 记录指令信息用于一致性校验 */
        ret = RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_SEND, tag, curCount, dataType,
            commInputSize, 0, HCCL_WORLD_GROUP);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] record CMD with parameter error", HCCL_ERROR_CODE(ret)), ret);
        /* 入参的正确性由HCCL确保 */
        ret = Send(tag, commInputPtr, curCount, dataType, destRank, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] SendOutPlace: send error, tag[%s], ptr[%p], count[%llu], dataType[%s]",
            HCCL_ERROR_CODE(ret), tag.c_str(), commInputPtr, curCount, GetDataTypeEnumStr(dataType).c_str()),
            ret);
        ret = RankConsistent::GetInstance().DelOpPara(tag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] delete CMD with parameters error. tag[%s]", HCCL_ERROR_CODE(ret), tag.c_str()),
            ret);
        CHK_PRT_RET((curCount == 0), HCCL_ERROR("In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        inputOffset = curSize;

        CHK_RET(LaunchTask(dispatcher_, stream));
    }
    return HCCL_SUCCESS;
}

HcclResult SendReceiveOperator::ReceiveRun(const std::string &tag, DeviceMem &outputPtr, u64 count,
    HcclDataType dataType, u32 srcUserRank, Stream &stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    if (currComm->commP2P.size() <= 0) {
        HCCL_ERROR("[SendReceiveOperator][ReceiveRun]errNo[0x%016llx] tag[%s],receive op p2p comm is empty", \
                   HCCL_ERROR_CODE(HCCL_E_INTERNAL), tag.c_str());
        return HCCL_E_INTERNAL;
    }

    std::unique_ptr<CommBase> &commP2P = currComm->commP2P[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commP2P);

    // 通过user_rank获取rank
    u32 srcRank = 0;
    HcclResult ret = commP2P->GetRankByUserRank(srcUserRank, srcRank);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[SendReceiveOperator][ReceiveRun]receive op srcuserrank[%u],get srcrank[%u]",
            srcUserRank, srcRank), HCCL_E_INTERNAL);

    std::shared_ptr<Transport> &srcLink = commP2P->GetTransportByRank(srcRank);
    CHK_SMART_PTR_NULL(srcLink);

    SendReceive executor(dispatcher_, srcLink);
    CHK_RET(executor.ReceivePrepare(outputPtr, srcRank, stream));
    CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(executor.ReceiveRunAsync());

    return HCCL_SUCCESS;
}

HcclResult SendReceiveOperator::ReceiveCommon(const std::string &tag, DeviceMem &outputPtr, u64 count,
    HcclDataType dataType, u32 srcUserRank, Stream stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    if (currComm->commP2P[0] == nullptr) {
        HCCL_ERROR("[hcclImpl][ReceiveCommon]errNo[0x%016llx] tag[%s],receive op p2p comm is NULL", \
                   HCCL_ERROR_CODE(HCCL_E_INTERNAL), tag.c_str());
        return HCCL_E_INTERNAL;
    }

    std::unique_ptr<CommBase> &commP2P = currComm->commP2P[0];
    CHK_SMART_PTR_NULL(commP2P);

    // 通过user_rank获取rank
    u32 srcRank = 0;
    HcclResult ret = commP2P->GetRankByUserRank(srcUserRank, srcRank);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[hcclImpl][ReceiveCommon]receive op srcuserrank[%u],get srcrank[%u]",
            srcUserRank, srcRank), HCCL_E_INTERNAL);

    std::shared_ptr<Transport> &srcLink = commP2P->GetTransportByRank(srcRank);
    CHK_SMART_PTR_NULL(srcLink);

    // 获取ring algorithm所需的通信连接
    std::unique_ptr<SendReceive> executor;
    executor.reset(new (std::nothrow) SendReceive(dispatcher_, srcLink));
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->ReceivePrepare(outputPtr, srcRank, stream));

    CHK_RET(executor->RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                                       stream));

    CHK_RET(executor->ReceiveRunAsync()); /* 执行send操作 */

    return HCCL_SUCCESS;
}

HcclResult SendReceiveOperator::Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, Stream stream)
{
    /* ------------集合通信资源准备------------ */

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 length = static_cast<u64>(count) * perDataSize;
    DeviceMem outputMem(outputPtr, length);

    /*  ------------执行算法-------------- */
    CHK_RET(hcclImpl_->PrepareCommRes(tag, outputMem, outputMem, AlgType::ALG_RESERVED, stream, srcRank, true, false));

    if (Is310P3Common()) {
        CHK_RET(ReceiveCommon(tag, outputMem, count, dataType, srcRank, stream));
    } else {
        CHK_RET(ReceiveRun(tag, outputMem, count, dataType, srcRank, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult SendReceiveOperator::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count,
    HcclDataType dataType, u32 srcRank, Stream stream)
{
    void *commOutputPtr = nullptr;
    u64 commOutputSize;
    HcclResult ret;

    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    u32 unitSize = SIZE_TABLE[dataType];

    auto meta = HcclOpMetaInfo::GetOneForRecieve();
    u8 *curOutputPtr = static_cast<u8 *>(outputPtr);
    u64 outputOffset = 0;
    u64 countLeft = count;
    while (countLeft > 0) {
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        curOutputPtr += outputOffset;
        HCCL_DEBUG("ReceiveOutPlace:outputOffset[%llu]", outputOffset);
        u64 curCount = ((countLeft * unitSize) > commOutputSize) ? (commOutputSize / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte

        HCCL_DEBUG("ReceiveOutPlace:curOutputPtr[%p], curCount[%llu], curSize[%llu]", curOutputPtr, curCount, curSize);
        /* 记录指令信息用于一致性校验 */
        ret = RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_RECEIVE, tag, curCount, dataType,
            commOutputSize, 0, HCCL_WORLD_GROUP);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] record CMD with parameter error", HCCL_ERROR_CODE(ret)), ret);
        /* 入参的正确性由HCCL确保 */
        ret = Receive(tag.c_str(), commOutputPtr, curCount, dataType, srcRank, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] ReceiveOutPlace: recv error, tag[%s], ptr[%p], count[%llu], dataType[%s]",
            HCCL_ERROR_CODE(ret), tag.c_str(), commOutputPtr, curCount, GetDataTypeEnumStr(dataType).c_str()),
            ret);
        ret = RankConsistent::GetInstance().DelOpPara(tag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] delete CMD with parameters error. tag[%s]", HCCL_ERROR_CODE(ret), tag.c_str()),
            ret);

        DeviceMem outCommMem(commOutputPtr, curSize);
        DeviceMem outMem(curOutputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, stream));
        CHK_PRT_RET((curCount == 0), HCCL_ERROR("In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        outputOffset = curSize;

        CHK_RET(LaunchTask(dispatcher_, stream));
    }
    return HCCL_SUCCESS;
}
}