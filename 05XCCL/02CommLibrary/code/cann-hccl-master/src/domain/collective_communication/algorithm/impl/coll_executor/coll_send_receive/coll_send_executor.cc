/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_send_executor.h"

namespace hccl {

CollSendExecutor::CollSendExecutor(const HcclDispatcher dispatcher,
                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollNativeExecutorBase(dispatcher, topoMatcher)
{
}

HcclResult CollSendExecutor::Orchestrate(const OpParam& param, const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    GetStreamInfo(algRes);

    HcclResult ret = HCCL_SUCCESS;
    // 图模式场景下不需要Loop
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DeviceMem InputMem = algRes.paramInputMem;
        ret = RunTemplate(param, InputMem);
    } else {
        ret = RunLoop(param, algRes);
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollSendExecutor][Orchestrate]errNo[0x%016llx]send excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("tag[%s] Send Excutor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_INPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollSendExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::CalcP2PCommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport, u32 dstRank)
{
    HCCL_INFO("[CollSendExecutor][CalcOuterCommInfo]tag[%s ]start", tag_.c_str());
    CommParaInfo commP2P(COMM_COMBINE, CommType::COMM_TAG_P2P);
    commP2P.peerUserRank = dstRank;
    CHK_RET(CalcCommPlaneInfo(tag_, commP2P, opTransport[COMM_COMBINE], inputType, outputType));
    HCCL_INFO("[CollSendExecutor][CalcOuterCommInfo]tag[%s] Calc P2PComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport, u32 dstRank)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CalcP2PCommInfo(inputType, outputType, opTransport, dstRank);
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    bool needAivBuffer = false;
    std::vector<LevelNSubCommTransport> opTransport {
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))
    };

    CalcCommInfo(opTransport, param.dstRank);

    BuildResourceRequest(scratchMemSize, streamNum, notifyNum, needAivBuffer, opTransport, resourceRequest);
    HCCL_INFO("streamNum[%u], notifyNum[%u], sctrachMemSize[%llu], needAivBuffer[%u]",
        resourceRequest.streamNum, resourceRequest.notifyNum, resourceRequest.scratchMemSize,
        resourceRequest.needAivBuffer);
    // 打印建链诉求
    PrintTransportRequest(resourceRequest);
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::RunLoop(const OpParam &param, const AlgResourceResponse &algRes)
{
    HcclResult ret;

    u64 commInputSize = algRes.cclInputMem.size();

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    auto meta = HcclOpMetaInfo::GetOneForSend();
    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    CHK_PTR_NULL(curInputPtr);

    u64 inputOffset = 0;
    u64 countLeft = param.DataDes.count;
    while (countLeft > 0) {
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey()));
        curInputPtr += inputOffset;

        HCCL_DEBUG("SendOutPlace:inputOffset[%llu]", inputOffset);
        u64 curCount = ((countLeft * unitSize) > commInputSize) ? (commInputSize / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte

        HCCL_DEBUG("SendOutPlace:curInputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr, curCount, curSize);

        DeviceMem inCommMem(algRes.cclInputMem.ptr(), curSize);
        DeviceMem inMem(curInputPtr, curSize);

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, const_cast<Stream&>(param.stream)));

        /* 记录指令信息用于一致性校验 */
        ret = RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_SEND, param.tag, curCount,
            param.DataDes.dataType, commInputSize, 0, HCCL_WORLD_GROUP);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] record CMD with parameter error", HCCL_ERROR_CODE(ret)), ret);

        ret = RunTemplate(param, inCommMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] SendOutPlace: send error, tag[%s], ptr[%p], count[%llu], dataType[%d]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), curInputPtr, curCount, param.DataDes.dataType),
            ret);

        ret = RankConsistent::GetInstance().DelOpPara(param.tag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] delete CMD with parameters error. tag[%s]", HCCL_ERROR_CODE(ret),
            param.tag.c_str()), ret);
        CHK_PRT_RET((curCount == 0), HCCL_ERROR("In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        inputOffset = curSize;

        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    }
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::RunTemplate(const OpParam &param, DeviceMem &inputMem)
{
    SubCommInfo commInfo = GetSubCommInfo(COMM_COMBINE, 0);
    if (commInfo.links.size() == 0) {
        HCCL_ERROR("[CollSendExecutor]links size is 0");
    }
    LINK transportLink = commInfo.links[0];

    SendReceive sendExecutor(dispatcher_, transportLink);
    sendExecutor.SendPrepare(inputMem, param.dstRank, param.stream);
    sendExecutor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream);
    sendExecutor.SendRunAsync();

    return HCCL_SUCCESS;
}

REGISTER_EXEC("SendExecutor", Send, CollSendExecutor);

} // namespace hcclss