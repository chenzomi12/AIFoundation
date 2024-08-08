/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_batch_send_recv_executor.h"

namespace hccl {

CollBatchSendRecvExecutor::CollBatchSendRecvExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollBatchSendRecvExecutor::GetSendRecvInfo(HcclSendRecvItem* itemPtr)
{
    remoteUserRank_ = itemPtr->remoteRank;
    sendRecvType_ = itemPtr->sendRecvType;

    return HCCL_SUCCESS;
}

void CollBatchSendRecvExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    HcclSendRecvItem** itemPtr = param.BatchSendRecvDataDes.orderedList;
    u32 itemNum = param.BatchSendRecvDataDes.itemNum;
    commTargetUserRankSet_.clear();
    for (u32 i = 0; i < itemNum; i++) {
        commTargetUserRankSet_.insert((*(itemPtr + i))->remoteRank);
        HCCL_INFO("[CollBatchSendRecvExecutor][ParseParam] insert remoteUserRank[%u] to Set",
            (*(itemPtr + i))->remoteRank);
    }
}

HcclResult CollBatchSendRecvExecutor::CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    (void)ParseParam(param);
    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    bool needAivBuffer = false;
 
    std::vector<LevelNSubCommTransport> opTransport {
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))
    };
    CalcCommInfo(opTransport);
    BuildResourceRequest(scratchMemSize, streamNum, notifyNum, needAivBuffer, opTransport, resourceRequest);
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::Orchestrate(const OpParam& param, const AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;

    algResResp_ = &algResource;
    auto rtStream = param.stream.ptr();
    CHK_RET(GetStreamInfo(algResource));

    HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(rtStream, param.tag, 0, algType_);
    CHK_RET(AddSubStreamToProfiling());

    if (topoMatcher_->GetExternalInputHcclEnableFfts()) {
        auto meta = HcclOpMetaInfo::GetOneForBatchSendRecv();
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey()));
        // 多流子图前后需加空拷贝
        CHK_RET(ExecutorBase::ExecEmptyTask(const_cast<DeviceMem &>(algResource.cclInputMem),
            const_cast<DeviceMem &>(algResource.cclOutputMem), const_cast<Stream&>(param.stream), dispatcher_));
    }

    HCCL_INFO("[BatchSendRecv] Stream sync: main stream record, subStream wait.");
    ret = LocalNotify::Post(const_cast<Stream&>(param.stream), dispatcher_, streamInfo_.ringSignalAux[STREAM_INDEX_0],
        PROF_STAGE_0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BatchSendRecv]substream ringSignalAux record failed"), ret);

    ret = LocalNotify::Wait(streamInfo_.ringStreams[STREAM_INDEX_0], dispatcher_,
        streamInfo_.ringSignalAux[STREAM_INDEX_0], PROF_STAGE_0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BatchSendRecv]substream wait failed"), ret);

    HcclSendRecvItem** itemPtr = param.BatchSendRecvDataDes.orderedList;
    for (u32 i = 0; i < param.BatchSendRecvDataDes.itemNum; i++) {
        HCCL_INFO("[BatchSendRecv] tag[%s], remoteRank[%u], buf[%p], count[%llu], dataType[%s], sendRecvType[%d].",
            tag_.c_str(), (*(itemPtr + i))->remoteRank, (*(itemPtr + i))->buf, (*(itemPtr + i))->count,
            GetDataTypeEnumStr((*(itemPtr + i))->dataType).c_str(), (*(itemPtr + i))->sendRecvType);
        CHK_RET(GetSendRecvInfo(*(itemPtr + i)));
        CHK_RET(RunLoop(param, algResource, i));
    }

    HCCL_INFO("[BatchSendRecv] Stream sync: subStream record, main stream wait.");
    ret = LocalNotify::Post(streamInfo_.ringStreams[STREAM_INDEX_0], dispatcher_,
        streamInfo_.ringSignal[STREAM_INDEX_0], PROF_STAGE_0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BatchSendRecv] substream ringSignal record failed"), ret);

    ret = LocalNotify::Wait(const_cast<Stream&>(param.stream), dispatcher_, streamInfo_.ringSignal[STREAM_INDEX_0],
        PROF_STAGE_0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BatchSendRecv] stream wait failed"), ret);

    if (topoMatcher_->GetExternalInputHcclEnableFfts()) {
        // 多流子图前后需加空拷贝
        CHK_RET(ExecutorBase::ExecEmptyTask(const_cast<DeviceMem &>(algResource.cclInputMem),
            const_cast<DeviceMem &>(algResource.cclOutputMem), const_cast<Stream&>(param.stream), dispatcher_));
        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    }
    HCCL_PROFILER_DEL_STREAM(rtStream);
    HCCL_PROFILER_DEL_TAG(param.tag);
    HCCL_INFO("tag[%s] BatchSendRecv Excutor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::RunLoop(const OpParam &param, const AlgResourceResponse &algRes, u32 index)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclSendRecvItem** sendRecvItem = param.BatchSendRecvDataDes.orderedList + index;
    u32 unitSize = SIZE_TABLE[(*sendRecvItem)->dataType];
    u8 *curInputPtr = 0U;
    u8 *curOutputPtr = 0U;
    u64 maxCountPerLoop = 0U;
    if (sendRecvType_ == HcclSendRecvType::HCCL_SEND) {
        curInputPtr = static_cast<u8 *>((*sendRecvItem)->buf);
        CHK_PTR_NULL(curInputPtr);
        maxCountPerLoop = CalcSendLoopMaxCount(const_cast<DeviceMem&>(algRes.cclInputMem), unitSize);
    } else if (sendRecvType_ == HcclSendRecvType::HCCL_RECV) {
        curOutputPtr = static_cast<u8 *>((*sendRecvItem)->buf);
        CHK_PTR_NULL(curOutputPtr);
        maxCountPerLoop = CalcRecvLoopMaxCount(const_cast<DeviceMem&>(algRes.cclOutputMem), unitSize);
    } else {
        HCCL_ERROR("[CollBatchSendRecvExecutor][RunLoop] sendRecvType is Wrong.");
        return HCCL_E_PARA;
    }
    /* 记录指令信息用于一致性校验 */
    CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV,
        param.tag, algRes.cclInputMem.size(), algRes.cclOutputMem.size(), HCCL_WORLD_GROUP));

    for (u64 countLeft = (*sendRecvItem)->count, curCount = 0, curOffset = 0; countLeft > 0;
        countLeft -= curCount) {
        curInputPtr += curOffset;
        curOutputPtr += curOffset;
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        if ((*sendRecvItem)->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            DeviceMem inMem(curInputPtr, curSize);
            DeviceMem inCommMem = algRes.cclInputMem.range(0, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, const_cast<Stream&>(param.stream)));
        }
        ExecMem execMem;
        execMem.inputMem = algRes.cclInputMem.range(0, curSize);
        execMem.outputMem = algRes.cclOutputMem.range(0, curSize);
        ret = KernelRun(param, execMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollBatchSendRecvExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
            "sendRecvType[%d], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), sendRecvType_, execMem.inputMem.ptr(), execMem.outputMem.ptr(),
            curCount, GetDataTypeEnumStr((*sendRecvItem)->dataType).c_str()), ret);
        if ((*sendRecvItem)->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            DeviceMem outMem(curOutputPtr, curSize);
            DeviceMem outCommMem = algRes.cclOutputMem.range(0, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, streamInfo_.ringStreams[STREAM_INDEX_0]));
        }

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][BatchSendRecv] In OP_BASE curCount is zero."), HCCL_E_PARA);
        curOffset = curSize;
    }
    CHK_RET(RankConsistent::GetInstance().DelOpPara(param.tag));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_COMBINE, COMM_SIZE_TWO));
    u32 commIndex = 0;
    HCCL_INFO("[CollBatchSendRecvExecutor][KernelRun] sendRecvType_[%d], remoteUserRank_[%u], userRank_[%u].",
        sendRecvType_, remoteUserRank_, topoAttr_.userRank);
    if ((sendRecvType_ == HcclSendRecvType::HCCL_SEND && remoteUserRank_ < topoAttr_.userRank) ||
        (sendRecvType_ == HcclSendRecvType::HCCL_RECV && remoteUserRank_ > topoAttr_.userRank)) {
        HCCL_INFO("[CollBatchSendRecvExecutor][KernelRun] CommIndex is 0.");
        commIndex = COMM_INDEX_0;
    } else if ((sendRecvType_ == HcclSendRecvType::HCCL_SEND && remoteUserRank_ > topoAttr_.userRank) ||
                (sendRecvType_ == HcclSendRecvType::HCCL_RECV && remoteUserRank_ < topoAttr_.userRank)) {
        HCCL_INFO("[CollBatchSendRecvExecutor][KernelRun] CommIndex is 1.");
        commIndex = COMM_INDEX_1;
    } else {
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] CommIndex doesn't match.");
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET(commIndex >= algResResp_->opTransportResponse[COMM_COMBINE].size(),
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op commIndex[%u] is larger than "\
        "userRank2subCommRank map size[%u]",
        remoteUserRank_, algResResp_->opTransportResponse[COMM_COMBINE].size()), HCCL_E_NOT_SUPPORT);
    SingleSubCommTransport &commCombined =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_COMBINE][commIndex]);

    CHK_PRT_RET(remoteUserRank_ >= commCombined.userRank2subCommRank.size(),
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op remoteUserRank[%u] is larger than "\
        "userRank2subCommRank map size[%u]",
        remoteUserRank_, commCombined.userRank2subCommRank.size()), HCCL_E_NOT_SUPPORT);

    u32 remoteRank = commCombined.userRank2subCommRank[remoteUserRank_];
    CHK_PRT_RET(remoteRank >= commCombined.links.size(),
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op remoteUserRank[%u], get remoteRank[%u]," \
        "the size of combinedCcomm links is [%u]", remoteUserRank_, remoteRank, commCombined.links.size()),
        HCCL_E_NOT_SUPPORT);
    LINK &targetLink = commCombined.links[remoteRank];
    CHK_SMART_PTR_NULL(targetLink);
    SendReceive executor(dispatcher_, targetLink);

    if (sendRecvType_ == HcclSendRecvType::HCCL_SEND) {
        CHK_RET(executor.SendPrepare(execMem.inputMem, remoteUserRank_, param.stream));
        CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(executor.BatchSendRunAsync());
    } else if (sendRecvType_ == HcclSendRecvType::HCCL_RECV) {
        CHK_RET(executor.ReceivePrepare(execMem.outputMem, remoteUserRank_, streamInfo_.ringStreams[STREAM_INDEX_0]));
        CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
            streamInfo_.ringStreams[STREAM_INDEX_0]));
        CHK_RET(executor.BatchReceiveRunAsync());
    } else {
        HCCL_ERROR("[CollBatchSendRecvExecutor][KernelRun] SendRecvType doesn't match. RemoteRank is [%u]",
            remoteUserRank_);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

u64 CollBatchSendRecvExecutor::CalcSendLoopMaxCount(DeviceMem& inCCLBuffer, const u32 unitSize)
{
    // 中转内存单次最多能够接受的input count
    u64 maxCountPerLoop = inCCLBuffer.size() / unitSize;
    HCCL_WARNING("[CollBatchSendRecvExecutor][CalcSendLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

u64 CollBatchSendRecvExecutor::CalcRecvLoopMaxCount(DeviceMem& outCCLBuffer, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = outCCLBuffer.size() / unitSize;
    HCCL_WARNING("[CollBatchSendRecvExecutor][CalcRecvLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollBatchSendRecvExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 1U;
    HCCL_INFO("[CollBatchSendRecvExecutor][CalcScratchMemSize] tag_[%s], streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}
HcclResult CollBatchSendRecvExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_COMBINE, CommType::COMM_TAG_PARTIAL_MESH_COMBINED, INVALID_VALUE_RANKID,
    INVALID_VALUE_RANKID, false, false, commTargetUserRankSet_);
    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;

    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE], inputType, outputType));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BatchSendRecv", BatchSendRecvExecutor, CollBatchSendRecvExecutor);
} // namespace hccl