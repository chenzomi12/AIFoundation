/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_scatter_executor.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "device_capacity.h"
#include "coll_alg_operator.h"

namespace hccl {
CollScatterExecutor::CollScatterExecutor(const HcclDispatcher dispatcher,
                                std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollNativeExecutorBase(dispatcher, topoMatcher)
{
}

HcclResult CollScatterExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollScatterExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_INPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_INPUT;
    }
    return HCCL_SUCCESS;
}

bool CollScatterExecutor::IsHugeData(u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollScatterExecutor::RunLoop(const OpParam &param, const AlgResourceResponse &algRes)
{
    auto dataType = param.DataDes.dataType;
    u32 unitSize = SIZE_TABLE[dataType];
    RankId root = param.root;

    auto totalRecvCount = param.DataDes.count;

    u8 *curUserInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curUserOutputPtr = static_cast<u8 *>(param.outputPtr);
    if (topoAttr_.userRank == root) {
        CHK_PTR_NULL(curUserInputPtr);
    }
    CHK_PTR_NULL(curUserOutputPtr);

    auto inCCLbuffer = algRes.cclInputMem;
    auto outCCLbuffer = algRes.cclOutputMem;

    u64 maxCountPerLoop = inCCLbuffer.size() / (topoAttr_.userRankSize * unitSize); // 中转内存单次最多能够接受的output count

    HCCL_DEBUG("[CollScatterExecutor][RunLoop]tag[%s], userRankSize is [%u], root is [%u], "
               "maxCountPerLoop is [%llu], totalRecvCount is [%llu]",
        tag_.c_str(), topoAttr_.userRankSize, root, maxCountPerLoop, totalRecvCount);

    for (u64 countLeft = totalRecvCount, curRecvCount = 0, inputOffset = 0, outputOffset = 0;
        countLeft > 0; countLeft -= curRecvCount) {
        curUserInputPtr += inputOffset;
        curUserOutputPtr += outputOffset;

        // 判断剩余数据量对应的input size是否大于中转input size
        curRecvCount =
            ((countLeft * unitSize * topoAttr_.userRankSize) > inCCLbuffer.size()) ? maxCountPerLoop : countLeft;
        CHK_PRT_RET((curRecvCount == 0), HCCL_ERROR("[RunLoop][Scatter]In OP_BASE curRecvCount is zero"), HCCL_E_PARA);
        u64 curRecvSize = curRecvCount * unitSize;               // 单位：字节
        u64 curSendSize = topoAttr_.userRankSize * curRecvSize;  // 单位：字节

        DeviceMem curCCLInputMem(inCCLbuffer.ptr(), curSendSize);
        DeviceMem curCCLOutputMem(outCCLbuffer.ptr(), curRecvSize);

        ExecMem execMem;
        execMem.count = curRecvCount;
        execMem.inputMem = curCCLInputMem;
        execMem.outputMem = curCCLOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curUserInputPtr;
        execMem.outputPtr = curUserOutputPtr;

        HCCL_DEBUG("[RunLoop][Scatter] ScatterLoop: inputOffset[%llu], outputOffset[%llu], "
            "curUserInputPtr[%llx], curUserOutputPtr[%llx], curRecvCount[%llu], curRecvSize[%llu], "
            "curSendSize[%llu], inCCLbuffer.ptr[%llx], outCCLbuffer.ptr[%llx]",
            inputOffset, outputOffset, curUserInputPtr, curUserOutputPtr, curRecvCount, curRecvSize,
            curSendSize, inCCLbuffer.ptr(), outCCLbuffer.ptr());

        CHK_RET(RunLoopInner(param, execMem, algRes));

        inputOffset = curRecvSize;
        outputOffset = curRecvSize;
        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    }
    return HCCL_SUCCESS;
}

HcclResult CollScatterExecutor::RunLoopInner(
    const OpParam &param, ExecMem &execMem, const AlgResourceResponse &algRes)
{
    auto dataType = param.DataDes.dataType;
    u32 unitSize = SIZE_TABLE[dataType];
    RankId root = param.root;

    auto totalRecvCount = param.DataDes.count;
    u64 totalRecvSize = totalRecvCount * unitSize;

    u64 recvSize = execMem.outputMem.size();

    auto meta = HcclOpMetaInfo::GetOneForScatter(root, IsHugeData(execMem.outputMem.size()));
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey()));

    DeviceMem dstMem;
    DeviceMem srcMem;
    if (topoAttr_.userRank == root) {
        // 本rank为root节点，非root节点不需要拷贝到中转内存
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为totalRecvSize
            srcMem = DeviceMem::create((u8*)execMem.inputPtr + totalRecvSize * i, recvSize);
            dstMem = algRes.cclInputMem.range(recvSize * i, recvSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }

    /* 记录指令信息用于一致性校验 */
    CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_SCATTER, tag_.c_str(),
        execMem.count, dataType, root, algRes.cclInputMem.size(), algRes.cclOutputMem.size()));

    /* 入参的正确性由HCCL确保 */
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollScatterExecutor][RunLoop]errNo[0x%016llx] OP_BASE hcclComm scatter error, tag[%s], "
                   "input_ptr[%p], output_ptr[%p], recvSize[%llu], data_type[%d], root[%u]",
            HCCL_ERROR_CODE(ret), tag_.c_str(), algRes.cclInputMem.ptr(), algRes.cclOutputMem.ptr(),
            recvSize, dataType, root),
        ret);

    CHK_RET(RankConsistent::GetInstance().DelOpPara(tag_));

    // 将 CCLOut 上的数据搬运到 userOut
    srcMem = algRes.cclOutputMem.range(0, recvSize);
    dstMem = DeviceMem::create(execMem.outputPtr, recvSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    return HCCL_SUCCESS;
}

HcclResult CollScatterExecutor::PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum,
    std::vector<Slice> &dataSlice)
{
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[CollScatterExecutor][PrepareDataSlice]sliceNum is zero."), HCCL_E_PARA);

    dataSlice.resize(sliceNum);
    u64 sliceSize = dataCount * unitSize;
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sliceSize;
        dataSlice[i].offset = (i * sliceSize);
    }
    return HCCL_SUCCESS;
}

HcclResult CollScatterExecutor::ReorderSlice(std::vector<Slice> &dataSlice, std::vector<u32> &order)
{
    CHK_PRT_RET((dataSlice.size() != order.size()),
        HCCL_ERROR("[ReorderSlice] data slize size [%zu], not equal to order size [%zu]",
            dataSlice.size(), order.size()), HCCL_E_INTERNAL);
    std::vector<Slice> tempDataSegsSlice(dataSlice.size());
    for (size_t i = 0; i < dataSlice.size(); i++) {
        CHK_PRT_RET(order[i] >= dataSlice.size(),
            HCCL_ERROR("[ReorderSlice] order value [%zu] >=  dataSlice size [%zu]", order[i], dataSlice.size()),
            HCCL_E_INTERNAL);
        tempDataSegsSlice[i] = dataSlice[order[i]];
    }
    dataSlice = tempDataSegsSlice;
    return HCCL_SUCCESS;
}

HcclResult CollScatterExecutor::KernelRunInner(DeviceMem& inputMem, u64 count, HcclDataType dataType,
    u32 &commIndex, u32 root, u32 &subRoot, CommPlane commLevel, Stream& stream)
{
    CHK_RET(CheckCommSize(commLevel, commIndex + 1));
    SubCommInfo subCommInfo = GetSubCommInfo(commLevel, commIndex);

    u32 subCommSize = subCommInfo.localRankSize;

    if (subCommSize <= 1 || subRoot != topoAttr_.userRank) {
        HCCL_INFO("[ScatterRing][KernelRunInner]: no need to run intra-server, subCommSize[%u], subRoot[%u], "
            "userRank[%u]", subCommSize, subRoot, topoAttr_.userRank);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[ScatterRing][KernelRunInner]: start to run intra-server, subCommSize[%u], subRoot[%u], "
        "userRank[%u]", subCommSize, subRoot, topoAttr_.userRank);

    u32 rootRankInner = 0;
    CHK_RET(GetRankByUserRank(commLevel, commIndex, root, rootRankInner));

    std::unique_ptr<ExecutorBase> innerExecutor;
    if (UseInterServerNBAlgo(algType_)) {
        // server间NB算法走NB
        innerExecutor.reset(new (std::nothrow) ScatterNB(dispatcher_));
        CHK_SMART_PTR_NULL(innerExecutor);
        HCCL_INFO("[ScatterRing][KernelRunInner]: using NB algo inter-server.");
        // 申请临时内存作为scratch内存
        CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * topoAttr_.userRankSize,
            dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0)));
    } else if (UseInterServerNHRAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) ScatterNHR(dispatcher_));
        CHK_SMART_PTR_NULL(innerExecutor);
        HCCL_INFO("[ScatterRing][KernelRunInner]: using NHR algo inter-server.");
        CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * topoAttr_.userRankSize,
            dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0)));
    } else {
        innerExecutor.reset(new (std::nothrow) ScatterRing(dispatcher_));
        CHK_SMART_PTR_NULL(innerExecutor);
        HCCL_INFO("[ScatterRing][KernelRunInner]: using ring algo inter-server.");
        CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * topoAttr_.userRankSize,
            dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0))); // count是output的数据个数
    }
    CHK_RET(innerExecutor->RegisterProfiler(
        (subCommSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(innerExecutor, subCommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollScatterExecutor::Orchestrate(const OpParam& param,
    const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    GetStreamInfo(algRes);
    auto rtStream = param.stream.ptr();
    HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(rtStream, param.tag, 0, algType_);
    CHK_RET(AddSubStreamToProfiling());

    HcclResult ret = HCCL_SUCCESS;
    // 图模式和单卡场景下不需要Loop
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    } else {
        ret = RunLoop(param, algRes);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollScatterExecutor][Orchestrate]errNo[0x%016llx]all reudce excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !is310P3Common_) {
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(param.tag);
    }
    HCCL_INFO("tag[%s] Scatter executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

}
