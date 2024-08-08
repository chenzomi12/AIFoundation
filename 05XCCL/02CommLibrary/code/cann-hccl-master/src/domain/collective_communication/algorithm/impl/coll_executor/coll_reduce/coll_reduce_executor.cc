/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_reduce_executor.h"

namespace hccl {

CollReduceExecutor::CollReduceExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceExecutor::Orchestrate(const OpParam& param, const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();

    tag_ = param.tag;
    if (UseInterServerHDAlgo(algType_)) {
        u32 part1Size = 2 * (topoAttr_.moduleNum - (1 << static_cast<u32>(log2(topoAttr_.moduleNum))));
        u32 rootId = param.root / topoAttr_.deviceNumPerAggregation;
        std::string appendTag = std::to_string((rootId >= part1Size) || ((rootId % 2) == 0));
        tag_ = param.tag + '_' + appendTag;
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    }

    algResResp_ = &algRes;
    GetStreamInfo(algRes);
    auto rtStream = param.stream.ptr();
    HCCL_PROFILER_ADD_TAG(tag_, algoAttr_.identifier, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(rtStream, tag_, 0, algType_);
    HCCL_PROFILER_ADD_OPDATA(tag_, param.DataDes.count, param.inputPtr, param.outputPtr, param.DataDes.dataType, \
        param.root, algoAttr_.identifier);
    HCCL_PROFILER_ADD_GROUPRANK(algoAttr_.identifier, topoAttr_.userRankSize, topoAttr_.userRank);
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
    } else if (topoAttr_.userRankSize == 1) {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    } else {
        ret = RunLoop(param, algRes);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceExecutor][Orchestrate]errNo[0x%016llx]reudce excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !is310P3Common_) {
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(tag_);
        HCCL_PROFILER_DEL_OPDATA(tag_);
        HCCL_PROFILER_DEL_GROUPRANK(tag_);
    }

    HCCL_INFO("tag[%s], Reduce executor orchestrate success, take time [%lld]us.", tag_.c_str(),
        DURATION_US(TIME_NOW() - startut));

    return HCCL_SUCCESS;
}

HcclResult CollReduceExecutor::RunLoop(const OpParam &param, const AlgResourceResponse &algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize, algRes);   // override

    HCCL_DEBUG("[CollReduceExecutor][RunLoop]tag[%s], userRankSize is [%llu], maxCountPerLoop is [%llu].",
        tag_.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft =  param.DataDes.count;
    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG("[CollReduceExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], " \
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d].",
            tag_.c_str(), inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        CHK_RET(RunLoopInner(param, reduceType, execMem));

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceExecutor::RunLoopInner(const OpParam &param, const ReduceType &reduceType, ExecMem &execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    HCCL_DEBUG("[CollReduceExecutor][RunLoopInner]inputMem[%p][%llu], outputMem[%p][%llu], " \
        "intputPtr[%p], outputPtr[%p], curCount[%llu], curSize[%llu]",
        execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
        execMem.inputPtr, execMem.outputPtr, execMem.count, curSize);
    CHK_PRT_RET((execMem.count == 0),
        HCCL_ERROR("[CollAllReduceExecutor][RunLoop]In OP_BASE curCount is zero."), HCCL_E_PARA);

    /* 设置子图复用标志 */
    bool isRootRank = param.root == topoAttr_.realUserRank ? true : false;
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    bool hugeData = IsHugeData(curSize);    // override
    auto opMeta =
        HcclOpMetaInfo::GetOneForReduce(isRootRank, param.root, autoSelectedAlgTypeLevel1, param.DataDes.dataType, reduceType, hugeData);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    /* 记录指令信息用于一致性校验 */
    CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE,
        tag_, execMem.count, param.DataDes.dataType, param.reduceType, param.root, execMem.inputMem.size(),
        execMem.outputMem.size()));

    execMem.inputMem = DeviceMem::create(execMem.inputMem.ptr(), curSize);
    execMem.outputMem = DeviceMem::create(execMem.outputMem.ptr(), curSize);

    // 执行
    // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
    DeviceMem inMem(execMem.inputPtr, curSize);
    DeviceMem inCommMem = execMem.inputMem.range(0, curSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, const_cast<Stream&>(param.stream)));
    HCCL_DEBUG("[CollReduceExecutor][RunLoop]copy from user in to ccl in.");

    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
        "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
        HCCL_ERROR_CODE(ret), tag_.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(),
        execMem.count, param.DataDes.dataType, param.reduceType),
        ret);

    if (topoAttr_.realUserRank == param.root) { // 只root rank需要把数据从中转内存拷贝出去
        DeviceMem outMem(execMem.outputPtr, curSize);
        DeviceMem outCommMem = execMem.outputMem.range(0, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, const_cast<Stream&>(param.stream)));
    }

    CHK_RET(RankConsistent::GetInstance().DelOpPara(tag_));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    return ret;
}

u64 CollReduceExecutor::CalcLoopMaxCount(const u32 unitSize, const AlgResourceResponse& algRes)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = algRes.cclInputMem.size() / unitSize;
    HCCL_WARNING("[CollReduceExecutor][CalcLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollReduceExecutor::IsHugeData(const u64 curSize)
{
    HCCL_WARNING("[CollReduceExecutor][IsHugeData]opMeta is using the default option.");
    bool hugeData = (curSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                    (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}
}