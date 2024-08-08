/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_executor.h"

namespace hccl {

CollReduceScatterExecutor::CollReduceScatterExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceScatterExecutor::Orchestrate(const OpParam& param,
    const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    ParseParam(param);
    tag_ = param.tag;
    GetStreamInfo(algRes);
    algResResp_ = &algRes;
    auto rtStream = param.stream.ptr();
    HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(rtStream, param.tag, 0, algType_);
    HCCL_PROFILER_ADD_OPDATA(param.tag, param.DataDes.count, param.inputPtr, param.outputPtr, param.DataDes.dataType, \
        INVALID_VALUE_RANKID, algoAttr_.identifier);
    HCCL_PROFILER_ADD_GROUPRANK(algoAttr_.identifier, topoAttr_.userRankSize, topoAttr_.userRank);
    CHK_RET(AddSubStreamToProfiling());

    HcclResult ret = HCCL_SUCCESS;
    // 图模式和单卡场景下不需要Loop
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        ExecMem execMem;
        execMem.count = param.DataDes.count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    } else if (topoAttr_.userRankSize == 1) {
        ExecMem execMem;
        execMem.count = param.DataDes.count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    } else {
        ret = RunLoop(param, algRes);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterExecutor][Orchestrate]errNo[0x%016llx]excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !is310P3Common_) {
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(param.tag);
        HCCL_PROFILER_DEL_OPDATA(param.tag);
        HCCL_PROFILER_DEL_GROUPRANK(param.tag);
    }
    HCCL_INFO("tag[%s], ReduceScatter executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    HCCL_INFO("[CollReduceScatterExecutor][CalcLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / (userRankSize * unitSize).", maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollReduceScatterExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                            (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

bool CollReduceScatterExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    HCCL_INFO("[CollReduceScatterExecutor][IsSmallData]opMeta is using the default option: not small data.");
    return false;
}

u32 CollReduceScatterExecutor::CalcDataSplit(const u64 curSize)
{
    HCCL_INFO("[CollReduceScatterExecutor][CalcDataSplit]opMeta is using the default option: not data split.");
    return 0;
}

HcclResult CollReduceScatterExecutor::RunLoop(const OpParam &param, const AlgResourceResponse &algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);
    HCCL_DEBUG("[CollReduceScatterExecutor][RunLoop]tag[%s], userRankSize is [%llu], maxCountPerLoop is [%llu].",
        param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    HcclResult ret;
    for (u64 countLeft = param.DataDes.count, curCount = 0, inputOffset = 0, outputOffset = 0;
            countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], " \
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d].",
            param.tag.c_str(), inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        if (scratchMemFlag_) {
            execMem.scratchMem = algRes.scratchMem;
        } else {
            execMem.scratchMem = algRes.cclOutputMem; // 不需要申请则传入outputmem为scratchmem
        }
        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoop]scratchMem address [%p]", execMem.scratchMem.ptr());

        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        ret = RunLoopInner(param, reduceType, execMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollReduceScatterExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s]",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterExecutor::RunLoopInner(const OpParam &param, const ReduceType &reduceType, ExecMem &execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    CHK_PRT_RET((execMem.count == 0),
        HCCL_ERROR("[CollReduceScatterExecutor][RunLoopInner]In OP_BASE curCount is zero."), HCCL_E_PARA);

    if (!is310P3Common_) {
        /* 设置子图复用标志 */
        auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
        bool hugeData = IsHugeData(curSize);
        bool smallData = IsSmallData(param.DataDes.count * unitSize, curSize);
        auto opMeta = HcclOpMetaInfo::GetOneForReduceScatter(autoSelectedAlgTypeLevel1,
            param.DataDes.dataType, reduceType, hugeData, smallData);
        opMeta.dataSplit = CalcDataSplit(curSize);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, param.tag,
            execMem.count, param.DataDes.dataType, param.reduceType, execMem.inputMem.size(),
            execMem.outputMem.size()));
    }

    if (CCLMemSlice_) {
        execMem.inputMem = execMem.inputMem.range(0, curSize * topoAttr_.userRankSize);
        execMem.outputMem = execMem.outputMem.range(0, curSize);
        if (scratchMemFlag_) {
            execMem.scratchMem = execMem.scratchMem.range(0, curSize * topoAttr_.userRankSize);
        }
    }

    // 执行
    if (!DMAReduceFlag_) {
        // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
        DeviceMem dstMem;
        DeviceMem srcMem;
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
            dstMem = execMem.inputMem.range(curSize * i, curSize);
            srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr) + param.DataDes.count * unitSize * i,
                curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        }
        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoopInner]copy from user in to ccl in.");
    }

    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterExecutor][RunLoopInner]errNo[0x%016llx]kernel run error, tag[%s], " \
        "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
        HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(),
        execMem.count, param.DataDes.dataType, param.reduceType),
        ret);

    if (!DMAReduceFlag_) {
        // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
        DeviceMem srcMem = execMem.outputMem.range(0, curSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    if (!is310P3Common_) {
        CHK_RET(RankConsistent::GetInstance().DelOpPara(param.tag));
        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    }
    return ret;
}

std::vector<std::vector<Slice>> CollReduceScatterExecutor::ReduceScatterRingSlicePrepare(u32 ringNum, u32 sliceNum,
    bool useInlineReduce, DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice, const std::string &tag)
{
    std::vector<std::vector<Slice>> multiStreamSlice;
    u64 outputMenSize = outputMem.size();
    dataSegsSlice.clear();
    Slice sliceTemp;
    for (u32 i = 0; i < sliceNum; i++) {    // 根据数据量算每个环上数据的偏移和大小
        sliceTemp.size = outputMenSize;
        sliceTemp.offset = outputMenSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }

    // 再将每个 slice 划分为 ringNum 份
    if (ringNum == OUTER_PLANE_NUM_IN_8PRING) {
        if (useInlineReduce) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag);
        } else if (outputMem.size() % CCE_REDUCE_ALIGN_SIZE == 0) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag);
        } else {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, true);
        }
    } else if (ringNum == OUTER_PLANE_NUM_IN_NPRING_DOUBLE) {
        // 双环场景，需要传入正确的 niclist (不涉及网口裁剪)
        if (useInlineReduce) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, topoAttr_.nicList);
        } else if (outputMem.size() % CCE_REDUCE_ALIGN_SIZE == 0) {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, topoAttr_.nicList);
        } else {
            multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, tag, true, topoAttr_.nicList);
        }
    } else {
        multiStreamSlice.push_back(dataSegsSlice);
    }

    return multiStreamSlice;
}

} // namespace hccl