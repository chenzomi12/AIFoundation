/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_executor.h"

namespace hccl {

CollAllReduceExecutor::CollAllReduceExecutor(const HcclDispatcher dispatcher,
                                             std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAllReduceExecutor::Orchestrate(const OpParam& param,
    const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    GetStreamInfo(algRes);
    auto rtStream = param.stream.ptr();
    HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(rtStream, param.tag, 0, algType_);
    HCCL_PROFILER_ADD_OPDATA(param.tag, param.DataDes.count, param.inputPtr, param.outputPtr, param.DataDes.dataType, \
        param.root, algoAttr_.identifier);
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
    } else if ((param.inputPtr == algRes.cclInputMem.ptr()) && (param.outputPtr == algRes.cclOutputMem.ptr())) {
        ret = AvoidSubgraphLoop(param, algRes);
    } else {
        ret = RunLoop(param, algRes);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceExecutor][Orchestrate]errNo[0x%016llx]all reudce excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !is310P3Common_) {
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(param.tag);
        HCCL_PROFILER_DEL_OPDATA(param.tag);
        HCCL_PROFILER_DEL_GROUPRANK(param.tag);
    }
    HCCL_INFO("tag[%s], AllReduce executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

u64 CollAllReduceExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / unitSize;
    HCCL_WARNING("[CollAllReduceExecutor][CalcLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollAllReduceExecutor::IsHugeData(const u64 curSize)
{
    HCCL_WARNING("[CollAllReduceExecutor][IsHugeData]opMeta is using the default option: not huge data.");
    return false;
}

bool CollAllReduceExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    HCCL_WARNING("[CollAllReduceExecutor][IsSmallData]opMeta is using the default option: not small data.");
    return false;
}

u64 CollAllReduceExecutor::GetSliceNum(const u64 totalSize)
{
    const AlgTypeLevel0 algLevel0 = GetLevel0AlgType(algType_);
    u64 actualSize = 0;
    u32 actualRankSize = 0;
    u64 sliceNum = 0;

    if (algLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
        // level0算法配null走单层拓扑场景
        actualSize = totalSize;
        actualRankSize = topoAttr_.userRankSize;
    } else {
        // 非单层拓扑场景
        const u32 localRankSize = topoAttr_.deviceNumPerAggregation;
        const u32 localRank = topoAttr_.userRank % localRankSize;
        const u64 tempPerSlice = (totalSize + localRankSize - 1) / localRankSize;
        const u64 sizePerSlice =
            ((tempPerSlice + (HCCL_MIN_SLICE_ALIGN - 1)) / HCCL_MIN_SLICE_ALIGN) * HCCL_MIN_SLICE_ALIGN;

        if ((localRank + 1) * sizePerSlice < totalSize) {
            actualSize = sizePerSlice;
        } else if (localRank * sizePerSlice < totalSize) {
            actualSize = totalSize - localRank * sizePerSlice;
        }

        actualRankSize = topoAttr_.userRankSize / localRankSize;
    }

    if (UseInterServerNBAlgo(algType_)) {
        if (totalSize > HCCL_MIN_SLICE_ALIGN) {
            u64 sliceSize = GetSliceSizeOfNB(actualSize, actualRankSize);
            sliceNum = std::ceil(actualSize * 1.0f / sliceSize);
        }
    }
    return sliceNum;
}

HcclResult CollAllReduceExecutor::RunLoop(const OpParam &param, const AlgResourceResponse &algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(algRes.cclInputMem.size(), unitSize);   // override
    if (maxCountPerLoop == 0) {
        HCCL_ERROR("[CollAllReduceExecutor][RunLoop]tag[%s], userRankSize is [%llu], maxCountPerLoop is [%llu].",
            param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[CollAllReduceExecutor][RunLoop]tag[%s], userRankSize is [%llu], maxCountPerLoop is [%llu].",
        param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    for (u64 countLeft = param.DataDes.count, curCount = 0, inputOffset = 0, outputOffset = 0;
            countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG("[CollAllReduceExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], " \
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d].",
            param.tag.c_str(), inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        CHK_RET(RunLoopInner(param, reduceType, execMem));

        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceExecutor::RunLoopInner(const OpParam &param, const ReduceType &reduceType, ExecMem &execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    HCCL_DEBUG("[CollAllReduceExecutor][RunLoopInner]inputMem[%p][%llu], outputMem[%p][%llu], " \
        "intputPtr[%p], outputPtr[%p], curCount[%llu], curSize[%llu]",
        execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
        execMem.inputPtr, execMem.outputPtr, execMem.count, curSize);
    CHK_PRT_RET((execMem.count == 0),
        HCCL_ERROR("[CollAllReduceExecutor][RunLoop]In OP_BASE curCount is zero."), HCCL_E_PARA);

    if (!is310P3Common_) {
        /* 设置子图复用标志 */
        auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
        bool hugeData = IsHugeData(curSize);    // override

        if (reduceType == ReduceType::TBE_REDUCE) {
            /* TBE reduce 当总count数超过INT32_MAX时，不使能子图复用 */
            hugeData = hugeData || param.DataDes.count > INT32_MAX;
        }

        u64 sliceNum = GetSliceNum(execMem.count * unitSize);
        bool smallData = IsSmallData(param.DataDes.count * unitSize, curSize);  // override
        u32 dataSplit = curSize <= HCCL_SDMA_RDMA_SPLIT_SIZE ? 1 : 0;
        auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(autoSelectedAlgTypeLevel1,
            param.DataDes.dataType, reduceType, smallData, 1, hugeData, CopyPattern::BCOPY, sliceNum);
        opMeta.dataSplit = dataSplit;
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLREDUCE,
            param.tag, execMem.count, param.DataDes.dataType, param.reduceType, execMem.inputMem.size(),
            execMem.outputMem.size()));
    }

    if (CCLMemSlice_) {
        execMem.inputMem = DeviceMem::create(execMem.inputMem.ptr(), curSize);
        execMem.outputMem = DeviceMem::create(execMem.outputMem.ptr(), curSize);
    }

    // 执行
    if (!DMAReduceFlag_) {
        // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
        DeviceMem inMem(execMem.inputPtr, curSize);
        DeviceMem inCommMem = execMem.inputMem.range(0, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, const_cast<Stream&>(param.stream)));
        HCCL_DEBUG("[CollAllReduceExecutor][RunLoop]copy from user in to ccl in.");
    }
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
        "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
        HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(),
        execMem.count, param.DataDes.dataType, param.reduceType),
        ret);

    if (!DMAReduceFlag_) {
        // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
        DeviceMem outCommMem = execMem.outputMem.range(0, curSize);
        DeviceMem outMem(execMem.outputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, const_cast<Stream&>(param.stream)));
    }

    if (!is310P3Common_) {
        CHK_RET(RankConsistent::GetInstance().DelOpPara(param.tag));
        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    }
    return ret;
}

HcclResult CollAllReduceExecutor::AvoidSubgraphLoop(const OpParam &param, const AlgResourceResponse &algRes)
{
    HCCL_DEBUG("[CollAllReduceExecutor][AvoidSubgraphLoop]start.");

    u64 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    bool hugeData =
        (param.DataDes.count * unitSize) / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE >
        RDMA_SEND_MAX_SIZE || (param.DataDes.count * unitSize) > SDMA_SEND_MAX_SIZE;
    auto opMeta =
        HcclOpMetaInfo::GetOneForAllReduce(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_128_KB, 1, hugeData, CopyPattern::ZCOPY);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    /* 记录指令信息用于一致性校验 */
    CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLREDUCE, param.tag, param.DataDes.count,
        param.DataDes.dataType, param.reduceType, algRes.cclInputMem.size(), algRes.cclOutputMem.size()));

    DeviceMem src(param.inputPtr, 0);
    DeviceMem dst(algRes.cclInputMem.ptr(), 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, const_cast<Stream&>(param.stream)));
    /* 入参的正确性由HCCL确保 */
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    execMem.scratchMem = algRes.scratchMem;
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Loop][Allreduce]errNo[0x%016llx] param.reduceTypebase hcclComm all reduce error, " \
        "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%d], op[%d]",
        HCCL_ERROR_CODE(ret), param.tag.c_str(), param.inputPtr, param.outputPtr, param.DataDes.count,
        param.DataDes.dataType, param.reduceType), ret);
    CHK_RET(RankConsistent::GetInstance().DelOpPara(param.tag));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    return HCCL_SUCCESS;
}

bool CollAllReduceExecutor::IsAllReduceSmallData(u64 size)
{
    if (UseInterServerNHRAlgo(algType_)) {
        const AlgTypeLevel0 algLevel0 = GetLevel0AlgType(algType_);
        if (algLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED) { // level0算法配null走单层拓扑场景
            if (size <= NHR_ALLREDUCE_SMALL_SIZE) {
                return true;
            }
        } else {
            if (size / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
                return true;
            }
        }
    }

    return false;
}

HcclResult CollAllReduceExecutor::PrepareSliceDataWithAlignSize(u64 totalSize, u32 sliceNum,
    u64 piplineOffset, std::vector<Slice>& dataSlice, u64 alignSize)
{
    Slice temp;
    dataSlice.clear();
    dataSlice.reserve(sliceNum);
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[Prepare][SliceData]data slice prepare, sliceNum is 0"), HCCL_E_PARA);
    u64 tempPerSlice = (totalSize + sliceNum - 1) / sliceNum; /* 1是为了向上取整 */
    u64 sizePerSlice = ExecutorBase::RoundUpWithDivisor(tempPerSlice, alignSize);
    HCCL_DEBUG("total_size:%llu sliceNum:%u temp_per_ring:%llu size_per_ring:%llu", totalSize, sliceNum, tempPerSlice,
        sizePerSlice);
    u64 residueSize = totalSize;
    u32 i = 0;
    while (residueSize > 0) {
        u64 sliceSize = sizePerSlice < residueSize ? sizePerSlice : residueSize;
        temp.size = sliceSize;
        temp.offset = totalSize - residueSize + piplineOffset;
        i++;
        CHK_PRT_RET((sliceSize <= 0), HCCL_ERROR("[Prepare][SliceData]data_slice_prepare sliceSize[%llu]", sliceSize),
            HCCL_E_PARA);
        residueSize -= sliceSize;
        dataSlice.push_back(temp);
    }
    while (i < sliceNum) {
        temp.size = 0;
        temp.offset = totalSize + piplineOffset;
        i++;
        dataSlice.push_back(temp);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceExecutor::PrepareAivBuffers(u32 rankSize, u32 rankId, u32 rankOffset,
    DeviceMem &inputMem, DeviceMem &outputMem, std::vector<LINK> &links, void **dataBuffers, void **flagBuffers,
    UserMemType dataMemType, UserMemType flagMemType, u32 dataMemOffset, u32 flagMemOffset)
{
    void *tmpCCLBufferData;
    void *tmpCCLBufferFlag;
    for (u32 i = 0; i < rankSize; i++) {
        if (i != rankId) {
            if (links[i + rankOffset] != nullptr) {
                CHK_RET(links[i + rankOffset]->GetRemoteMem(dataMemType, &(tmpCCLBufferData)));
                CHK_RET(links[i + rankOffset]->GetRemoteMem(flagMemType, &(tmpCCLBufferFlag)));
                dataBuffers[i] = static_cast<u8 *>(tmpCCLBufferData) + dataMemOffset;
                flagBuffers[i] = static_cast<u8 *>(tmpCCLBufferFlag) + flagMemOffset;
            }
        } else {
            dataBuffers[i] = static_cast<u8 *>(inputMem.ptr()) + dataMemOffset;
            flagBuffers[i] = static_cast<u8 *>(outputMem.ptr()) + flagMemOffset;
        }
    }
    return HCCL_SUCCESS;
}

} // namespace hccl