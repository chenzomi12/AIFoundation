/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_executor.h"

namespace hccl {
CollAllGatherExecutor::CollAllGatherExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAllGatherExecutor::Orchestrate(const OpParam& param,
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
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !is310P3Common_) {
        HCCL_PROFILER_ADD_OPDATA(param.tag, param.DataDes.count, param.inputPtr, param.outputPtr,
            param.DataDes.dataType, INVALID_VALUE_RANKID, algoAttr_.identifier);
        HCCL_PROFILER_ADD_GROUPRANK(algoAttr_.identifier, topoAttr_.userRankSize, topoAttr_.userRank);
    }

    HcclResult ret = HCCL_SUCCESS;
    // 图模式和单卡场景下不需要Loop
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 totalSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
        execMem.inputMem = DeviceMem::create(algRes.paramInputMem.ptr(), totalSize);
        execMem.outputMem = DeviceMem::create(algRes.paramOutputMem.ptr(), totalSize * topoAttr_.userRankSize);
        execMem.scratchMem = algRes.scratchMem;
        HCCL_DEBUG("[CollAllGatherExecutor][Orchestrate]offload inputMem[%p][%u], outputMem[%p][%u]," \
            "scratchMem[%p][%u], inputPtr[%p] outputPtr[%p], count[%llu]",
            execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
            execMem.scratchMem.ptr(), execMem.scratchMem.size(), execMem.inputPtr, execMem.outputPtr, execMem.count);
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
        HCCL_ERROR("[CollAllGatherExecutor][Orchestrate]errNo[0x%016llx]all reudce excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !is310P3Common_) {
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(param.tag);
        HCCL_PROFILER_DEL_OPDATA(param.tag);
        HCCL_PROFILER_DEL_GROUPRANK(param.tag);
    }
    HCCL_INFO("tag[%s], Allgather executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}


u64 CollAllGatherExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / (unitSize * topoAttr_.userRankSize);
    HCCL_WARNING("[CollAllGatherExecutor][CalcLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollAllGatherExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
            curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

u32 CollAllGatherExecutor::IsDataSplit(const u64 curSize)
{
    HCCL_INFO("[CollAllGatherExecutor][IsDataSplit]opMeta is using the default option: not data split.");
    return 0;
}

HcclResult CollAllGatherExecutor::RunLoop(const OpParam &param, const AlgResourceResponse &algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(algRes.cclInputMem.size(), unitSize);   // override
    HCCL_DEBUG("[CollAllGatherExecutor][RunLoop]tag[%s], userRankSize is [%llu], maxCountPerLoop is [%llu].",
        param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    for (u64 countLeft = param.DataDes.count, curCount = 0, inputOffset = 0, outputOffset = 0;
            countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG("[CollAllGatherExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], " \
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
        CHK_RET(RunLoopInner(param, execMem));

        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherExecutor::RunLoopInner(const OpParam &param, ExecMem &execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    void *commInputPtr = execMem.inputMem.ptr();
    void *commOutputPtr = execMem.outputMem.ptr();
    CHK_PRT_RET((execMem.count == 0),
        HCCL_ERROR("[CollAllGatherExecutor][RunLoop]In OP_BASE curCount is zero."), HCCL_E_PARA);

    if (!is310P3Common_) {
        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER,
            param.tag, execMem.count, param.DataDes.dataType, execMem.inputMem.size(), execMem.outputMem.size(),
            HCCL_WORLD_GROUP));
        /* 设置子图复用标志 */
        auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
        bool hugeData = IsHugeData(curSize);    // override
        u32 dataSplit = IsDataSplit(curSize);
        auto opMeta = HcclOpMetaInfo::GetOneForAllGather(autoSelectedAlgTypeLevel1, hugeData);
        opMeta.dataSplit = dataSplit;
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    }

    execMem.inputMem = DeviceMem::create(commInputPtr, curSize);
    execMem.outputMem = DeviceMem::create(commOutputPtr, curSize * topoAttr_.userRankSize);
    HCCL_DEBUG("[CollAllGatherExecutor][RunLoopInner]inputMem[%p][%llu], outputMem[%p][%llu], " \
        "intputPtr[%p], outputPtr[%p], curCount[%llu], curSize[%llu]",
        execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
        execMem.inputPtr, execMem.outputPtr, execMem.count, curSize);

    // 执行
    if (!DMAReduceFlag_) {
        // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
        DeviceMem srcMem = DeviceMem::create(execMem.inputPtr, curSize);
        DeviceMem dstMem = DeviceMem::create(commInputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        HCCL_DEBUG("[CollAllGatherExecutor][RunLoop]copy from user in to ccl in.");
    }
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
        "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
        HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(),
        execMem.count, param.DataDes.dataType),
        ret);

    if (!DMAReduceFlag_) {
        // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            // 拷贝中转output上每个slice的数据到output内存，目的端中每个slice的size固定为output的size
            u8 *curOutputPtr = static_cast<u8 *>(execMem.outputPtr);
            DeviceMem dstMem = DeviceMem::create(curOutputPtr + param.DataDes.count * unitSize * i, curSize);
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(commOutputPtr) + curSize * i, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }

    if (!is310P3Common_) {
        CHK_RET(RankConsistent::GetInstance().DelOpPara(param.tag));
        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    }
    return ret;
}

HcclResult CollAllGatherExecutor::PrepareAllgatherSlice(u32 sliceNum, u64 inputMemSize,
    std::vector<Slice> &dataSegsSlice) const
{
    Slice sliceTemp;
    for (u32 i = 0; i < sliceNum; i++) { // 根据数据量计算每个环上数据的偏移和大小
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = inputMemSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherExecutor::CalculateLevel1AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
    std::vector<std::vector<Slice>> multRingsSliceZero, std::vector<std::vector<Slice>> &multRingsSlice) const
{
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level1DataSlice;
        for (u32 level0Idx = 0; level0Idx < level0RankSize; level0Idx++) {
            for (u32 level1Idx = 0; level1Idx < level1RankSize; level1Idx++) {
                Slice tmpSlice;
                tmpSlice.size = multRingsSliceZero[ringIndex][level0Idx].size;
                tmpSlice.offset =
                    multRingsSliceZero[ringIndex][level0Idx].offset + level1Idx * level0RankSize * inputMemSize;
                level1DataSlice.push_back(tmpSlice);
            }
        }
        multRingsSlice.push_back(level1DataSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherExecutor::CalculateLevel2AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
    u32 level2RankSize, std::vector<Slice> dataSegsSlice, std::vector<Slice> &level0DataSlice) const
{
    for (u32 i = 0; i < level0RankSize; i++) {
        for (u32 j = 0; j < level1RankSize; j++) {
            for (u32 z = 0; z < level2RankSize; z++) {
                Slice rankSliceTemp;
                rankSliceTemp.size = dataSegsSlice[i].size;
                rankSliceTemp.offset = dataSegsSlice[i].offset +
                    (j * level0RankSize * level1RankSize +  z * level1RankSize) * inputMemSize;
                level0DataSlice.push_back(rankSliceTemp);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherExecutor::AllGatherLevel2(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, Stream &stream, const HcomCollOpInfo *opInfo)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = outerCommInfo.localRank;
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, commIndex + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, commIndex);

    u64 inputMemSize = inputMem.size();
    u32 level0RankSize = outerCommInfo.localRankSize;
    u32 level1RankSize = innerCommInfo.localRankSize;
    u32 level2RankSize = level2CommInfo.localRankSize;
    u32 level0ServerIndex = outerCommInfo.localRank;
    u32 level1ServerIndex = innerCommInfo.localRank;

    std::unique_ptr<ExecutorBase> level2AGExecutor;
    if (UseLevel2RingAlgo(algType_)) {
        level2AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        HCCL_INFO("allgather ring: using ring algo inter-server.");
    } else {
        level2AGExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
        HCCL_INFO("allgather ring: using halving-doubling algo inter-server.");
    }

    // 计算slice, 不同超节点相同slice
    std::vector<Slice> level2DataSegsSlice;
    Slice sliceTemp;
    for (u32 i = 0; i < level2RankSize; i++) {
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = i * level1RankSize * level0RankSize * inputMemSize;
        level2DataSegsSlice.push_back(sliceTemp);
    }
    //  outputMem传整块，通过baseOffset偏移
    u64 level2BaseOffset = (level0ServerIndex + level1ServerIndex * level1RankSize) * inputMemSize;
    CHK_RET(level2AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, level2BaseOffset));

    CHK_RET(level2AGExecutor->RegisterProfiler((
        level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
    HCCL_INFO("allgather double ring [superpod] level2 allgather run success");

    // 第二步，各个AI Server 间 all gather (ring/NHR)
    HCCL_INFO("commIdx:%u Tag[%s].commInner.size():%u", commIndex, tag.c_str(),
        level1RankSize);

    std::unique_ptr<ExecutorBase> level1AGExecutor;
    if (UseInterServerRingAlgo(algType_)) {
        level1AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        HCCL_INFO("allgather ring: using ring algo inter-server.");
    } else {
        level1AGExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
        HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
    }

    // 计算slice, 不同超节点相同slice
    std::vector<Slice> level1DataSegsSlice;
    for (u32 j = 0; j < level1RankSize; j++) {
        for (u32 i = 0; i < level2RankSize; i++) {
            sliceTemp.size = inputMemSize;
            sliceTemp.offset =
                j * level0RankSize *inputMemSize + i * level1RankSize * level0RankSize * inputMemSize;
            level1DataSegsSlice.push_back(sliceTemp);
        }
    }
    //  outputMem传整块，通过baseOffset偏移?
    u64 level1BaseOffset = level0ServerIndex * inputMemSize;
    CHK_RET(level1AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, level1BaseOffset));

    CHK_RET(level1AGExecutor->RegisterProfiler((
        level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(level1AGExecutor, innerCommInfo));
    HCCL_INFO("allgather double ring [superpod] level1 allgather run success");

    // 节点内做all gather double ring
    std::vector<Slice> dataSegsSlice;
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    // 多环数据切分
    multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag, false, topoAttr_.nicList);

    // 计算slice
    std::vector<std::vector<Slice>> multRingsSlice;
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level0DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(inputMemSize, level0RankSize, level1RankSize,
            level2RankSize, dataSegsSlice, level0DataSlice));
        multRingsSlice.push_back(level0DataSlice);
    }

    CHK_RET(ActiveSlaveStreams(stream));
    CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count, dataType,
                               multRingsSliceZero, stream, PROF_STAGE_1, 0, opInfo));
    HCCL_INFO("allgather double ring [superpod] level2 allgather run success");
    return HCCL_SUCCESS;
}

} // namespace hccl