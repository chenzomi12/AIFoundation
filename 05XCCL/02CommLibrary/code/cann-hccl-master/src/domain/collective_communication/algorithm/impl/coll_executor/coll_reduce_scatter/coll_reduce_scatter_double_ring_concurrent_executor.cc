/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_double_ring_concurrent_executor.h"

namespace hccl {

CollReduceScatterDoubleRingConcurrentExecutor::CollReduceScatterDoubleRingConcurrentExecutor(
    const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

void CollReduceScatterDoubleRingConcurrentExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
        IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = inCCLbufferSize_ + CCE_REDUCE_ALIGN_FACTOR * CCE_REDUCE_ALIGN_SIZE;
        } else {
            scratchMemSize = totalSize_ + CCE_REDUCE_ALIGN_FACTOR * CCE_REDUCE_ALIGN_SIZE;
        }
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO("[CollReduceScatterDoubleRingConcurrentExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%u]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    // DoubleRing只支持910_73场景
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    }
    if (GetExternalInputEnableRdmaSdmaConcurrent()) {
        totalStreamNum += RDMA_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollReduceScatterDoubleRingConcurrentExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::CCL_OUTPUT;
        }
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::PARAM_OUTPUT;
        }
    }
    HCCL_INFO("[CollReduceScatterDoubleRingConcurrentExecutor][CalcTransportMemType] tag[%s] "
        "inputType[%d], outputType[%d]", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    commParaLevel0.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    if (GetExternalInputEnableRdmaSdmaConcurrent()) {
        CommParaInfo commParaLevel0Rdma(COMM_LEVEL0_RDMA, CommType::COMM_TAG_RING_INNER);
        commParaLevel0Rdma.forceRdma = true;
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0Rdma, opTransport[COMM_LEVEL0_RDMA],
            inputType, outputType));
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (UseLevel2RingAlgo(algType_)) {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterDoubleRingConcurrentExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterDoubleRingConcurrentExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                   (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

u32 CollReduceScatterDoubleRingConcurrentExecutor::CalcDataSplit(const u64 curSize)
{
    u32 dataSplit = 0;
    u64 dataValue = curSize * topoAttr_.userRankSize;
    if ((topoAttr_.serverNum > 1) && ((dataValue / topoAttr_.serverNum) <= HCCL_SDMA_RDMA_SPLIT_SIZE)) {
        dataSplit = 1;
    } else if (dataValue <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
        dataSplit = HCCL_SPLIT_FLAG;
    }
    return dataSplit;
}

HcclResult CollReduceScatterDoubleRingConcurrentExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun] The ReduceScatterDoubleRingConcurrentExecutor"
        "starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移

    u32 sliceNum = outerCommInfo.localRankSize;
    Slice sliceTemp;
    u32 commIndex = outerCommInfo.localRank;
    commIndex = RefreshCommIdx(commIndex, topoAttr_.nicList, topoAttr_.devicePhyId);

    /* 超节点间通信域是commLevel2 */
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    u32 level2RankSize = level2CommInfo.localRankSize;

    if (level2RankSize > 1) {
        /* ****************** 超节点间 reducescatter *******************************/
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> level2Executor;

        if (UseLevel2RingAlgo(algType_)) {
            level2Executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using ring algo inter-superPod.");
            CHK_SMART_PTR_NULL(level2Executor);

            u64 ringCount = execMem.inputMem.size() / (level2RankSize * perDataSize);
            CHK_RET(level2Executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else {
            level2Executor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using halving-doubling algo inter-superPod.");

            CHK_SMART_PTR_NULL(level2Executor);
            u64 inputDataCount = execMem.inputMem.size() / perDataSize; // count是output的数据个数
            CHK_RET(level2Executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, inputDataCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        }
        CHK_RET(level2Executor->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2Executor, level2CommInfo));

        /* ****************** 节点间 reducescatter *******************************/
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        u32 innerRankSize = innerCommInfo.localRankSize;

        if (innerRankSize > 1) {
            std::unique_ptr<ExecutorBase> innerExecutor;
            u32 level1Index = innerCommInfo.localRank;

            if (UseInterServerRingAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = execMem.inputMem.size() / (innerRankSize * level2RankSize);
                u64 ringCount = ringSize / perDataSize;
                u64 level1SliceOffset = ringSize * level1Index;
                DeviceMem level1InputMem = execMem.inputMem.range(level1SliceOffset, ringSize);
                CHK_SMART_PTR_NULL(level1InputMem.ptr());

                CHK_RET(innerExecutor->Prepare(level1InputMem, level1InputMem, execMem.scratchMem, ringCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0),
                    level1SliceOffset));
            } else {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");

                CHK_SMART_PTR_NULL(innerExecutor);
                u64 inputDataCount = execMem.inputMem.size() / (perDataSize * level2RankSize);
                u64 level1SliceSize = execMem.inputMem.size() / level2RankSize;
                u64 level1SliceOffset = level1SliceSize * level1Index;

                DeviceMem level1InputMem = execMem.inputMem.range(level1SliceOffset, level1SliceSize);
                // count是output的数据个数
                CHK_RET(innerExecutor->Prepare(level1InputMem, level1InputMem, execMem.scratchMem, inputDataCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0),
                    level1SliceOffset));
            }
            CHK_RET(innerExecutor->RegisterProfiler(
                (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
        }

        /* *********** 节点内reducescatter (正常场景) *****************************/
        CHK_RET(ActiveSlaveStreams(param.stream));

        bool useInlineRduce = false;
        bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(),
            param.DataDes.dataType, param.reduceType);
        useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
        multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem,
            dataSegsSlice, param.tag);
        bool bRet = (multiStreamSlice.size() != ringNum);
        CHK_PRT_RET(bRet,
            HCCL_ERROR("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun]sliceNum-1[%u] != multiStreamSlice" \
            "size[%llu]", sliceNum - 1, multiStreamSlice.size()), HCCL_E_INTERNAL);

        DeviceMem srcMem;
        // 每个server分配的slice大小
        u64 serverSliceSize = execMem.inputMem.size() / (innerRankSize * level2RankSize);
        // 每个服务器对应的偏移
        u32 serverIndex = innerCommInfo.localRank;
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        HCCL_DEBUG("inputMem.size=%llu, outerCommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
            "commIndex=%u innerCommInfo.localRank=%u", execMem.inputMem.size(), outerCommInfo.localRankSize,
            serverSliceSize, serverSliceOffset, commIndex, innerCommInfo.localRank);
        DeviceMem reduceScatterRingInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
        DeviceMem reduceScatterRingOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);

        u64 countLocal = serverSliceSize / perDataSize;
        CHK_RET(MultiRingReduceScatter(param.tag, reduceScatterRingInput, reduceScatterRingOutput, countLocal,
            param.DataDes.dataType, param.reduceType, multiStreamSlice, param.stream, PROF_STAGE_1, serverSliceOffset));

        srcMem = execMem.inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset,
            execMem.count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
        HCCL_INFO("reducescatter double ring concurrent run success");
        return HCCL_SUCCESS;
    }

    /* ****************** 节点间 reducescatter *******************************/
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 innerRankSize = innerCommInfo.localRankSize;
    if (innerRankSize > 1) {
        std::vector<Slice> innerDataSegsSlice; // 节点间数据分成ranksize份，每份的起始偏移和大小
        // 基于2环数据切分2环SDMA+2环ROH; bool = true表示SDMA
        std::vector<std::pair<bool, std::vector<Slice>>> innerMultSlice;
        innerDataSegsSlice.resize(innerRankSize);
        for (u32 i = 0; i < innerRankSize; i++) {
            innerDataSegsSlice[i].size = execMem.inputMem.size() / innerRankSize;
            innerDataSegsSlice[i].offset = (i * execMem.inputMem.size() / innerRankSize);
        }

        u32 syncTrans = BEST_SPLIT_VALUE;
        u64 totalDataSize = execMem.inputMem.size() / innerRankSize;
        if (totalDataSize <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
            syncTrans = MAX_SPLIT_VALUE;
        }
        // 把innerDataSegsSlice的一份数据分成 SDMA+RDMA
        innerMultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);
        std::vector<Slice> sdmaSlice;
        std::vector<Slice> rdmaSlice;
        for (u32 segsIndex = 0; segsIndex < innerDataSegsSlice.size(); segsIndex++) {
            auto totalSize = innerDataSegsSlice[segsIndex].size;
            auto sdmaSliceOffset = innerDataSegsSlice[segsIndex].offset;
            auto sdmaSliceSize = (totalSize <= HCCL_MIN_SLICE_ALIGN_910_73) ? totalSize:
                ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_73) * HCCL_MIN_SLICE_ALIGN_910_73;
            Slice sdmaSliceTmp;
            sdmaSliceTmp.offset = sdmaSliceOffset;
            sdmaSliceTmp.size = sdmaSliceSize;
            Slice rdmaSliceTmp;
            rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
            rdmaSliceTmp.size = totalSize - sdmaSliceSize;
            sdmaSlice.push_back(sdmaSliceTmp);
            rdmaSlice.push_back(rdmaSliceTmp);
            HCCL_DEBUG("Inner data segId:%u, Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "\
                "rdma [offset %llu, size %llu]", segsIndex, sdmaSliceOffset, totalSize,
                sdmaSliceTmp.offset, sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
        }
        innerMultSlice[0] = std::make_pair(true, sdmaSlice); // true表示使用sdma
        innerMultSlice[1] = std::make_pair(false, rdmaSlice); // false表示rdma
        if (syncTrans == MAX_SPLIT_VALUE) {
            innerMultSlice.erase(innerMultSlice.end() - 1, innerMultSlice.end());
        }

        u32 commPlaneNum = innerMultSlice.size();
        std::vector<std::vector<u32>> ringNics;
        CHK_RET(GetRingNics(param.tag, ringNics));
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> innerExecutor;
        HcclResult ret = HCCL_SUCCESS;
        // 节点间共2个通信域，分别走SDMA和RDMA
        for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
            std::vector<Slice> singleSlice = innerMultSlice[planeIndex].second;
            CHK_PRT_RET(singleSlice.empty(),
                HCCL_ERROR("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun]singleSlice is empty"),
                HCCL_E_INTERNAL);

            CHK_RET(CheckCommSize(COMM_LEVEL1_RDMA, commIndex + 1));
            SubCommInfo innerRdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_RDMA, commIndex);
            SubCommInfo ringCommInfo = innerMultSlice[planeIndex].first ? innerCommInfo : innerRdmaCommInfo;

            if (planeIndex != (commPlaneNum - 1)) {  // 0~ringNum-2的环
                ret = streamInfo_.ringSignalAux[planeIndex]->Wait(
                    streamInfo_.ringStreams[planeIndex], dispatcher_, PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun]stream[%u] wait failed",
                    planeIndex), ret);

                if (UseInterServerRingAlgo(algType_)) {
                    innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                    HCCL_INFO("reducescatter ring: using ring algo inter-server.");
                    CHK_SMART_PTR_NULL(innerExecutor);

                    u64 ringSize = execMem.inputMem.size() / innerRankSize;
                    u64 ringCount = ringSize / perDataSize;

                    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                        param.DataDes.dataType, streamInfo_.ringStreams[planeIndex], param.reduceType,
                        OUTER_BRIDGE_RANK_ID, singleSlice));
                } else if (UseInterServerNHRAlgo(algType_)) {
                    innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
                    HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-server.");
                    CHK_SMART_PTR_NULL(innerExecutor);

                    u64 ringSize = execMem.inputMem.size() / innerRankSize;
                    u64 ringCount = ringSize / perDataSize;
                    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                        param.DataDes.dataType, streamInfo_.ringStreams[planeIndex], param.reduceType,
                        OUTER_BRIDGE_RANK_ID, singleSlice));
                } else if (UseInterServerNBAlgo(algType_)) {
                    innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
                    HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
                    CHK_SMART_PTR_NULL(innerExecutor);

                    u64 ringSize = execMem.inputMem.size() / innerRankSize;
                    u64 ringCount = ringSize / perDataSize;
                    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                        param.DataDes.dataType, streamInfo_.ringStreams[planeIndex], param.reduceType,
                        OUTER_BRIDGE_RANK_ID, singleSlice));
                } else {
                    innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_,
                                                                                                 reduceAttr));
                    HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");

                    CHK_SMART_PTR_NULL(innerExecutor);
                    u64 inputDataCount = execMem.inputMem.size() / perDataSize;
                    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem,
                        inputDataCount, param.DataDes.dataType, streamInfo_.ringStreams[planeIndex], param.reduceType,
                        OUTER_BRIDGE_RANK_ID, singleSlice));
                }
                CHK_RET(innerExecutor->RegisterProfiler(
                    (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + ringCommInfo.localRank,
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

                CHK_RET(RunTemplate(innerExecutor, ringCommInfo));

                ret = streamInfo_.ringSignal[planeIndex]->Post(
                    streamInfo_.ringStreams[planeIndex], dispatcher_, PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun]stream[%u] record failed",
                        planeIndex), ret);

                /* 主环record启动从环 */
                ret = streamInfo_.ringSignalAux[planeIndex]->Post(const_cast<Stream&>(param.stream), dispatcher_,
                    PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun]stream[%u] record failed",
                        planeIndex), ret);
            } else {
                if (UseInterServerRingAlgo(algType_)) {
                    innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                    HCCL_INFO("reducescatter ring: using ring algo inter-server.");
                    CHK_SMART_PTR_NULL(innerExecutor);

                    u64 ringSize = execMem.inputMem.size() / innerRankSize;
                    u64 ringCount = ringSize / perDataSize;

                    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                        param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, singleSlice));
                } else if (UseInterServerNHRAlgo(algType_)) {
                    innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
                    HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-server.");
                    CHK_SMART_PTR_NULL(innerExecutor);

                    u64 ringSize = execMem.inputMem.size() / innerRankSize;
                    u64 ringCount = ringSize / perDataSize;
                    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                        param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, singleSlice));
                } else if (UseInterServerNBAlgo(algType_)) {
                    innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
                    HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
                    CHK_SMART_PTR_NULL(innerExecutor);

                    u64 ringSize = execMem.inputMem.size() / innerRankSize;
                    u64 ringCount = ringSize / perDataSize;
                    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                        param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, singleSlice));
                } else {
                    innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_,
                                                                                                 reduceAttr));
                    HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");

                    CHK_SMART_PTR_NULL(innerExecutor);
                    u64 inputDataCount = execMem.inputMem.size() / perDataSize;
                    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem,
                        inputDataCount, param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
                        singleSlice)); // count是output的数据个数
                }
                CHK_RET(innerExecutor->RegisterProfiler(
                    (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
                CHK_RET(RunTemplate(innerExecutor, ringCommInfo));

                for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                    /* 等待executor执行完毕 */
                    ret = streamInfo_.ringSignal[ring]->Wait(const_cast<Stream&>(param.stream), dispatcher_,
                        PROF_STAGE_0);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun]stream[%u] wait failed",
                            ring), ret);
                }
            }
        }
        CHK_RET(ExecutorBase::ExecEmptyTask(execMem.inputMem, execMem.outputMem, const_cast<Stream&>(param.stream),
            dispatcher_));
    }
    /* *********** 节点内reducescatter (正常场景) *****************************/
    std::vector<std::pair<bool, std::vector<Slice>>> mult4RingsSlice; // 基于2环数据切分2环SDMA+2环ROH bool = true表示SDMA
    CHK_RET(ActiveSlaveStreams(param.stream));

    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(), param.DataDes.dataType,
        param.reduceType);
    useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem,
        dataSegsSlice, param.tag);
    bool bRet = (multiStreamSlice.size() != ringNum);
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[CollReduceScatterDoubleRingConcurrentExecutor][KernelRun]sliceNum-1[%u] != multiStreamSlice "
            "size[%llu]", sliceNum - 1, multiStreamSlice.size()), HCCL_E_INTERNAL);
    u32 syncTrans = BEST_SPLIT_VALUE;
    u64 totalDataSize = execMem.outputMem.size() * dataSegsSlice.size();
    if (totalDataSize <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
        syncTrans = MAX_SPLIT_VALUE;
    }
    mult4RingsSlice.resize(multiStreamSlice.size() * SLICES_FACTOR);
    for (u32 ringIndex = 0; ringIndex < multiStreamSlice.size(); ringIndex++) {
        std::vector<Slice> sdmaSlice;
        std::vector<Slice> rdmaSlice;
        for (u32 segsIndex = 0; segsIndex < multiStreamSlice[ringIndex].size(); segsIndex++) {
            auto totalSize = multiStreamSlice[ringIndex][segsIndex].size;
            auto sdmaSliceOffset = multiStreamSlice[ringIndex][segsIndex].offset;
            auto sdmaSliceSize = (totalSize <= HCCL_MIN_SLICE_ALIGN_910_73) ? totalSize:
                ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_73) * HCCL_MIN_SLICE_ALIGN_910_73;
            Slice sdmaSliceTmp;
            sdmaSliceTmp.offset = sdmaSliceOffset;
            sdmaSliceTmp.size = sdmaSliceSize;
            Slice rdmaSliceTmp;
            rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
            rdmaSliceTmp.size = totalSize - sdmaSliceSize;
            sdmaSlice.push_back(sdmaSliceTmp);
            rdmaSlice.push_back(rdmaSliceTmp);
            HCCL_DEBUG("Intra index:%u, segId:%u, Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "\
                "rdma [offset %llu, size %llu]", ringIndex, segsIndex, sdmaSliceOffset, totalSize,
                sdmaSliceTmp.offset, sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
        }
        mult4RingsSlice[ringIndex] = std::make_pair(true, sdmaSlice); // true表示使用sdma
        mult4RingsSlice[ringIndex + multiStreamSlice.size()] = std::make_pair(false, rdmaSlice); // false表示rdma
    }
    if (syncTrans == MAX_SPLIT_VALUE) {
        mult4RingsSlice.erase(mult4RingsSlice.end() - multiStreamSlice.size(), mult4RingsSlice.end());
    }

    DeviceMem srcMem;
    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size() / innerRankSize;
    // 每个服务器对应的偏移
    u32 serverIndex = innerCommInfo.localRank;
    u64 serverSliceOffset = serverSliceSize * serverIndex;
    HCCL_DEBUG("inputMem.size=%llu, outerCommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
        "commIndex=%u innerCommInfo.localRank=%u", execMem.inputMem.size(), outerCommInfo.localRankSize,
        serverSliceSize, serverSliceOffset, commIndex, innerCommInfo.localRank);
    DeviceMem reduceScatterRingInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterRingInput.ptr());
    DeviceMem reduceScatterRingOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterRingOutput.ptr());
    u64 countLocal = serverSliceSize / perDataSize;
    CHK_RET(MultiRingReduceScatterConcurrent(param.tag, reduceScatterRingInput, reduceScatterRingOutput, countLocal,
        param.DataDes.dataType, param.reduceType, mult4RingsSlice, param.stream, PROF_STAGE_1, serverSliceOffset,
        nullptr));

    srcMem = execMem.inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset, execMem.count * perDataSize);
    CHK_SMART_PTR_NULL(srcMem.ptr());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));

    HCCL_INFO("reducescatter double ring concurrent run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterDoubleRingConcurrentExecutor", ReduceScatterDoubleRingConcurrent,
    CollReduceScatterDoubleRingConcurrentExecutor);
}