/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_ring_for_910_73_executor.h"

namespace hccl {

CollReduceScatterRingFor91073Executor::CollReduceScatterRingFor91073Executor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

void CollReduceScatterRingFor91073Executor::ParseParam(const OpParam& param)
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

HcclResult CollReduceScatterRingFor91073Executor::CalcScratchMemSize(u64& scratchMemSize)
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
    HCCL_INFO("[CollReduceScatterRingFor91073Executor][CalcScratchMemSize] tag[%s] scratchMemSize[%u]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91073Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? OUTER_PLANE_NUM_IN_NPRING_DOUBLE :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE);
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollReduceScatterRingFor91073Executor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91073Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91073Executor::CalcTransportMemType(TransportMemType &inputType,
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
    HCCL_INFO("[CollReduceScatterRingFor91073Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91073Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91073Executor::CalcLevel2CommInfo(TransportMemType inputType,
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

u64 CollReduceScatterRingFor91073Executor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterRingFor91073Executor::IsHugeData(const u64 curSize)
{
    bool hugeData;
    if (DMAReduceFlag_) {
        hugeData = curSize > SDMA_SEND_MAX_SIZE;
    } else {
        hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                   (curSize > SDMA_SEND_MAX_SIZE);
    }

    return hugeData;
}

HcclResult CollReduceScatterRingFor91073Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceScatterRingFor91073Executor][KernelRun] The ReduceScatterDoubleRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 ringNum;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    } else {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
    }

    u32 sliceNum = outerCommInfo.localRankSize;
    Slice sliceTemp;
    u32 commIndex = outerCommInfo.localRank;
    commIndex = RefreshCommIdx(commIndex, topoAttr_.nicList, topoAttr_.devicePhyId);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    u32 level2RankSize = level2CommInfo.localRankSize;

    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    /* 超节点间通信域是commLevel2 */
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
            HCCL_ERROR("[CollReduceScatterRingFor91073Executor][KernelRun]sliceNum-1[%u] != multiStreamSlice" \
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
        HCCL_INFO("reducescatter double ring run success");
        return HCCL_SUCCESS;
    }

    // 节点内reduce scatter
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 innerRankSize = innerCommInfo.localRankSize;

    // 计算slice
    std::vector<std::vector<Slice> > level0DataSegsSlice;
    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(), param.DataDes.dataType,
        param.reduceType);
    useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem,
        dataSegsSlice, param.tag);
    for (u32 ringIndex = 0; ringIndex < multiStreamSlice.size(); ringIndex++) {
        std::vector<Slice> dataSlice;
        for (u32 level0Idx = 0; level0Idx < sliceNum; level0Idx++) {
            Slice sliceTemp;
            for (u32 level1Idx = 0; level1Idx < innerRankSize; level1Idx++) {
                sliceTemp.size = multiStreamSlice[ringIndex][level0Idx].size;
                sliceTemp.offset =
                    multiStreamSlice[ringIndex][level0Idx].offset + level1Idx * sliceNum * execMem.outputMem.size();
                dataSlice.push_back(sliceTemp);
            }
        }
        level0DataSegsSlice.push_back(dataSlice);
    }
    std::vector<std::vector<Slice>> multRingsUserMemSlice;

    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType};
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    if (opInfoPtr == nullptr) {
        multRingsUserMemSlice = level0DataSegsSlice;
    } else {
        for (u32 ringIndex = 0; ringIndex < level0DataSegsSlice.size(); ringIndex++) {
            std::vector<Slice> level1UserMemSlice;
            for (auto &cclSlice : level0DataSegsSlice[ringIndex]) {
                Slice tmpSlice;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset =
                    (cclSlice.offset / execMem.outputMem.size()) * param.DataDes.count * perDataSize +
                    multiStreamSlice[ringIndex][0].offset;
                level1UserMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(level1UserMemSlice);
        }
    }
    // 区分消减拷贝场景
    if (opInfoPtr != nullptr && innerRankSize > 1) {
        HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfoPtr;
        opInfoByReduceScatterDMAreduce.outputAddr      = nullptr;
        CHK_RET(MultiRingReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType, level0DataSegsSlice,
            param.stream, PROF_STAGE_1, 0, &opInfoByReduceScatterDMAreduce, multRingsUserMemSlice));
    } else {
        CHK_RET(MultiRingReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType,
            level0DataSegsSlice, param.stream, PROF_STAGE_1, 0, opInfoPtr, multRingsUserMemSlice));
    }
    // 对于单server图模式场景最后一步需要把数据从ccl input拷贝到ccl output上
    if (innerRankSize == 1 && opInfoPtr == nullptr) {
        DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(),
            execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    if  (innerRankSize > 1) {
        // 节点间做reduce scatter（ring/NHR)
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> innerExecutor;

        // 计算slice
        u32 level0ServerIndex = 0;
        HcclResult ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, topoAttr_.userRank, level0ServerIndex);

        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollReduceScatterRingFor91073Executor][KernelRun]Get Rank[%u] "
            "by User Rank[%u] from CommOuter[%u] Failed!", level0ServerIndex, topoAttr_.userRank, commIndex), ret);

        std::vector<Slice> level1DataSegsSlice;
        for (u32 i = 0; i < innerRankSize; i++) {
            sliceTemp.size = execMem.outputMem.size();
            u32 level1UserRank;
            CHK_RET(GetUserRankByRank(COMM_LEVEL1, commIndex, i, level1UserRank));
            sliceTemp.offset = level1UserRank * execMem.outputMem.size();
            level1DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG("rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i,
                sliceTemp.offset, sliceTemp.size);
        }
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using ring algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);

        CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, level1DataSegsSlice));
        CHK_RET(innerExecutor->RegisterProfiler(
            (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

        // 区分消减拷贝场景（消减拷贝数据需要拷贝到user output上）
        DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(),
            execMem.outputMem.size());
        if (opInfoPtr != nullptr) {
            DeviceMem dstMem = DeviceMem::create(static_cast<u8 *>(opInfoPtr->outputAddr), execMem.outputMem.size());
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }

    HCCL_INFO("reducescatter double ring run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterRingFor91073Executor", ReduceScatterRingFor91073, CollReduceScatterRingFor91073Executor);
}