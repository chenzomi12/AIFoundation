/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_double_ring_executor.h"

namespace hccl {

CollBroadcastDoubleRingExecutor::CollBroadcastDoubleRingExecutor(const HcclDispatcher dispatcher,
                                               std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollBroadcastDoubleRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollBroadcastDoubleRingExecutor][CalcStreamNum] tag[%s] streamNum_[%u]",
                tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastDoubleRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastDoubleRingExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastDoubleRingExecutor::CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
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

HcclResult CollBroadcastDoubleRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[BroadCastOperator][BroadCastDoubleRingExecutor] The BroadCastDoubleRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    std::vector<Slice>              dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> mulRingSlice;  // 数据基于该rank上环0的偏移
    // step1: 节点内的scatter
    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 按ranksize得到内存切分slice数
    u32 sliceNum = outerCommInfo.localRankSize;
    // 将根节点数据切分成sliceNum份
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    HCCL_DEBUG("[BroadCastDoubleRingExecutor]: ringNum[%u] sliceNum[%u]", ringNum, sliceNum);

    /* 外层:scatter */
    // 将每slice再切分成2份，按各ring的dev顺序排列
    // 构造ring algorithm对应的reduce-scatter实例
    mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    CHK_PRT_RET(mulRingSlice.size() != ringNum,
                HCCL_ERROR("[BroadCastOperator][BroadCastDoubleRingExecutor]"
                           "ringNum[%u] !=mulRingSlice size[%llu]",
                           ringNum, mulRingSlice.size()), HCCL_E_INTERNAL);

    HcomCollOpInfo *scatterOpInfoPtr = nullptr;
    HcomCollOpInfo scatterOpInfo = {
        "", execMem.inputPtr, nullptr, param.DataDes.count, param.DataDes.dataType, param.root
    };

    if (DMAReduceFlag_) {
        scatterOpInfoPtr = &scatterOpInfo;
    }
    CHK_RET(MultiRingScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
                             mulRingSlice, param.root, param.stream, scatterOpInfoPtr));

    HCCL_INFO("Broadcast double ring stage0 run success");

    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    u32 level2RankSize = level2CommInfo.localRankSize;

    u64 hdCount = 0;
    u64 hdSize  = 0;

    if (topoAttr_.devNumInLevel2 <= 1) {
        HCCL_INFO("Broadcast double ring No level2.");
        // step2: server间的broadcast
        u32 segmentIdx;
        u32 commIndex;
        CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, outerCommInfo, mulRingSlice, param.tag));

        hdCount = hdSize / perDataSize;

        HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, param.tag.c_str(), level2RankSize);

        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        std::unique_ptr<ExecutorBase> innerExecutor;
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        if (UseInterServerNHRAlgo(algType_)) {
            HCCL_DEBUG("broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
                innerExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
            } else {
                innerExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            const u32 innerRankSize = innerCommInfo.localRankSize;
            if (ShouldUseBinaryBroadcastOfNB(curSize / topoAttr_.deviceNumPerAggregation, innerRankSize,
                                             topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation)) {
                innerExecutor.reset(new (std::nothrow) BroadcastNBBinary(dispatcher_));
            } else {
                innerExecutor.reset(new (std::nothrow) BroadcastNB(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("broadcast ring: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);

        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        CHK_PRT_RET(
            subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[HcclImpl][BroadCastDoubleRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                       subUserrankRoot, topoAttr_.userRank, param.root),
            HCCL_E_INTERNAL);
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));
        u32 ranksize = innerCommInfo.localRankSize;
        // 节点间的hd 使用环0来记录
        CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.outputMem, hdCount, param.DataDes.dataType,
                                       param.stream, HCCL_REDUCE_RESERVED, planeRoot, std::vector<Slice>(0),
                                       dataSegsSlice[segmentIdx].offset));

        CHK_RET(innerExecutor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                                                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

        HCCL_INFO("Broadcast double ring stage1 run success");

        // step3: 节点内的allgatherring
        HcomCollOpInfo *allgatherOpInfoPtr = nullptr;
        HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, param.root
        };
        if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
        }
        CHK_RET(MultiRingAllGather(param.tag, execMem.inputMem, execMem.outputMem, hdCount, param.DataDes.dataType,
                                   mulRingSlice, param.stream, PROF_STAGE_2, 0, allgatherOpInfoPtr));

        HCCL_INFO("Broadcast double ring stage2 run success");
    } else {
        HCCL_INFO("Broadcast double ring WITH level2.");
        // step2: 节点间的scatter
        /* count数据准备 */
        std::vector<Slice> level1dataSegsSlice; // 数据分成inner ranksize份，每份的起始偏移和大小
        u32 commIndex = outerCommInfo.localRank;

        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

        // level 1通信数据量
        u64 level1count = execMem.count / sliceNum;

        // 按ranksize得到内存切分slice数
        u32 level1sliceNum = innerCommInfo.localRankSize;
        // 将根节点数据切分成level1sliceNum份
        CHK_RET(ExecutorBase::PrepareSliceData(level1count, perDataSize, level1sliceNum, 0, level1dataSegsSlice));

        u64 level1segmentIdx = innerCommInfo.localRank;

        DeviceMem level1InputMem
            = execMem.inputMem.range(level1dataSegsSlice[level1segmentIdx].offset, (level1count * perDataSize));
        DeviceMem level1OutputMem
            = execMem.outputMem.range(level1dataSegsSlice[level1segmentIdx].offset, (level1count * perDataSize));

        std::unique_ptr<ExecutorBase> level1Executor;
        level1Executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
        CHK_SMART_PTR_NULL(level1Executor);
        CHK_RET(level1Executor->Prepare(level1InputMem, level1InputMem, level1OutputMem, level1count, param.DataDes.dataType, param.stream,
                                        HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID, level1dataSegsSlice,
                                        level1dataSegsSlice[level1segmentIdx].offset));
        CHK_RET(level1Executor->RegisterProfiler((level1sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                                                     + innerCommInfo.localRank,
                                                 PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1Executor, innerCommInfo));
        HCCL_INFO("Broadcast double ring [superpod] level1 run success");

        // step3: 超节点间的broadcast
        u64 level2hdSize;
        u32 level2segmentIdx;
        u32 level2commIndex;
        std::vector<std::vector<Slice>> multiRingSlice;
        multiRingSlice.push_back(level1dataSegsSlice);
        CHK_RET(PrepareInnerCommInfo(level2segmentIdx, level2commIndex, level2hdSize, outerCommInfo,
                                                multiRingSlice, param.tag));

        u64 level2hdCount = level2hdSize / perDataSize;

        CHK_RET(CheckCommSize(COMM_LEVEL2, level2commIndex + 1));

        std::unique_ptr<ExecutorBase> level2Executor;
        if (UseLevel2RingAlgo(algType_)) {
            level2Executor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
            HCCL_INFO("broadcast ring: using ring algo inter-server.");
        } else {
            level2Executor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("broadcast ring: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level2Executor);

        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        CHK_PRT_RET(
            subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[BroadCastOperator][BroadCastDoubleRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],"
                       "root[%u]",
                       subUserrankRoot, topoAttr_.userRank, param.root),
            HCCL_E_INTERNAL);

        u32 planeRoot  = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL2, COMM_INDEX_0, subUserrankRoot, planeRoot));

        u32 ranksize = level2CommInfo.localRankSize;
        // 节点间的hd 使用环0来记录
        CHK_RET(level2Executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.outputMem, level2hdCount, param.DataDes.dataType,
                                        param.stream, HCCL_REDUCE_RESERVED, planeRoot, std::vector<Slice>(0),
                                        level1dataSegsSlice[level2segmentIdx].offset));

        CHK_RET(level2Executor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                                                 PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2Executor, level2CommInfo));

        // step4: 节点间的allgather
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("allgather ring: using halving-doubling algo inter-server.");
        }

        CHK_SMART_PTR_NULL(innerExecutor);
        //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
        CHK_RET(innerExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, level2hdCount,
                                       param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED,
                                       INVALID_VALUE_RANKID, std::vector<Slice>(COMM_INDEX_0), 0));

        u32 rankSize = innerCommInfo.localRankSize;
        CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                                                PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

        // step5: 节点内的allgatherring
        u64 level0count = level2hdCount * rankSize;
        HcomCollOpInfo *allgatherOpInfoPtr = nullptr;
        HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root
        };
        if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
        }
        CHK_RET(MultiRingAllGather(param.tag, execMem.inputMem, execMem.outputMem, level0count, param.DataDes.dataType, mulRingSlice, param.stream,
                                       PROF_STAGE_2, 0, allgatherOpInfoPtr));
        HCCL_INFO("Broadcast[superpod] double ring stage5 run success");
    }

    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadCastDoubleRingExecutor", BroadcastDoubleRing, CollBroadcastDoubleRingExecutor);

} // namespace hccl