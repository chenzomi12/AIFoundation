/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_reduce_double_ring_executor.h"

namespace hccl {

CollReduceDoubleRingExecutor::CollReduceDoubleRingExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceDoubleRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    // DoubleRing只支持910_73场景
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else {
        totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllReduceDoubleRingExecutor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceDoubleRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CalcLevel0CommInfo(inputType, outputType, opTransport);
    CalcLevel1CommInfo(inputType, outputType, opTransport);
    CalcLevel2CommInfo(inputType, outputType, opTransport);
    return HCCL_SUCCESS;
}

HcclResult CollReduceDoubleRingExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollReduceDoubleRingExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceDoubleRingExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollReduceDoubleRingExecutor][CalcOuterCommInfo]tag[%s ]start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollReduceDoubleRingExecutor][CalcOuterCommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}
 
HcclResult CollReduceDoubleRingExecutor::CalcLevel2CommInfo(TransportMemType inputType,
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

HcclResult CollReduceDoubleRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceDoubleRingExecutor][Run]The CollReduceDoubleRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiRingsSliceZero; // 数据基于该rank上环0的偏移
    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 sliceNum = outerCommInfo.localRankSize;
    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 三步算法step1：外层 - 节点内 reduce-scatter */
    // 构造ring algorithm对应的reduce-scatter实例
    multiRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag_, false, topoAttr_.nicList);
    CHK_PRT_RET(multiRingsSliceZero.size() != ringNum, HCCL_ERROR("[CollReduceDoubleRingExecutor][Run]"\
        "ringNum[%u] != multiRingsSliceZero size[%llu]", ringNum, multiRingsSliceZero.size()),
        HCCL_E_INTERNAL);

    HcomCollOpInfo *reduceScatterOpInfoPtr = nullptr;
    // 第一步的reducescatter输出放在CCL buffer上，通过设置nullptr指示不做最后一步的DMA削减动作

    CHK_RET(MultiRingReduceScatter(tag_, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.reduceType, multiRingsSliceZero, param.stream,
        PROF_STAGE_0, 0, reduceScatterOpInfoPtr));
    HCCL_INFO("reduce double ring stage0 run success");

    // step2: 节点间的reduce
    u32 commIndex = 0;
    u64 level1Size = 0;
    u32 segmentIdx = 0;
    CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, level1Size, outerCommInfo, multiRingsSliceZero, tag_));
    u64 level1Count = level1Size / perDataSize;
    if (topoAttr_.devNumInLevel2 <= 1) {
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        DeviceMem reduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, level1Size);
        CHK_SMART_PTR_NULL(reduceInput);
        DeviceMem reduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, level1Size);
        CHK_SMART_PTR_NULL(reduceOutput);
        u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, param.DataDes.dataType,  param.reduceType);
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("[CollReduceDoubleRingExecutor]using ring algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("[CollReduceDoubleRingExecutor]using Recursive halving-doubling algo inter-server.");
        }
        u32 rankSize = innerCommInfo.localRankSize;
        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[CollReduceDoubleRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, topoAttr_.userRank, param.root), HCCL_E_INTERNAL);
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));
        // 节点间的hd 使用环0来记录
        CHK_SMART_PTR_NULL(innerExecutor);
        CHK_RET(innerExecutor->Prepare(reduceInput, reduceOutput, reduceOutput, level1Count, param.DataDes.dataType,
            param.stream, param.reduceType, planeRoot, std::vector<Slice>(0),
            dataSegsSlice[segmentIdx].offset));
        CHK_RET(innerExecutor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
    } else {
        //节点间 reduce scatter
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        SubCommInfo innerZeroCommInfo = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0);
        sliceNum = innerZeroCommInfo.localRankSize;
        CHK_RET(ExecutorBase::PrepareSliceData(level1Count, perDataSize, sliceNum, 0, dataSegsSlice));

        DeviceMem reducescatterInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, level1Size);
        DeviceMem reducescatterOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, level1Size);

        u64 reduceAttr = GetReduceAttr(reducescatterInput, reducescatterOutput, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> level1RSExecutor;
        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));
        if (UseInterServerRingAlgo(algType_)) {
            level1RSExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(level1RSExecutor);
            CHK_RET(level1RSExecutor->Prepare(
                reducescatterInput, reducescatterInput, reducescatterOutput, level1Count, param.DataDes.dataType, param.stream, param.reduceType,
                planeRoot, dataSegsSlice, dataSegsSlice[segmentIdx].offset));
            HCCL_INFO("[CollReduceDoubleRingExecutor]reducescatter ring: using ring algo inter-server.");
        } else {
            level1RSExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(level1RSExecutor);
            CHK_RET(level1RSExecutor->Prepare(
                reducescatterInput, reducescatterOutput, reducescatterOutput, level1Count, param.DataDes.dataType, param.stream, param.reduceType,
                planeRoot, dataSegsSlice, dataSegsSlice[segmentIdx].offset));
            HCCL_INFO("[CollReduceDoubleRingExecutor]reducescatter ring: using halving-doubling algo inter-server.");
        }
        CHK_RET(level1RSExecutor->RegisterProfiler(
            (sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1RSExecutor, innerCommInfo));
        HCCL_INFO("[CollReduceDoubleRingExecutor]reduce double ring [superpod] level1 reduce-scatter run success");

        // 超节点 reduce
        u64 rSize;
        std::vector<std::vector<Slice>> rdSlice;
        rdSlice.push_back(dataSegsSlice);
        CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, rSize, innerZeroCommInfo, rdSlice, tag_));
        u64 arCount = rSize / perDataSize;

        CHK_RET(CheckCommSize(COMM_LEVEL2, commIndex + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, commIndex);
        u32 rankSize = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0).localRankSize;

        u32 subUserrankRootSupperPod = topoMatcher_->GetSubRootUserRankWithSuperPod(topoAttr_.userRank, param.root);
        u32 planeRootSupperPod = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL2, commIndex, subUserrankRootSupperPod, planeRootSupperPod));

        DeviceMem reduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, rSize);
        DeviceMem reduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, rSize);

        reduceAttr = GetReduceAttr(reduceInput, reduceOutput, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> level2RExecutor;
        if (UseLevel2RingAlgo(algType_)) {
            level2RExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("[CollReduceDoubleRingExecutor]reducescatter ring: using ring algo inter-server.");
        } else {
            level2RExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("[CollReduceDoubleRingExecutor]reducescatter ring: using halving-doubling algo inter-server.");
        }

        CHK_RET(level2RExecutor->Prepare(
            reduceInput, reduceOutput, reduceOutput, arCount, param.DataDes.dataType, param.stream, param.reduceType, planeRootSupperPod,
            std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
        CHK_SMART_PTR_NULL(level2RExecutor);
        CHK_RET(level2RExecutor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2RExecutor, level2CommInfo));
        HCCL_INFO("[CollReduceDoubleRingExecutor]reduce double ring [superpod] level2 reduce run success");
        // 节点间 gather
        std::unique_ptr<ExecutorBase> level1GExecutor;
        DeviceMem gatherInput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, rSize);
        DeviceMem gatherOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, rSize*sliceNum);
        level1GExecutor.reset(new (std::nothrow) GatherRing(dispatcher_));
        CHK_SMART_PTR_NULL(level1GExecutor);
        CHK_RET(level1GExecutor->Prepare(gatherOutput, gatherOutput, gatherOutput, arCount, param.DataDes.dataType, param.stream,
            HcclReduceOp::HCCL_REDUCE_RESERVED, planeRoot, dataSegsSlice,
            dataSegsSlice[segmentIdx].offset));
        CHK_RET(level1GExecutor->RegisterProfiler(
            (sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1GExecutor, innerCommInfo));
        HCCL_INFO("[CollReduceDoubleRingExecutor]reduce double ring [superpod] level1 gather run success");
    }
    HCCL_INFO("[CollReduceDoubleRingExecutor]stage1 run success");

    // step3: 节点内的gatherring，只有在root所在server内进行gather操作
    SingleSubCommTransport &outerTransportInfo =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_LEVEL0][COMM_INDEX_0]);

    if (outerTransportInfo.userRank2subCommRank.find(param.root) !=
        outerTransportInfo.userRank2subCommRank.end()) {
        CHK_RET(MultiRingGather(tag_, execMem.outputMem, execMem.outputMem, level1Count, param.DataDes.dataType,
            multiRingsSliceZero, param.reduceType, param.root, const_cast<Stream &>(param.stream), PROF_STAGE_2));
    }
    HCCL_INFO("[CollReduceDoubleRingExecutor]reduce double ring stage2 run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceDoubleRingExecutor", ReduceDoubleRing, CollReduceDoubleRingExecutor);

} // namespace hccl