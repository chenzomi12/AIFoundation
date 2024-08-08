/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_double_ring_concurrent_executor.h"

namespace hccl {

CollAllGatherDoubleRingConcurrentExecutor::CollAllGatherDoubleRingConcurrentExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcStreamNum(u32& streamNum)
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
    HCCL_INFO("[CollAllGatherDoubleRingConcurrentExecutor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherDoubleRingConcurrentExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcLevel0CommInfo(TransportMemType inputType,
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

HcclResult CollAllGatherDoubleRingConcurrentExecutor::CalcLevel2CommInfo(TransportMemType inputType,
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
u64 CollAllGatherDoubleRingConcurrentExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

u32 CollAllGatherDoubleRingConcurrentExecutor::IsDataSplit(const u64 curSize)
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

HcclResult CollAllGatherDoubleRingConcurrentExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun]AllGatherDoubleRingConcurrentExecutor starts.");

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun]errNo[0x%016llx] datatype[%d] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), param.DataDes.dataType), HCCL_E_PARA);

    // 获取子通信域信息
    auto nicList = topoAttr_.nicList;
    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 outerRankSize = outerCommInfo.localRankSize;
    u32 commIndex = outerCommInfo.localRank;
    commIndex = RefreshCommIdx(commIndex, nicList, topoAttr_.devicePhyId);
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    u32 serverIndex = GetSubCommInfo(COMM_LEVEL1, commIndex).localRank;

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * outerRankSize;
    u64 outerOffset = commIndex * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(baseOffset + outerOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };
    HcomCollOpInfo *opInfoPtr = nullptr;

    if (!DMAReduceFlag_) {
        HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherDoubleRingExecutor][KernelRun]all gather double "
                "ring memcpy Failed, Offset[%llu], Size[%llu]",
            baseOffset + outerOffset, inputMemSize), ret);
    } else {
        opInfoPtr = &opInfo;
    }

    // 第二步，各个AI Server 内 multi ring all gather
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    u32 sliceNum = outerRankSize;
    CHK_RET(PrepareAllgatherSlice(sliceNum, inputMemSize, dataSegsSlice));

    //  多环数据切分
    auto mult2RingsSlice = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, nicList);
    std::vector<std::pair<bool, std::vector<Slice>>> mult4RingsSlice;
    // 基于2环数据切分2环SDMA+2环ROH; bool = true表示SDMA;
    u32 syncTrans = BEST_SPLIT_VALUE;
    u64 totalDataSize = inputMemSize * dataSegsSlice.size();
    if (totalDataSize <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
        syncTrans = MAX_SPLIT_VALUE;
    }
    mult4RingsSlice.resize(mult2RingsSlice.size() * SLICES_FACTOR);
    for (u32 ringIndex = 0; ringIndex < mult2RingsSlice.size(); ringIndex++) {
        std::vector<Slice> sdmaSlice;
        std::vector<Slice> rdmaSlice;
        for (u32 segsIndex = 0; segsIndex < mult2RingsSlice[ringIndex].size(); segsIndex++) {
            auto totalSize = mult2RingsSlice[ringIndex][segsIndex].size;
            auto sdmaSliceOffset = mult2RingsSlice[ringIndex][segsIndex].offset;
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
            HCCL_DEBUG("Ring index:%u, segId:%u, Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "\
                "rdma [offset %llu, size %llu]", ringIndex, segsIndex, sdmaSliceOffset, totalSize,
                sdmaSliceTmp.offset, sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
        }
        mult4RingsSlice[ringIndex] = std::make_pair(true, sdmaSlice); // true表示使用sdma
        mult4RingsSlice[ringIndex + mult2RingsSlice.size()] = std::make_pair(false, rdmaSlice); // false表示rdma
    }
    if (syncTrans == MAX_SPLIT_VALUE) {
        mult4RingsSlice.erase(mult4RingsSlice.end() - mult2RingsSlice.size(), mult4RingsSlice.end());
    }
    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem.range(baseOffset, inputMemSize * outerRankSize);
    CHK_SMART_PTR_NULL(currentOutputMem);
    CHK_RET(ActiveSlaveStreams(param.stream));

    CHK_RET(MultiRingAllGatherConcurrent(param.tag, execMem.inputMem, currentOutputMem, execMem.count, param.DataDes.dataType,
                                         mult4RingsSlice, param.stream, PROF_STAGE_1, baseOffset, opInfoPtr));

    HCCL_INFO("all gather double ring outer run success");

    //  第三步， AI server 间 recursive halving doubling all gather
    u64 hdSize = 0;
    std::vector<u32>::iterator iterNic = std::find(nicList.begin(), nicList.end(), topoAttr_.devicePhyId);
    if (iterNic != nicList.end()) {
        hdSize = inputMemSize * outerRankSize;
    }

    u64 hdCount = hdSize / perDataSize;
    std::unique_ptr<ExecutorBase> innerExecutor;
    u64 firstCommInnerSize = ((syncTrans * hdSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_73) *
                             HCCL_MIN_SLICE_ALIGN_910_73;
    std::vector<u64> sendSize{firstCommInnerSize, hdSize - firstCommInnerSize};
    std::vector<u64> sendOffset{0, firstCommInnerSize};
    for (int innerCommIndex = 0; innerCommIndex < RDMA_PLANE_NUM_IN_NPRING_DOUBLE; ++innerCommIndex) {
        if (sendSize[innerCommIndex] == 0 || (!GetExternalInputEnableRdmaSdmaConcurrent() && innerCommIndex > 0)) {
            continue;
        }
        if (GetExternalInputEnableRdmaSdmaConcurrent() || UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        } else if (UseInterServerNHRAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
            HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("allgather ring: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);

        CHK_RET(CheckCommSize(COMM_LEVEL1_RDMA, commIndex + 1));
        SubCommInfo innerCommInfo = (innerCommIndex == 0 ?
            GetSubCommInfo(COMM_LEVEL1, commIndex) : GetSubCommInfo(COMM_LEVEL1_RDMA, commIndex));

        if (topoAttr_.devNumInLevel2 <= 1) {
            //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
            u32 rankSize = innerCommInfo.localRankSize;
            std::vector<Slice> inputSlices(rankSize, Slice());
            for (u32 i = 0; i < rankSize; i++) {
                inputSlices[i].size = sendSize[innerCommIndex];
                inputSlices[i].offset = hdSize * i + sendOffset[innerCommIndex];
            }
            auto &innerCommStream = streamInfo_.ringStreams[innerCommIndex];
            auto ret = streamInfo_.ringSignalAux[innerCommIndex]->Wait(innerCommStream, dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun] "\
                " inner wait main [%u] failed", innerCommIndex), ret);

            CHK_RET(innerExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, hdCount, param.DataDes.dataType, innerCommStream,
                HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, inputSlices, 0));

            CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
                innerCommInfo.localRank, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, innerCommStream));

            CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

            ret = streamInfo_.ringSignal[innerCommIndex]->Post(innerCommStream, dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun] "\
                " inner post mains [%u] failed", innerCommIndex), ret);

            ret = streamInfo_.ringSignalAux[innerCommIndex]->Post(const_cast<Stream&>(param.stream),
                dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun] "\
                " main post inner [%u] failed", innerCommIndex), ret);
        } else {
            u32 innerRankSize = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0).localRankSize;
            u64 innerBaseOffset = baseOffset * innerRankSize;
            DeviceMem innerInputMem = execMem.outputMem.range(innerBaseOffset, inputMemSize * outerRankSize);
            DeviceMem innerOutputMem = execMem.outputMem.range(innerBaseOffset,
                inputMemSize * outerRankSize * innerRankSize);

            std::vector<Slice> inputSlices(innerRankSize, Slice());
            for (u32 i = 0; i < innerRankSize; i++) {
                inputSlices[i].size = sendSize[innerCommIndex];
                inputSlices[i].offset = hdSize * i + sendOffset[innerCommIndex];
            }

            auto &innerCommStream = streamInfo_.ringStreams[innerCommIndex];
            auto ret = streamInfo_.ringSignalAux[innerCommIndex]->Wait(innerCommStream, dispatcher_, PROF_STAGE_2);

            //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
            CHK_RET(innerExecutor->Prepare(innerInputMem, innerOutputMem, execMem.inputMem, hdCount,
                param.DataDes.dataType, innerCommStream, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
                inputSlices, 0));

            SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
            u32 rankSize = innerCommInfo.localRankSize;
            CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, innerCommStream));

            CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
            ret = streamInfo_.ringSignal[innerCommIndex]->Post(innerCommStream, dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun] "\
                "inner post mains [%u] failed", innerCommIndex), ret);

            ret = streamInfo_.ringSignalAux[innerCommIndex]->Post(const_cast<Stream&>(param.stream),
                dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun] "\
                "main post inner [%u] failed", innerCommIndex), ret);

             // 超节点间做allgather
            ret = AllGatherLevel2(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
                param.DataDes.dataType, const_cast<Stream&>(param.stream));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun]tag[%s], all_gather failed, "\
                "return[%d]", param.tag.c_str(), ret), ret);
        }
        if (sendSize[innerCommIndex] == 0 || (!GetExternalInputEnableRdmaSdmaConcurrent() && innerCommIndex > 0)) {
            continue;
        }

        auto ret = streamInfo_.ringSignal[innerCommIndex]->Wait(const_cast<Stream&>(param.stream),
            dispatcher_, PROF_STAGE_2);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollAllGatherDoubleRingConcurrentExecutor][KernelRun] "\
            "main wait inner [%u] failed", innerCommIndex), ret);
    }
    HCCL_INFO("all gather double ring inner run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherDoubleRingConcurrentExecutor", AllGatherDoubleRingConcurrent,
    CollAllGatherDoubleRingConcurrentExecutor);

} // namespace hccl