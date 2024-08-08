/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_ring_executor.h"

namespace hccl {

CollReduceScatterRingExecutor::CollReduceScatterRingExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        topoAttr_.deviceType == DevType::DEV_TYPE_910_73);
}

void CollReduceScatterRingExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        (topoAttr_.deviceType == DevType::DEV_TYPE_910B || topoAttr_.deviceType == DevType::DEV_TYPE_910_73) &&
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
        IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

HcclResult CollReduceScatterRingExecutor::CalcScratchMemSize(u64& scratchMemSize)
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
    HCCL_INFO("[CollReduceScatterRingExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%u]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 1U;
    switch (algType_) {
        case AlgType::ALG_8P_RING_PLUS_HD:
        case AlgType::ALG_8P_RING_PLUS_RING:
        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_8P_RING_PLUS_NB:
        case AlgType::ALG_8P_RING_PLUS_PIPELINE:
            totalStreamNum = OUTER_PLANE_NUM_IN_8PRING;
            break;
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
            if (topoAttr_.deviceType == DevType::DEV_TYPE_910_73) {
                if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
                } else {
                    totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
                }
            }
            break;
        default:
            break;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollReduceScatterRingExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingExecutor::CalcTransportMemType(TransportMemType &inputType,
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
    HCCL_INFO("[CollReduceScatterRingExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollReduceScatterRingExecutor][CalcLevel0CommInfo]tag[%s ]start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollReduceScatterRingExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

u64 CollReduceScatterRingExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterRingExecutor::IsHugeData(const u64 curSize)
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

HcclResult CollReduceScatterRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceScatterRingExecutor][KernelRun] The ReduceScatterRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? OUTER_PLANE_NUM_IN_8PRING :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE;

    u32 commIndex = (ringNum == OUTER_PLANE_NUM_IN_8PRING) ? topoAttr_.devicePhyId : outerCommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    /* ******************网口裁剪步骤: 节点内allreduce *******************************/
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    u32 sliceNum = outerCommInfo.localRankSize;
    // Slice sliceTemp;
    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && topoAttr_.nicList.size() != DEVICE_EIGHT;
    if (isMultiNic) {
        u64 inputDataCount = execMem.inputMem.size() / perDataSize;
        CHK_RET(ExecutorBase::PrepareSliceData(inputDataCount, perDataSize, sliceNum, 0, dataSegsSlice));
        multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, param.tag);
        CHK_PRT_RET(multiStreamSlice.size() != ringNum,
            HCCL_ERROR("[CollReduceScatterRingExecutor][KernelRun]ringNum[%u] != multiStreamSlice size[%llu]",
                ringNum, multiStreamSlice.size()), HCCL_E_INTERNAL);

        CHK_RET(MultiRingAllReduce(param.tag, execMem.inputMem, execMem.scratchMem, inputDataCount,
            param.DataDes.dataType, param.reduceType, multiStreamSlice, param.stream, PROF_STAGE_0));

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.inputMem, execMem.scratchMem,
            const_cast<Stream&>(param.stream)));
    }

    std::vector<u32> &nicList = const_cast<std::vector<u32> &>(topoAttr_.nicList);
    std::vector<u32>::iterator iterNic = std::find(nicList.begin(), nicList.end(), topoAttr_.devicePhyId);
    bool innRunRet = isMultiNic && (iterNic == nicList.end());
    if (!innRunRet) { // 1. 8P ring的拓扑。2. 网口不满配。3. 当前device不出网口。 的情况下不进行节点间的reduce scatter
        /* ******************第一步: 节点间reducescatter *******************************/
        u32 innerRankSize = innerCommInfo.localRankSize;
        if (innerRankSize > 1) {
            u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType,
                param.reduceType);
            std::unique_ptr<ExecutorBase> innerExecutor;

            if (UseInterServerRingAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = execMem.inputMem.size() / innerRankSize;
                u64 ringCount = ringSize / perDataSize;

                CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
                    std::vector<Slice>(0)));
            } else if (UseInterServerNHRAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nhr algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = execMem.inputMem.size() / innerRankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
                    std::vector<Slice>(0)));
            } else if (UseInterServerNHRV1Algo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNHRV1(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nhr_v1 algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = execMem.inputMem.size() / innerRankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
                    std::vector<Slice>(0)));
            } else if (UseInterServerNBAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
                CHK_SMART_PTR_NULL(innerExecutor);

                u64 ringSize = execMem.inputMem.size() / innerRankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
                    std::vector<Slice>(0)));
            } else {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");

                CHK_SMART_PTR_NULL(innerExecutor);
                u64 inputDataCount = execMem.inputMem.size() / perDataSize; // count是output的数据个数
                CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, inputDataCount,
                    param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
                    std::vector<Slice>(0)));
            }
            CHK_RET(innerExecutor->RegisterProfiler(
                (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
        }
    }

    /* ***********第二步: 节点内reducescatter(正常场景), 节点内多根结点scatter(网口裁剪)*****************************/
    CHK_RET(ActiveSlaveStreams(param.stream));

    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(), param.DataDes.dataType,
        param.reduceType);
    useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem,
        dataSegsSlice, param.tag);
    bool bRet = (multiStreamSlice.size() != ringNum);
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[CollReduceScatterRingExecutor][KernelRun]sliceNum-1[%u] != multiStreamSlice size[%llu]", \
        sliceNum - 1, multiStreamSlice.size()), HCCL_E_INTERNAL);

    if (isMultiNic) { // 网口裁剪情况下需要改变slice最终在rank上位置
        PrepareMultiRingSlice(dataSegsSlice, param.tag, false, nicList); // 刷新多环ringRankList信息
        std::vector<std::vector<u32>> ringNics;
        CHK_RET(GetRingNics(param.tag, ringNics));

        for (u32 ringIdx = 0; ringIdx < ringNum; ringIdx++) {     // 按第一个网口位置改变slice最终在rank上的位置
            u32 firstNicIdx = ringNics[ringIdx][0];
            std::rotate(multiStreamSlice[ringIdx].begin(), multiStreamSlice[ringIdx].begin() + firstNicIdx,
                        multiStreamSlice[ringIdx].end());
        }
    }

    DeviceMem srcMem;
    if (isMultiNic) {
        u32 innerRankSize = topoAttr_.userRankSize / DEVICE_EIGHT; // currComm->commOuter[0]->UserRankSize();
        // 每个server分配的slice大小
        CHK_PRT_RET(innerRankSize == 0,
            HCCL_ERROR("[CollReduceScatterRingExecutor][KernelRun]innerRankSize is illegal"), HCCL_E_PARA);
        u64 serverSliceSize = execMem.inputMem.size() / innerRankSize;
        // 每个服务器对应的偏移
        u32 serverIndex = innerCommInfo.localRank;
        CHK_PRT_RET(serverIndex == INVALID_VALUE_RANKID,
            HCCL_ERROR("[CollReduceScatterRingExecutor][KernelRun]get rank of "
            "bridgeRank failed, commIdx[%u]", commIndex), HCCL_E_PARA);
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        if (UseInterServerRingAlgo(algType_)) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.scratchMem, execMem.inputMem,
                const_cast<Stream&>(param.stream)));
        }
        DeviceMem reduceScatterRingOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingOutput.ptr());
        u64 countLocal = serverSliceSize / perDataSize;
        CHK_RET(MultiRingMultiRootScatter(param.tag, reduceScatterRingOutput, reduceScatterRingOutput, countLocal,
            param.DataDes.dataType, multiStreamSlice, serverIndex * DEVICE_EIGHT, param.stream, serverSliceOffset));

        srcMem = reduceScatterRingOutput.range(dataSegsSlice[topoAttr_.devicePhyId].offset,
            execMem.count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());
    } else {
        u32 innerRankSize = innerCommInfo.localRankSize;
        // 每个server分配的slice大小
        u64 serverSliceSize = execMem.inputMem.size() / innerRankSize;
        // 每个服务器对应的偏移
        u32 serverIndex = innerCommInfo.localRank;
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        HCCL_DEBUG("inputMem.size=%llu, outerCommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
            "commIndex=%u commInner[commIndex]->rank=%u", execMem.inputMem.size(), outerCommInfo.localRankSize,
            serverSliceSize, serverSliceOffset, commIndex, innerCommInfo.localRank);
        DeviceMem reduceScatterRingInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingInput.ptr());
        DeviceMem reduceScatterRingOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingOutput.ptr());
        u64 countLocal = serverSliceSize / perDataSize;

        HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
            param.root, param.reduceType};
        HcomCollOpInfo *opInfoPtr = nullptr;
        if (DMAReduceFlag_) {
            opInfoPtr = &opInfo;
        }

        CHK_RET(MultiRingReduceScatter(param.tag, reduceScatterRingInput, reduceScatterRingOutput, countLocal,
            param.DataDes.dataType, param.reduceType, multiStreamSlice, param.stream, PROF_STAGE_1, serverSliceOffset,
            opInfoPtr));

        srcMem = execMem.inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset,
            execMem.count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterRingExecutor", ReduceScatterRing, CollReduceScatterRingExecutor);

}

