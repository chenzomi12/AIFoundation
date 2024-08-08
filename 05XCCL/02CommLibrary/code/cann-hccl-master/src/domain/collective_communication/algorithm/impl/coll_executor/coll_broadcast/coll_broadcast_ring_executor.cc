/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_ring_executor.h"

namespace hccl {

CollBroadcastRingExecutor::CollBroadcastRingExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        topoAttr_.deviceType == DevType::DEV_TYPE_910_73) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollBroadcastRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? OUTER_PLANE_NUM_IN_8PRING :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE;

    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_73) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
        } else {
            totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
        }
    }

    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollBroadcastRingExecutor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastRingExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollBroadcastRingExecutor][CalcOuterCommInfo]tag[%s ]start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollBroadcastRingExecutor][CalcOuterCommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollBroadcastRingExecutor][Run]The CollBroadcastRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multRingsSliceZero; // 数据基于该rank上环0的偏移
    // step1: 节点内的scatter
    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? OUTER_PLANE_NUM_IN_8PRING :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE;

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // 按ranksize得到内存切分slice数
    u32 sliceNum = level0CommInfo.localRankSize;
    // 将根节点数据切分成sliceNum份
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    HCCL_DEBUG("[CollBroadcastRingExecutor][KernelRun] execMem.count[%llu], perDataSize[%llu], sliceNum[%llu], ringNum[%llu] ",
                execMem.count, perDataSize, sliceNum, ringNum);

    /* 外层:scatter */
    // 将每slice再切分成4份，按各ring的dev顺序排列
    if (ringNum == OUTER_PLANE_NUM_IN_8PRING) {
        // 构造ring algorithm对应的scatter实例
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        CHK_PRT_RET(multRingsSliceZero.size() != ringNum, HCCL_ERROR("[CollBroadcastRingExecutor]"\
            "ringNum[%u] !=multRingsSliceZero size[%llu]", ringNum, multRingsSliceZero.size()), HCCL_E_INTERNAL);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }

    HcomCollOpInfo *scatterOpInfoPtr = nullptr;
    HcomCollOpInfo scatterOpInfo = {
        "", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root
    };
    if (DMAReduceFlag_) {
        scatterOpInfoPtr = &scatterOpInfo;
    }
    CHK_RET(MultiRingScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
                             multRingsSliceZero, param.root, param.stream, scatterOpInfoPtr));

    HCCL_INFO("broadcast 8PringHD stage0 run success");

    // step2: 节点间的broadcast
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, level0CommInfo, multRingsSliceZero, param.tag));

    u64 hdCount = hdSize / perDataSize;
    auto nicList = topoAttr_.nicList;
    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList.size() != DEVICE_EIGHT;
    std::vector<u32>::iterator iterNic = std::find(nicList.begin(), nicList.end(), topoAttr_.devicePhyId);
    bool innRunRet = isMultiNic && (iterNic == nicList.end());
    if (!innRunRet) { // 满足以下条件, 不做server间通信: 1. 8P ring的拓扑 2. 网口不满配 3. 当前device不出网口
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerNHRAlgo(algType_)) {
            HCCL_DEBUG("broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
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
            const u32 innerRankSize = level1CommInfo.localRankSize;
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
        CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[BroadCastOperator][BroadCastRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                subUserrankRoot, topoAttr_.userRank, param.root), HCCL_E_INTERNAL);
        u32 planeRoot = 0;
        u32 level1RankSize = level1CommInfo.localRankSize;
        u32 level1LocalRank = level1CommInfo.localRank;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));

        // 节点间的hd 使用环0来记录
        CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.outputMem, hdCount, param.DataDes.dataType,
                param.stream, HCCL_REDUCE_RESERVED, planeRoot, std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));

        CHK_RET(innerExecutor->RegisterProfiler((level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1LocalRank, \
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
    }
    HCCL_INFO("broadcast 8PringHD stage1 run success");

    // step3: 节点内的allgatherring
    HcomCollOpInfo *allgatherOpInfoPtr = nullptr;
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, HCCL_REDUCE_RESERVED
    };
    if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
    }
    CHK_RET(MultiRingAllGather(param.tag, execMem.inputMem, execMem.outputMem, hdCount, param.DataDes.dataType,
                             multRingsSliceZero, param.stream, PROF_STAGE_2, 0, allgatherOpInfoPtr));

    HCCL_INFO("broadcast 8PringHD stage2 run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadCastRingExecutor", BroadcastRing, CollBroadcastRingExecutor);

 } // namespace hccl