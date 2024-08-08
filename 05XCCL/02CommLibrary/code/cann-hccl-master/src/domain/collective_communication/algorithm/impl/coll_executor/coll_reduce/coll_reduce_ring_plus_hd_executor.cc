/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_reduce_ring_plus_hd_executor.h"

namespace hccl {

CollReduceRingPlusHdExecutor::CollReduceRingPlusHdExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceRingPlusHdExecutor::CalcStreamNum(u32& streamNum)
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
            totalStreamNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
            break;
        default:
            break;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollReduceRingPlusHdExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingPlusHdExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CalcLevel0CommInfo(inputType, outputType, opTransport);
    CalcLevel1CommInfo(inputType, outputType, opTransport);
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingPlusHdExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollReduceRingPlusHdExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingPlusHdExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollReduceRingPlusHdExecutor][CalcOuterCommInfo]tag[%s ]start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollReduceRingPlusHdExecutor][CalcOuterCommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingPlusHdExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > mulRingSlice; // 数据基于该rank上环0的偏移

    // step1: 节点内的reducescatter
    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? OUTER_PLANE_NUM_IN_8PRING :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE;

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 按ranksize得到内存切分slice数为8
    u32 sliceNum = outerCommInfo.localRankSize;
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 外层:reducescatter */
    // 将每slice再切分成4份，按各ring的dev顺序排列
    if (ringNum == OUTER_PLANE_NUM_IN_8PRING) {
        // 构造ring algorithm对应的reduce-scatter实例
        mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, tag_, false, topoAttr_.nicList);
        CHK_PRT_RET(mulRingSlice.size() != ringNum, HCCL_ERROR("[CollReduceRingPlusHdExecutor]ringNum[%u] "\
            "!=mulRingSlice size[%llu]", ringNum, mulRingSlice.size()), HCCL_E_INTERNAL);
    } else {
        mulRingSlice.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }
    CHK_RET(MultiRingReduceScatter(tag_, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.reduceType, mulRingSlice, param.stream,
                                   PROF_STAGE_0, 0, nullptr));

    HCCL_INFO("reduce 8PringHD stage0 run success");

    // step2: 节点间的reduce
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, outerCommInfo, mulRingSlice, tag_));

    u64 hdCount = hdSize / perDataSize;

    HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, tag_.c_str(),
        outerCommInfo.localRankSize);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    DeviceMem reduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
    CHK_SMART_PTR_NULL(reduceInput);
    DeviceMem reduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
    CHK_SMART_PTR_NULL(reduceOutput);

    u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<ExecutorBase> innerExecutor;
    if (UseInterServerRingAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
    } else {
        innerExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
    CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR("[ReduceOperator][ReduceRingPlusHd]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, topoAttr_.userRank, param.root), HCCL_E_INTERNAL);

    u32 planeRoot = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));

    u32 ranksize = innerCommInfo.localRankSize;
    // 节点间的hd 使用环0来记录
    CHK_RET(innerExecutor->Prepare(reduceInput, reduceOutput, reduceOutput, hdCount, param.DataDes.dataType,
        param.stream, param.reduceType, planeRoot, std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));

    CHK_RET(innerExecutor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank, \
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

    HCCL_INFO("reduce 8PringHD stage1 run success");

    // step3: 节点内的gatherring，只有在root所在server内进行gather操作
    SingleSubCommTransport &outerTransportInfo =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_LEVEL0][COMM_INDEX_0]);

    if (outerTransportInfo.userRank2subCommRank.find(param.root) !=
        outerTransportInfo.userRank2subCommRank.end()) {
        CHK_RET(MultiRingGather(tag_, execMem.outputMem, execMem.outputMem, hdCount, param.DataDes.dataType,
            mulRingSlice, param.reduceType, param.root, const_cast<Stream &>(param.stream), PROF_STAGE_2));
    }
    HCCL_INFO("reduce 8PringHD stage2 run success");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceRingPlusHd", ReduceRingPlusHd, CollReduceRingPlusHdExecutor);

} // namespace hccl