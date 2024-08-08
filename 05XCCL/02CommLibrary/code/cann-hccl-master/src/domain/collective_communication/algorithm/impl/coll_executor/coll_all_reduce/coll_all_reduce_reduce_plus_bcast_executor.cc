/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_reduce_plus_bcast_executor.h"

namespace hccl {

CollAllReduceReducePlusBcastExecutor::CollAllReduceReducePlusBcastExecutor(const HcclDispatcher dispatcher,
                                                                           std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAllReduceReducePlusBcastExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0;
    HCCL_INFO("[CollAllReduceReducePlusBcastExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceReducePlusBcastExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceReducePlusBcastExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllReduceReducePlusBcastExecutor][CalcTransportMemType]" \
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceReducePlusBcastExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollAllReduceReducePlusBcastExecutor][CalcOuterCommInfo]tag[%s ]start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollAllReduceReducePlusBcastExecutor][CalcOuterCommInfo]tag[%s] Calc RingComm finish",
        tag_.c_str());
    return HCCL_SUCCESS;
}

bool CollAllReduceReducePlusBcastExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollAllReduceReducePlusBcastExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    bool smallData = IsAllReduceSmallData(curSize);
    return smallData;
}

HcclResult CollAllReduceReducePlusBcastExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<ExecutorBase> reduceExecutor =
        std::make_unique<ReduceRecursiveHalvingDoubling>(dispatcher_, reduceAttr);
    CHK_SMART_PTR_NULL(reduceExecutor);

    std::vector<u32> nicRankList{0, 1};
    CHK_RET(reduceExecutor->Prepare(execMem.inputMem, execMem.outputMem, execMem.inputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, 0,
        std::vector<Slice>(0), 0, nicRankList));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    CHK_RET(RunTemplate(reduceExecutor, outerCommInfo));

    // AllReduce算子实现为input->output, 所以此处将reduce算子的结果从output拷贝到input
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_,
        execMem.inputMem, execMem.outputMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("MemcpyAsync failed"), ret);

    // 执行server间allreduce
    if (topoAttr_.devicePhyId == 0) {
        std::unique_ptr<ExecutorBase> allreduceExecutor = nullptr;
        if (UseInterServerRingAlgo(algType_)) {
            allreduceExecutor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce ring: using ring algo inter-server.");
        } else if (UseInterServerNHRAlgo(algType_)) {
            u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType]; // 单位 byte
            HCCL_DEBUG("allreduce recursive hd: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
                allreduceExecutor.reset(new (std::nothrow) AllReduceNHROneshot(dispatcher_, reduceAttr));
            } else {
                allreduceExecutor.reset(new (std::nothrow) AllReduceNHR(dispatcher_, reduceAttr));
            }
            HCCL_INFO("allreduce recursive hd: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            allreduceExecutor.reset(new (std::nothrow) AllReduceNHRV1(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce recursive hd: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            allreduceExecutor.reset(new (std::nothrow) AllReduceNB(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce recursive hd: using nb algo inter-server.");
        } else {
            allreduceExecutor.reset(new (std::nothrow) AllReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce recursive hd: using halving-doubling algo inter-server.");
        }

        CHK_SMART_PTR_NULL(allreduceExecutor);
        CHK_RET(allreduceExecutor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType, 0,
            std::vector<Slice>(0), 0, nicRankList));

        CHK_RET(CheckCommSize(COMM_LEVEL1, COMM_INDEX_0 + 1));
        SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0);
        CHK_RET(RunTemplate(allreduceExecutor, innerCommInfo));
    }

    // 执行server内broadcast
    std::unique_ptr<ExecutorBase> bcastExecutor = std::make_unique<BroadcastRing>(dispatcher_);
    CHK_SMART_PTR_NULL(bcastExecutor);
    CHK_RET(bcastExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, 0));
    CHK_RET(RunTemplate(bcastExecutor, outerCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceReducePlusBcast", AllReduceReducePlusBcast, CollAllReduceReducePlusBcastExecutor);

} // namespace hccl
