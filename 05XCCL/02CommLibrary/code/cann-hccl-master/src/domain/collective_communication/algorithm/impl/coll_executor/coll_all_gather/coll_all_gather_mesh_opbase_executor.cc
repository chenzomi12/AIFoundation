/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_mesh_opbase_executor.h"

namespace hccl {
CollAllGatherMeshOpbaseExecutor::CollAllGatherMeshOpbaseExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllGatherMeshOpbaseExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherMeshOpbaseExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshOpbaseExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshOpbaseExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherMeshOpbaseExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshOpbaseExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherMeshOpbaseExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    return maxCountPerLoop;
}

bool CollAllGatherMeshOpbaseExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollAllGatherMeshOpbaseExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u8 *curInputPtr = static_cast<u8 *>(execMem.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(execMem.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    // 获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = 0;
    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小

    CHK_RET(ActiveSlaveStreams(param.stream));

    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem.range(baseOffset, inputMemSize); // 减少dma out大小
    CHK_SMART_PTR_NULL(currentOutputMem);

    // DMA消减场景，打包opInfo
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };

    std::unique_ptr<ExecutorBase> outerExecutor;
    outerExecutor.reset(
        new (std::nothrow) AllgatherMeshDirect(dispatcher_, streamInfo_.ringStreams,
        streamInfo_.ringSignal, streamInfo_.ringSignalAux, outerCommInfo.localRank, outerCommInfo.localRankSize,
        topoAttr_.userRank, &opInfo));
    CHK_SMART_PTR_NULL(outerExecutor);
    CHK_RET(outerExecutor->Prepare(currentOutputMem, currentOutputMem, execMem.inputMem, execMem.count,
        param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
        dataSegsSlice, baseOffset));

    u32 rankSize = outerCommInfo.localRankSize;
    CHK_RET(outerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(outerExecutor, outerCommInfo));

    HCCL_INFO("all gather mesh outer run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherMeshOpbaseExecutor", AllGatherOpbase, CollAllGatherMeshOpbaseExecutor);
} // namespace hccl
