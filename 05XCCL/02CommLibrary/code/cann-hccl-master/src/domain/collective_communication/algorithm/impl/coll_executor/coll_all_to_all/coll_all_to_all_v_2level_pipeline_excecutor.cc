/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_all_to_all_v_2level_pipeline_excecutor.h"
namespace hccl {

CollRunAlltoAllVTwoLevelPipeline::CollRunAlltoAllVTwoLevelPipeline(const HcclDispatcher dispatcher,
                                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

// 计算 alltoall pipeline 910B 的两级流水算法本卡需要的 scratch 大小(图模式需要)
u64 CollRunAlltoAllVTwoLevelPipeline::GetAlltoall2LevelPipelineScratchSize910B(
    u32 rank, std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 scratchSize = 0;
    u32 meshRankStart = (rank / topoAttr_.meshAggregationRankSize) * topoAttr_.meshAggregationRankSize;
    u32 meshRankEnd = meshRankStart + topoAttr_.meshAggregationRankSize - 1;
    u32 rankIntraMesh = rank - meshRankStart;
    for (u32 sendRank = rankIntraMesh, userRankSize = allMeshAggregationSendRecvInfo.size();
        sendRank < userRankSize; sendRank += topoAttr_.meshAggregationRankSize) {
        const std::vector<u64>& remoteSendLength = allMeshAggregationSendRecvInfo[sendRank].sendLength;
        const std::vector<u64>& remoteSendOffset = allMeshAggregationSendRecvInfo[sendRank].sendOffset;
        scratchSize += (remoteSendOffset[meshRankEnd] + remoteSendLength[meshRankEnd] -
            remoteSendOffset[meshRankStart]);
    }
    return scratchSize;
}

// 计算 alltoall pipeline 910B 的两级流水算法所有卡需要的 scratch 大小的最大值(单算子模式需要)
u64 CollRunAlltoAllVTwoLevelPipeline::GetAlltoall2LevelPipelineMaxScratchSize910B(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 maxScratchSize = 0;
    for (u32 rank = 0, userRankSize = allMeshAggregationSendRecvInfo.size(); rank < userRankSize; rank++) {
        u64 currRankScratchSize = GetAlltoall2LevelPipelineScratchSize910B(rank, allMeshAggregationSendRecvInfo);
        maxScratchSize = (currRankScratchSize > maxScratchSize ? currRankScratchSize : maxScratchSize);
    }
    return maxScratchSize;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = 0U;
    u64 tmpMemSize = 0U;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        // 图模式才需要申请 scratch 在此只计算scratchMem size
        tmpMemSize = GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo_);
    }
    scratchMemSize = CalAlltoAllVScratchMemSize(tmpMemSize);
    HCCL_INFO("[CollRunAlltoAllVTwoLevelPipeline][CalcScratchMemSize] tag_[%s] scratchMemSize[%u]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollRunAlltoAllVTwoLevelPipeline][CalcStreamNum] tag_[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_MESH_L0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_MESH_L1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_MESH_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CalNoScratchAlltoallCommInfo(inputType, outputType, opTransport);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalNoScratchAlltoallCommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CalcLevel0CommInfo(TransportMemType::CCL_OUTPUT, TransportMemType::CCL_OUTPUT, opTransport);
        CalcLevel1CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport);
    } else {
        CalcLevel0CommInfo(TransportMemType::SCRATCH, TransportMemType::CCL_OUTPUT, opTransport);
        CalcLevel1CommInfo(TransportMemType::CCL_INPUT, TransportMemType::SCRATCH, opTransport);
    }

    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollRunAlltoAllVTwoLevelPipeline][KernelRun] alltoall two level pipeline start");

    // 子图
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        bool hugeData = algRes_.paramInputMem.size() > SDMA_SEND_MAX_SIZE;
        bool alltoallPingPong = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
            !topoAttr_.multiModuleDiffDeviceNumMode &&
            GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo_) >
            execMem.inputMem);
        if (AlltoAllVParam_.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            auto opMeta = HcclOpMetaInfo::GetOneForAllToAllV((isAlltoAllZCopyMode_ ?
                CopyPattern::ZCOPY : CopyPattern::BCOPY), algRes_.paramInputMem.size(),
                hugeData || alltoallPingPong);
            CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        } else {
            auto opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::BCOPY,
                algRes_.paramInputMem.size(), hugeData || alltoallPingPong);
            CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        }
    }

    bool cclEnough = true;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo_) >
            execMem.inputMem) {
        cclEnough = false;
    }
    HCCL_DEBUG("[CollRunAlltoAllVTwoLevelPipeline][KernelRun] alltoall pipeline run %s algo",
        cclEnough ? "cclEnough" : "ping pong");
    A2aPipelineMemory a2aPipelineMemory;
    a2aPipelineMemory.userInput = algRes_.paramInputMem;
    a2aPipelineMemory.userOutput = algRes_.paramOutputMem;
    // 具体传入 A2aPipelineMemory 对象的 alltoall pipeline executor 会根据图模式还是单算子模式
    // 选择使用 ccl 还是 scratch，不会访问空指针
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        a2aPipelineMemory.cclInBuffer = execMem.inputMem;
        a2aPipelineMemory.cclOutBuffer = execMem.outputMem;
    } else {
        a2aPipelineMemory.scratchMem = execMem.scratchMem;
    }

    std::unique_ptr<AlltoallPipelineBase> alltoallPipe = nullptr;
    if (cclEnough) {
        alltoallPipe.reset(new (std::nothrow)AlltoallPipelineMeshPairwiseCCLEnough(dispatcher_,
            allMeshAggregationSendRecvInfo_, GetWorkflowMode()));
    } else {
        alltoallPipe.reset(new (std::nothrow)AlltoallPipelineMeshPairwisePingPong(dispatcher_,
            allMeshAggregationSendRecvInfo_, GetWorkflowMode()));
    }

    CHK_RET(CheckCommSize(COMM_MESH_L0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);
    CHK_RET(CheckCommSize(COMM_MESH_L1, COMM_INDEX_0 + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_MESH_L1, COMM_INDEX_0); // 待确认 zlj

    alltoallPipe->Prepare(topoAttr_.userRank, a2aPipelineMemory, outerCommInfo, innerCommInfo,
        const_cast<Stream&>(param.stream), streamInfo_.ringStreams, streamInfo_.ringSignal, streamInfo_.ringSignalAux);
    alltoallPipe->RunAsync();
    HCCL_INFO("[CollRunAlltoAllVTwoLevelPipeline][kernelRun] alltoall two level pipeline end");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllVTwoLevelPipeline", AlltoAllVTwoLevelPipeline, CollRunAlltoAllVTwoLevelPipeline);
} // namespace hccl