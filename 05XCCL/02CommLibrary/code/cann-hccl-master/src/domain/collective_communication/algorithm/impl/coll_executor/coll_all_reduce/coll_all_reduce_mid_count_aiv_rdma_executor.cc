/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mid_count_aiv_rdma_executor.h"

namespace hccl {
constexpr s32 INTRA_RS_STEP = 0;
constexpr s32 INTRA_AG_STEP = 2;
 
CollAllReduceMidCountAivRdmaExecutor::CollAllReduceMidCountAivRdmaExecutor(const HcclDispatcher dispatcher,
                                                                           std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    // AIV通信需要AIV buffer
    needAivBuffer = true;
    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u]",
        tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    // 中数据量：使用AIVIN，标记区在AIVIN末尾，单算子模式用CCLOUT，图模式用USEROUT
    inputType = TransportMemType::AIV_INPUT;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::Orchestrate(const OpParam& param, const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    CHK_RET(GetStreamInfo(algRes));

    // 中数据量：使用AIVIN，标记区在AIVIN末尾，单算子模式用CCLOUT，图模式用USEROUT
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.aivInputMem;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.outputMem = algRes.cclOutputMem;
    } else {
        execMem.outputMem = algRes.paramOutputMem;
    }
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceMidCountAivRdmaExecutor]errNo[0x%016llx] tag[%s] excutor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllReduce executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor][KernelRun]allreduce aiv enter");
    HcclWorkflowMode workflow = GetWorkflowMode();
    bool isOpbase = (workflow == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    CHK_RET(ActiveSlaveStreams(param.stream));

    // 获取通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 数据准备，按照server内rankSize切片
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];
    u64 totalSize = param.DataDes.count * perDataSize;
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    u32 sliceNum = outerCommInfo.localRankSize;
    CHK_RET(PrepareSliceDataWithAlignSize(totalSize, sliceNum, 0, dataSegsSlice, HCCL_ALIGN_COUNT_32_B));
    CHK_PRT_RET(commIndex >= dataSegsSlice.size(),
        HCCL_ERROR("[CollAllReduceMeshExecutor][Run]commIndex[%u] >= dataSegsSlice size[%llu]", commIndex,
        dataSegsSlice.size()), HCCL_E_INTERNAL);
    std::vector<hccl::LINK> intraLinks = outerCommInfo.links;
    std::vector<hccl::LINK> interLinks = innerCommInfo.links;
    u32 intraRankSize = outerCommInfo.localRankSize;
    u32 intraRankId = outerCommInfo.localRank;

    // reduce scatter阶段，inputMem0-31m做数据区，32M开始后的1M做标记区
    void* dataBuffers[MAX_RANK_SIZE];
    void* flagBuffers[MAX_RANK_SIZE];  // 标记区的具体偏移在kernel中决定
    CHK_RET(PrepareAivBuffers(intraRankSize, intraRankId, 0, execMem.inputMem, execMem.inputMem, intraLinks,
        dataBuffers, flagBuffers, UserMemType::INPUT_MEM, UserMemType::INPUT_MEM, 0, HCCL_MID_COUNT_32_MB));
    // 先做本地拷贝到AIVIN再跨片拷贝；output统一为allreduceInput的位置，即buffer中原位
    CHK_RET(ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLREDUCE, execMem.inputPtr, nullptr, execMem.count,
        param.DataDes.dataType, param.reduceType, intraRankId, intraRankSize, 0, dataBuffers, flagBuffers,
        param.stream.ptr(), isOpbase, execMem.inputMem.size(), INTRA_RS_STEP, false));

    // allreduce 阶段
    std::unique_ptr<ExecutorBase> innerExecutor;
    DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceInput);
    DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceOutput);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(autoSelectedAlgTypeLevel1, param.DataDes.dataType,
        ReduceType::INLINE_REDUCE, IsAllReduceSmallData(totalSize), 1, false, hccl::CopyPattern::BCOPY, 1, true);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (UseInterServerRingAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce mesh: using ring algo inter-server.");
    } else if (UseInterServerNHRAlgo(algType_)) {
        u64 curSize = execMem.count * perDataSize; // 单位 byte
        HCCL_DEBUG("allreduce mesh: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
            curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
        if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
            innerExecutor.reset(new (std::nothrow) AllReduceNHROneshot(dispatcher_, reduceAttr));
        } else {
            innerExecutor.reset(new (std::nothrow) AllReduceNHR(dispatcher_, reduceAttr));
        }
        HCCL_INFO("allreduce mesh: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllReduceNHRV1(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce mesh: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) AllReduceNB(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce mesh: using nb algo inter-server.");
    } else {
        innerExecutor.reset(new (std::nothrow) AllReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
        HCCL_INFO("allreduce mesh: using Recursive halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    u32 rankSize = innerCommInfo.localRankSize;
    u64 hdCount = dataSegsSlice[commIndex].size / perDataSize;
    CHK_RET(innerExecutor->Prepare(allreduceInput, allreduceOutput, allreduceOutput, hdCount,
        param.DataDes.dataType, param.stream, param.reduceType,
        OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

    CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        innerCommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor] rdma stage run success.");
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));

    // allgather阶段，outputMem做数据区，32M开始后的1M做标记区
    CHK_RET(PrepareAivBuffers(intraRankSize, intraRankId, 0, execMem.outputMem, execMem.inputMem, intraLinks,
        dataBuffers, flagBuffers, UserMemType::OUTPUT_MEM, UserMemType::INPUT_MEM, 0, HCCL_MID_COUNT_32_MB));
    // 输入统一为allreduceOutput的位置，各卡不同；单算子模式需要outputAddr，先做本地拷贝再跨片拷贝；图模式结果直接放在CCL Out中
    CHK_RET(ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, execMem.outputPtr, execMem.count,
        param.DataDes.dataType, param.reduceType, intraRankId, intraRankSize, 0, dataBuffers, flagBuffers,
        param.stream.ptr(), isOpbase, execMem.inputMem.size(), INTRA_AG_STEP, false));

    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor][KernelRun]allreduce aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMidCountAivRdmaExecutor", AllReduceMidCountAivRdma, CollAllReduceMidCountAivRdmaExecutor);

} // namespace hccl