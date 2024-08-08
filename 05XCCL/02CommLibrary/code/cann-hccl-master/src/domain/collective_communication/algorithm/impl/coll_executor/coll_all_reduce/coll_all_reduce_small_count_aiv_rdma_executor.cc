/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_small_count_aiv_rdma_executor.h"

namespace hccl {
constexpr s32 INTRA_RS_STEP = 0;
constexpr s32 INTER_AR_STEP = 1;
constexpr s32 INTRA_AG_STEP = 2;
constexpr u32 A_X_AGGR_SIZE = 2;
constexpr u64 HALF_OFFSET = 16 * 1024 * 1024;

u64 CollAllReduceSmallCountAivRdmaExecutor::allreduceSmallDataAivRdmaCount_ = 0;
 
CollAllReduceSmallCountAivRdmaExecutor::CollAllReduceSmallCountAivRdmaExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    // AIV通信需要AIV buffer
    needAivBuffer = true;
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u]",
        tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));

    // aiv+rdma小数据量在server间使用HD通信域，并在多机A+X场景下当未设置使用RDMA时，默认使用PCIE
    if (topoMatcher_->GetExternalInputIntraRoceSwitch() == 0) {
        std::vector<SingleSubCommTransport> &commTransportLevel1 = opTransport[COMM_LEVEL1];
        for (u32 ringIndex = 0; ringIndex < commTransportLevel1.size(); ringIndex++) {
            commTransportLevel1[ringIndex].isUsedRdma = false;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    // 小数据量：使用AIVIN+AIVOUT，标记区在AIVOUT，RS前从inputPtr到AIVIN做本地拷贝
    inputType = TransportMemType::AIV_INPUT;
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL1, CommType::COMM_TAG_HALVING_DOUBLING);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::Orchestrate(const OpParam& param, const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    allreduceSmallDataAivRdmaCount_ += 1;
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][Orchestrate] AllreduceSmallCountAivRdma has been called [%d].",
        allreduceSmallDataAivRdmaCount_);
    tag_ = param.tag;
    algResResp_ = &algRes;
    CHK_RET(GetStreamInfo(algRes));

    // 小数据量：使用AIVIN+AIVOUT，标记区在AIVOUT
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.aivInputMem;
    execMem.outputMem = algRes.aivOutputMem;
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceSmallCountAivRdmaExecutor]errNo[0x%016llx] tag[%s] excutor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllReduce executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::InterServerHDOneshot(const OpParam &param, ExecMem &execMem,
    u32 &outputOffset, u64 sliceCount, u32 dbOffset, u32 interRankSize, u32 interRankId, bool isOpbase,
    std::vector<LINK> &interLinks)
{
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][InterServerHDOneshot]reduceAttr is [%llu]", reduceAttr);
    std::unique_ptr<Sender> senderInfo;
    std::unique_ptr<Reducer> reducerInfo;
    senderInfo.reset(new (std::nothrow) Sender(param.DataDes.dataType, param.reduceType, reduceAttr));
    CHK_SMART_PTR_NULL(senderInfo);
    reducerInfo.reset(new (std::nothrow) Reducer(param.DataDes.dataType, param.reduceType, reduceAttr));
    CHK_SMART_PTR_NULL(reducerInfo);
    u32 hdStepNum = log2(interRankSize);
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor] Find interlink type for cross-aggregation link[%d].",
        interLinks[(interRankId + 1) % A_X_AGGR_SIZE]->GetLinkType());
    u32 sliceSize = sliceCount * SIZE_TABLE[param.DataDes.dataType];
    auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(0, param.DataDes.dataType, ReduceType::INLINE_REDUCE,
                true, 0, false, hccl::CopyPattern::BCOPY, 1, true);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    for (u32 step = 1; step <= hdStepNum; step++) {
        u32 peerMask = 1 << (hdStepNum - step);
        u32 peer = interRankId ^ peerMask;
        HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][InterServerHDOneshot] Step %llu, peer %llu.", step, peer);
        if (step == 1 && interLinks[peer]->GetLinkType() == LinkType::LINK_PCIE) {
            // a+x单机，走PCIE
            void *dataBuffers[MAX_RANK_SIZE];
            void *flagBuffers[MAX_RANK_SIZE];
            void* arOutput = static_cast<u8 *>(execMem.inputMem.ptr()) + HCCL_SMALL_COUNT_2_MB + sliceSize + dbOffset;
            CHK_RET(PrepareAivBuffers(A_X_AGGR_SIZE, interRankId % A_X_AGGR_SIZE,
                interRankId - interRankId % A_X_AGGR_SIZE, execMem.inputMem, execMem.outputMem, interLinks, dataBuffers,
                flagBuffers, UserMemType::INPUT_MEM, UserMemType::OUTPUT_MEM, HCCL_SMALL_COUNT_2_MB + dbOffset, 0));
            CHK_RET(ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, arOutput, sliceCount,
                param.DataDes.dataType, param.reduceType, interRankId, interRankSize, 0, dataBuffers, flagBuffers,
                param.stream.ptr(), isOpbase, execMem.inputMem.size(), INTER_AR_STEP, true));
        } else {
            // 其他情况走RDMA
            auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(0, param.DataDes.dataType, ReduceType::INLINE_REDUCE,
                true, step, false, hccl::CopyPattern::BCOPY, 1, true);
            CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
            u32 sliceForReadOffset = HCCL_SMALL_COUNT_2_MB + (step - 1) * sliceSize + dbOffset;
            u32 sliceForWriteOffset = HCCL_SMALL_COUNT_2_MB + step * sliceSize + dbOffset;
            DeviceMem src = execMem.inputMem.range(sliceForReadOffset, sliceSize);
            DeviceMem dst = execMem.inputMem.range(sliceForWriteOffset, sliceSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, const_cast<Stream&>(param.stream)));
            interLinks[peer]->TxAck(const_cast<Stream&>(param.stream));
            interLinks[peer]->RxAck(const_cast<Stream&>(param.stream));
            if (interLinks[peer]->IsSupportTransportWithReduce() && 
                ((interLinks[peer]->GetLinkType() == LinkType::LINK_STANDARD_ROCE) ||
                (RDMA_REDUCE_BITMASK & reduceAttr))) {
                CHK_RET(senderInfo->run(interLinks[peer], sliceForWriteOffset, src, const_cast<Stream&>(param.stream),
                    UserMemType::INPUT_MEM));
                CHK_RET(reducerInfo->run(dispatcher_, interLinks[peer], 0, src, src, src, 
                    const_cast<Stream&>(param.stream), DstMemType::RESULT_INPUT_MEM, UserMemType::INPUT_MEM));
            } else {
                CHK_RET(interLinks[peer]->TxAsync(UserMemType::INPUT_MEM, HALF_OFFSET + sliceForWriteOffset, 
                    src.ptr(), src.size(), const_cast<Stream&>(param.stream)));
                DeviceMem localSrc = execMem.inputMem.range(HALF_OFFSET + sliceForWriteOffset, sliceSize);
                CHK_RET(reducerInfo->run(dispatcher_, interLinks[peer], 0, localSrc, dst, src, 
                    const_cast<Stream&>(param.stream), DstMemType::RESULT_INPUT_MEM, UserMemType::INPUT_MEM));
            }
        }
    }
     CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    outputOffset = HCCL_SMALL_COUNT_2_MB + hdStepNum * sliceSize;
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][KernelRun]allreduce aiv enter.");
    HcclWorkflowMode workflow = GetWorkflowMode();
    bool isOpbase = (workflow == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

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
    CHK_RET(PrepareSliceDataWithAlignSize(totalSize, sliceNum, 0, dataSegsSlice, perDataSize));
    CHK_PRT_RET(commIndex >= dataSegsSlice.size(),
        HCCL_ERROR("[CollAllReduceMeshExecutor][Run]commIndex[%u] >= dataSegsSlice size[%llu]", commIndex,
        dataSegsSlice.size()), HCCL_E_INTERNAL);
    std::vector<hccl::LINK> intraLinks = outerCommInfo.links;
    std::vector<hccl::LINK> interLinks = innerCommInfo.links;
    u32 intraRankSize = outerCommInfo.localRankSize;
    u32 interRankSize = innerCommInfo.localRankSize;
    u32 intraRankId = outerCommInfo.localRank;
    u32 interRankId = innerCommInfo.localRank;

    // reduce scatter via AIV
    void* dataBuffers[MAX_RANK_SIZE];
    void* flagBuffers[MAX_RANK_SIZE];  // 标记区的具体偏移在kernel中决定
    CHK_RET(PrepareAivBuffers(intraRankSize, intraRankId, 0, execMem.inputMem, execMem.outputMem, intraLinks,
        dataBuffers, flagBuffers, UserMemType::INPUT_MEM, UserMemType::OUTPUT_MEM, 0, 0));
    // RS总数据量最大1m，rs的结果存储到2m处
    void* rsOutput = static_cast<u8 *>(execMem.inputMem.ptr()) + HCCL_SMALL_COUNT_2_MB;
    CHK_RET(ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLREDUCE, execMem.inputPtr, rsOutput, execMem.count,
        param.DataDes.dataType, param.reduceType, intraRankId, intraRankSize, 0, dataBuffers, flagBuffers,
        param.stream.ptr(), isOpbase, execMem.inputMem.size(), INTRA_RS_STEP, true));

    // use hd algo
    u32 arOutputOffset = 0;  // 跨机allreduce的结果的位置，相对于inputMem的偏移
    CHK_RET(InterServerHDOneshot(param, execMem, arOutputOffset, dataSegsSlice[commIndex].size / perDataSize,
        0, interRankSize, interRankId, isOpbase, interLinks));
    void *arOutput = static_cast<u8 *>(execMem.inputMem.ptr()) + arOutputOffset;

    // all gather via AIV
    CHK_RET(PrepareAivBuffers(intraRankSize, intraRankId, 0, execMem.inputMem, execMem.outputMem, intraLinks,
        dataBuffers, flagBuffers, UserMemType::INPUT_MEM, UserMemType::OUTPUT_MEM, HCCL_SMALL_COUNT_8_MB,
        0));
    CHK_RET(ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLREDUCE, arOutput, execMem.outputPtr, execMem.count,
        param.DataDes.dataType, param.reduceType, intraRankId, intraRankSize, 0, dataBuffers, flagBuffers,
        param.stream.ptr(), isOpbase, execMem.inputMem.size(), INTRA_AG_STEP, true));

    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][KernelRun]allreduce aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceSmallCountAivRdmaExecutor", AllReduceSmallCountAivRdma, CollAllReduceSmallCountAivRdmaExecutor);

} // namespace hccl