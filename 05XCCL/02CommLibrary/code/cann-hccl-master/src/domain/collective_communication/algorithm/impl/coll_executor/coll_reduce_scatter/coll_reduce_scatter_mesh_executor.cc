/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_mesh_executor.h"

namespace hccl {

CollReduceScatterMeshExecutor::CollReduceScatterMeshExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

void CollReduceScatterMeshExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool isInlineReduce = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType,
        param.reduceType);
    meshSinglePlane_ = (topoAttr_.deviceType == DevType::DEV_TYPE_910B) &&
        topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_CONFIG_DISABLE &&
        isInlineReduce && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

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

HcclResult CollReduceScatterMeshExecutor::CalcScratchMemSize(u64& scratchMemSize)
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
    HCCL_INFO("[CollReduceScatterMeshExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%u]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshExecutor::CalcTransportMemType(TransportMemType &inputType,
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
    HCCL_INFO("[CollReduceScatterMeshExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = meshSinglePlane_;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterMeshExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterMeshExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                    (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

HcclResult CollReduceScatterMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    /* ******************第一步: 节点间reducescatter *******************************/
    u32 commIndex = outerCommInfo.localRank; // 找到rank所在的节点间平面

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u32 innerRankSize = innerCommInfo.localRankSize;
    if (innerRankSize > 1) {
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("reducescatter mesh: using ring algo inter-server.");
            u64 ringSize = execMem.inputMem.size() / innerRankSize;
            u64 ringCount = ringSize / perDataSize;
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else if (UseInterServerNHRAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter mesh: using nhr algo inter-server.");
            CHK_SMART_PTR_NULL(innerExecutor);
            u64 ringSize = execMem.inputMem.size() / innerRankSize;
            u64 ringCount = ringSize / perDataSize;
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNHRV1(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter mesh: using nhr_v1 algo inter-server.");
            CHK_SMART_PTR_NULL(innerExecutor);
            u64 ringSize = execMem.inputMem.size() / innerRankSize;
            u64 ringCount = ringSize / perDataSize;
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter mesh: using nonuniform-bruck algo inter-server.");
            CHK_SMART_PTR_NULL(innerExecutor);
            u64 ringSize = execMem.inputMem.size() / innerRankSize;
            u64 ringCount = ringSize / perDataSize;
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("reducescatter mesh: using halving-doubling algo inter-server.");
            // 申请临时内存作为scratch内存
            u64 inputDataCount = execMem.inputMem.size() / perDataSize; // count是output的数据个数
            CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, inputDataCount,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        }
        CHK_RET(innerExecutor->RegisterProfiler(
            (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
    }

    /* *******************第二步: 节点内reducescatter ******************************************/
    CHK_RET(ActiveSlaveStreams(param.stream));

    u32 sliceNum = outerCommInfo.localRankSize;
    // 根据数据量算每个环上数据的偏移和大小，把做完hd的slice均分成RankSize份
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareReduceScatterSliceData(execMem.count, perDataSize, sliceNum, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size() / innerRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * innerCommInfo.localRank;

    HCCL_DEBUG("inputMem.size=%llu, outerCommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "\
        "commIndex=%u innerCommInfo.localRank=%u", execMem.inputMem.size(), outerCommInfo.localRankSize,
        serverSliceSize, serverSliceOffset, commIndex, innerCommInfo.localRank);

    DeviceMem reduceScatterMeshInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshInput);
    DeviceMem reduceScatterMeshOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshOutput);

    HcomCollOpInfo *opInfoPtr = nullptr;

    if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_CONFIG_DISABLE &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64) &&
        (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(param.tag, reduceScatterMeshInput, reduceScatterMeshOutput, // 非确定性
            execMem.count, param.DataDes.dataType, param.reduceType, dataSegsSlice, const_cast<Stream&>(param.stream),
            COMM_LEVEL0, serverSliceOffset, opInfoPtr));
    } else {
        std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(ExecutorBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(param.tag, reduceScatterMeshInput, reduceScatterMeshOutput, // 确定性
            execMem.count, param.DataDes.dataType, param.reduceType, multiStreamSlice,
            const_cast<Stream&>(param.stream), COMM_LEVEL0, serverSliceOffset));
    }

    DeviceMem srcMem = execMem.inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset,
        execMem.count * perDataSize);
    CHK_SMART_PTR_NULL(srcMem);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterMeshExecutor", ReduceScatterMesh, CollReduceScatterMeshExecutor);
}