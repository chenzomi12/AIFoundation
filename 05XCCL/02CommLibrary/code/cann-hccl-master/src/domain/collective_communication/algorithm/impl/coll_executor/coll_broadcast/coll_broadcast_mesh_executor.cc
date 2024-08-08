/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "coll_broadcast_mesh_executor.h"

 namespace hccl {

CollBroadcastMeshExecutor::CollBroadcastMeshExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollBroadcastMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    switch(algType_) {
        case AlgType::ALG_4P_MESH_PLUS_HD:
        case AlgType::ALG_4P_MESH_PLUS_RING:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NB:
            totalStreamNum = OUTER_PLANE_NUM_IN_4PMESH;
            break;
        default:
            if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
                (topoAttr_.deviceType == DevType::DEV_TYPE_910B) && topoAttr_.isSingleMeshAggregation ) {
                totalStreamNum = topoAttr_.deviceNumPerAggregation;
            } else if ((topoAttr_.deviceType == DevType::DEV_TYPE_910_73)) { // && (isAicpuModeEn == true)
                totalStreamNum = topoAttr_.deviceNumPerAggregation;
            } else if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
                       (topoAttr_.deviceType == DevType::DEV_TYPE_910B) && UseInterServerPipelineAlgo(algType_)) {
                totalStreamNum = topoAttr_.deviceNumPerAggregation + 1; /* pipeline ring场景下性能优化 */
            } else {
                totalStreamNum = topoAttr_.deviceNumPerAggregation - 1;
            }
            break;
    }

    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollBroadcastMeshExecutor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    std::unique_ptr<ExecutorBase> outer1Executor;
    std::unique_ptr<ExecutorBase> innerExecutor;
    std::unique_ptr<ExecutorBase> outer2Executor;

    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));

    outer1Executor.reset(
        new (std::nothrow) ScatterMesh(dispatcher_, outerCommInfo.localRank, outerCommInfo.localRankSize));
    CHK_SMART_PTR_NULL(outer1Executor);

    /* 内层topo:all_reduce */
    /* 外层所有rank均参与内层的broadcast计算，所以此处对rank不作限制，但是每个rank需找到自己所在的内层通信域 */
    std::vector<Slice> slice;
    CHK_RET(GetRankSliceSize(param.DataDes.dataType, execMem.count, outerCommInfo.localRankSize, slice));

    CHK_PRT_RET(slice.empty(), HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]got slice is empty"),
        HCCL_E_INTERNAL);

    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    if (UseInterServerNHRAlgo(algType_)) {
        HCCL_DEBUG("broadcast mesh: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
            curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
        if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
            innerExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
        } else {
            innerExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
        }
        HCCL_INFO("broadcast mesh: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        innerExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
        HCCL_INFO("broadcast mesh: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        const u32 innerRankSize = innerCommInfo.localRankSize;
        if (ShouldUseBinaryBroadcastOfNB(curSize / topoAttr_.deviceNumPerAggregation, innerRankSize,
                topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation)) {
            innerExecutor.reset(new (std::nothrow) BroadcastNBBinary(dispatcher_));
        } else {
            innerExecutor.reset(new (std::nothrow) BroadcastNB(dispatcher_));
        }
        HCCL_INFO("broadcast mesh: using nonuniform-bruck algo inter-server.");
    } else {
        innerExecutor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
        HCCL_INFO("broadcast mesh: using Recursive halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    /* 外层topo:all_gather */
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        outer2Executor.reset(
            new (std::nothrow) AllGatherMeshAtomic(dispatcher_, streamInfo_.ringStreams,
            streamInfo_.ringSignal, streamInfo_.ringSignalAux, outerCommInfo.localRank, outerCommInfo.localRankSize,
            topoAttr_.userRank));
    } else {
        outer2Executor.reset(
            new (std::nothrow) AllGatherMesh(dispatcher_, streamInfo_.ringStreams, streamInfo_.ringSignal,
            streamInfo_.ringSignalAux, outerCommInfo.localRank, outerCommInfo.localRankSize,
            topoAttr_.userRank));
    }
    CHK_SMART_PTR_NULL(outer2Executor);

    /* 节点内执行器 stage0 */
    u32 rootRank = 0;
    HcclResult ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRank);
    CHK_PRT_RET(ret == HCCL_E_PARA,
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]invalid root[%u] to get userrank", param.root), ret);

    if (ret == HCCL_SUCCESS) {
        CHK_RET(outer1Executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
                param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, rootRank, slice));

        u32 rankSize = outerCommInfo.localRankSize;
        CHK_RET(outer1Executor->RegisterProfiler(
            (0 << PROF_RINGINDEX_OFFSET_OF_PLANEID)+(rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
            outerCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(outer1Executor, outerCommInfo));
    } else {
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]invalid root[%u] to get userrank", param.root);
    }
    HCCL_INFO("broadcast meshhd stage0 run success");
    u64 hdCount = slice[outerCommInfo.localRank].size / perDataSize;
    /* 节点间执行器 stage1 */

    u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
    CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
        subUserrankRoot, topoAttr_.userRank, param.root),
        HCCL_E_INTERNAL);

    u32 subRoot = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, subRoot));

    // 增加偏移参数
    CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, hdCount,
                                   param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, subRoot,
                                   std::vector<Slice>(0), slice[outerCommInfo.localRank].offset));

    u32 rankSize = innerCommInfo.localRankSize;
    CHK_RET(innerExecutor->RegisterProfiler((0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
    HCCL_INFO("broadcast meshhd stage1 run success");

    /* 节点内执行器 stage2 */
    {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
            for (u32 streamIndex = 0; streamIndex < streamInfo_.ringStreams.size(); streamIndex++) {
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    streamInfo_.ringStreams[streamIndex].ptr(), param.stream.ptr()));
            }
        }
        CHK_RET(outer2Executor->Prepare(execMem.outputMem, execMem.outputMem, execMem.outputMem, execMem.count,
                                        param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID, slice));

        u32 rankSize = outerCommInfo.localRankSize;
        CHK_RET(outer2Executor->RegisterProfiler((0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(outer2Executor, outerCommInfo));
    }

    HCCL_INFO("broadcast meshhd stage2 run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadCastMeshExecutor", BroadcastMesh, CollBroadcastMeshExecutor);

 } // namespace hccl