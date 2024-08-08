/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "coll_alg_op_registry.h"

namespace hccl {
BroadCastOperator::BroadCastOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CommonOperator(pImpl, topoMatcher, HcclCMDType::HCCL_CMD_BROADCAST)
{
    // 由于bcast/allgather/reducescatter/reduce/send/recv暂不支持server间ring，需继续使用HD或NHR
    if (!UseInterServerNHRAlgo(algType_) && !UseInterServerNHRV1Algo(algType_) && !UseInterServerNBAlgo(algType_)) {
        SetInterServerHDAlgo(algType_);
        HCCL_WARNING("[BroadCastOperator][BroadCastOperator] do not support ring in AlgoLevel1 yet, reset algType=HD.");
    }
}
BroadCastOperator::~BroadCastOperator()
{
}

HcclResult BroadCastOperator::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
    Stream stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret;
    /* ------------集合通信资源准备------------ */
    u32 perDataSize = SIZE_TABLE[dataType];
    DeviceMem devMem(const_cast<void *>(ptr), count * perDataSize);

    if (isHaveCpuRank_) {
        algType_ = AlgType::ALG_NP_STAR;
    }

    CHK_RET(hcclImpl_->PrepareCommRes(tag, devMem, devMem, algType_, stream, root, false, isHaveCpuRank_));

    // 异构Broadcast的stream为空指针
    if (stream.ptr() != nullptr) {
        HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);
    }

    // 添加从流profiling, 用于维护planID
    CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_BROADCAST));

    /*  ------------执行算法-------------- */
    HcclUs startut = TIME_NOW();

    ret = RunBroadCast(tag, devMem, devMem, count, dataType, HCCL_REDUCE_RESERVED, root, stream, opInfo);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[BroadCastOperator][Broadcast]group has been destroyed. Break!"),
        ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastOperator][Broadcast]errNo[0x%016llx] tag[%s],braodcast op run failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);

    HCCL_INFO("tag[%s],broadcast run success,take time [%lld]us.", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadCastCommFor310P(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, u32 root, Stream &stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    std::unique_ptr<CommBase> &commCombined = currComm->commIntraServer;
    std::unique_ptr<ExecutorBase> executor;

    executor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
    CHK_SMART_PTR_NULL(executor);
    // 获取root
    u32 rootRank = 0;
    CHK_RET(commCombined->GetRankByUserRank(root, rootRank));

    CHK_RET(executor->Prepare(inputMem, outputMem, outputMem, count, dataType,
        stream, HCCL_REDUCE_RESERVED, root));

    u32 rankSize = commCombined->RankSize();
    CHK_RET(executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commCombined->Rank(),
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commCombined->RunExecutor(executor));
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::RunBroadCast(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    if (Is310P3Common()) {
        ret = BroadCastCommFor310P(tag, inputMem, outputMem, count, dataType, root, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][RunBroadCast]tag[%s], broad_cast_run failed, return[%d]", tag.c_str(), ret), ret);
        return HCCL_SUCCESS;
    }

    switch (topoType_) {
        case TopoType::TOPO_TYPE_2P_MESH:
            if (Is310PDevice()) {
                HCCL_INFO("Run broadcast by broadcast plus bcastcast");
                ret = BroadcastPlusBroadcast(tag, inputMem, outputMem, count, dataType, op, root, stream);
                break;
            }
        case TopoType::TOPO_TYPE_NP_MESH:
        case TopoType::TOPO_TYPE_4P_MESH:
        case TopoType::TOPO_TYPE_1P_MESH:
            ret = BroadCastMeshExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream);
            break;

        case TopoType::TOPO_TYPE_4P_RING:
            ret = BroadCast4pRingExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream);
            break;
        case TopoType::TOPO_TYPE_NP_SINGLE_RING:
        case TopoType::TOPO_TYPE_8P_RING:
            ret = BroadCastRingExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream, opInfo);
            break;
        case TopoType::TOPO_TYPE_NP_DOUBLE_RING:
            ret = BroadCastDoubleRingExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream, opInfo);
            break;
        case TopoType::TOPO_TYPE_ES_MESH:
            ret = BroadcastStarExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream);
            break;
        case TopoType::TOPO_TYPE_COMMON: // 后续需要重新适配
        case TopoType::TOPO_TYPE_HETEROG:
        case TopoType::TOPO_TYPE_RESERVED:
        default:
            ret = BroadCastComm(tag, inputMem, outputMem, count, dataType, op, root, stream);
            break;
    }
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[BroadCastOperator][RunBroadCast]group has been destroyed. Break!"),
        ret);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastOperator][RunBroadCast]tag[%s], broadcast failed, retrun[%d]", tag.c_str(), ret), ret);

    return ret;
}

bool BroadCastOperator::IsBroadcastSmallData(u64 size)
{
    const AlgTypeLevel0 algLevel0 = GetLevel0AlgType(algType_);
    
    u64 actualSize;
    u64 actualRankSize;
 
    if (algLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
        // level0算法配null走单层拓扑场景
        actualSize = size;
        actualRankSize = userRankSize_;
    } else {
        // 非单层拓扑场景
        actualSize = size / deviceNumPerAggregation_;
        actualRankSize = userRankSize_ / deviceNumPerAggregation_;
    }

    if (UseInterServerNHRAlgo(algType_)) {
        return actualSize <= NHR_BCAST_SMALL_SIZE;
    } else if (UseInterServerNBAlgo(algType_)) {
        return ShouldUseBinaryBroadcastOfNB(actualSize, actualRankSize, userRankSize_, deviceNumPerAggregation_);
    }

    return false;
}

HcclResult BroadCastOperator::BroadcastOutPlace(const std::string &tag, void *ptr, u64 count,
    HcclDataType dataType, u32 root, Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    if(userRankSize_ == 1 ){
        return HCCL_SUCCESS ;
    }
    // 只用commInput这段中转buffer来完成
    u32 unitSize = SIZE_TABLE[dataType];

    bool isRootRank = root == realUserRank_ ? true : false;
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    u8 *curPtr = static_cast<u8 *>(ptr);
    u64 inputOffset = 0;
    u64 countLeft = count;

    auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
    bool isMeshTopo            = IsAlgTypeLevel0Mesh(originalAlgTypeLevel0);
    bool isDMAreduceOn91073     = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
                              && (deviceType_ == DevType::DEV_TYPE_910_73) && !isMeshTopo);

    std::string newTag = tag;
    if (UseInterServerHDAlgo(algType_)) {
        u32 part1Size = 2 * (moduleNum_ - (1 << static_cast<u32>(log2(moduleNum_))));
        u32 root_id = root / deviceNumPerAggregation_;
        std::string appendTag = std::to_string((root_id >= part1Size) || ((root_id % 2) == 0));
        newTag = tag + '_' + appendTag;
        if (opBaseAtraceInfo != nullptr) {
            CHK_RET(opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, tag));
        }
    }
    HCCL_PROFILER_ADD_TAG(newTag, identifier_, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(stream.ptr(), newTag, 0, algType_);

    while (countLeft > 0) {
        curPtr += inputOffset;
        u64 curCount = ((countLeft * unitSize) > inCCLbuffer.size()) ? (inCCLbuffer.size() / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte
        HCCL_INFO("BroadcastOutPlace: buffer offset[%llu]", inputOffset);

        bool hugeData = (inCCLbuffer.size() / deviceNumPerAggregation_ > RDMA_SEND_MAX_SIZE) ||
            (curSize > SDMA_SEND_MAX_SIZE);
        bool isSmallData = IsBroadcastSmallData(curSize);
        auto meta = HcclOpMetaInfo::GetOneForBroadcast(isRootRank, root, hugeData, isSmallData);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        HCCL_INFO("BroadcastOutPlace:curPtr[%p], curCount[%llu], curSize[%llu], isSmallData[%u], "
            "deviceNumPerAggregation[%u].", curPtr, curCount, curSize, isSmallData, deviceNumPerAggregation_);
            
        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_BROADCAST, newTag, curCount, dataType,
            root, inCCLbuffer.size(), 0));

        HCCL_INFO("BroadcastOutPlace:curPtr[%p], curCount[%llu], curSize[%llu]", curPtr, curCount, curSize);
        HcclResult ret;
        /* 入参的正确性由HCCL确保 */
        if (isDMAreduceOn91073) {
            HcomCollOpInfo opInfo;
            opInfo.inputAddr = curPtr;
            opInfo.outputAddr = curPtr;
            opInfo.count = count;
            opInfo.dataType = dataType;
            ret = Broadcast(newTag, inCCLbuffer.ptr(), curCount, dataType, root, stream, &opInfo);
        } else {
            DeviceMem commMem = inCCLbuffer.range(0, curSize);
            DeviceMem userMem(curPtr, curSize);
            if (userRank_ == root) { // 本rank为root节点，非root节点不需要拷贝到中转内存
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, commMem, userMem, stream));
            }
            ret = Broadcast(newTag, inCCLbuffer.ptr(), curCount, dataType, root, stream);
            if (realUserRank_ != root) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMem, commMem, stream));
            }
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Broadcast]errNo[0x%016llx] OP_BASE hcclComm broadcast, tag[%s], input_ptr[%p], "
                       "count[%llu], data_type[%s], root[%u]",
            HCCL_ERROR_CODE(ret), newTag.c_str(), inCCLbuffer.ptr(), curCount, GetDataTypeEnumStr(dataType).c_str(),
                root),
            ret);

        ret = RankConsistent::GetInstance().DelOpPara(newTag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Broadcast]errNo[0x%016llx] delete CMD with parameters error. tag[%s]",
            HCCL_ERROR_CODE(ret), newTag.c_str()),
            ret);

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][Broadcast]In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        inputOffset = curSize;

        CHK_RET(LaunchTask(dispatcher_, stream));
    }
    HCCL_PROFILER_DEL_STREAM(stream.ptr());
    HCCL_PROFILER_DEL_TAG(newTag);
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadCastComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    (void)op;
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    if (currComm->commInner.size() <= 0) {
        HCCL_ERROR("[BroadCastOperator][BroadCastComm]errNo[0x%016llx] tag[%s],broadcast op inner comm is empty",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), tag.c_str());
        return HCCL_E_INTERNAL;
    }

    std::unique_ptr<CommBase> &commCombine = (isHaveCpuRank_ == true) ?
        currComm->commOuter[COMM_INDEX_0] : currComm->commInner[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commCombine);

    std::unique_ptr<ExecutorBase> executor;
    u64 curSize = count * SIZE_TABLE[dataType];
    if (UseInterServerNHRAlgo(algType_)) {
        if (curSize <= NHR_BCAST_SMALL_SIZE) {
            executor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
        } else {
            executor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
        }
        HCCL_INFO("broadcast comm: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        executor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
        HCCL_INFO("broadcast comm: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        if (ShouldUseBinaryBroadcastOfNB(curSize, commCombine->RankSize(), userRankSize_, deviceNumPerAggregation_)) {
            executor.reset(new (std::nothrow) BroadcastNBBinary(dispatcher_));
        } else {
            executor.reset(new (std::nothrow) BroadcastNB(dispatcher_));
        }
        HCCL_INFO("broadcast comm: using nonuniform-bruck algo inter-server.");
    } else {
        executor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
        HCCL_INFO("broadcast comm: using ring algo inter-server.");
    }
    CHK_SMART_PTR_NULL(executor);

    // 获取root
    u32 rootRank = 0;
    CHK_RET(commCombine->GetRankByUserRank(root, rootRank));

    CHK_RET(RunExecutor(commCombine, executor, inputMem, outputMem, count, dataType,
                        HCCL_REDUCE_RESERVED, rootRank, stream));

    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadCastMeshExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    u32 perDataSize = SIZE_TABLE[dataType];

    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    std::unique_ptr<ExecutorBase> outer1Executor;
    std::unique_ptr<ExecutorBase> innerExecutor;
    std::unique_ptr<ExecutorBase> outer2Executor;

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    if (currComm->commOuter.size() == 0) {
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]errNo[0x%016llx] tag[%s]'s outer comm is empty",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), tag.c_str());
        return HCCL_E_INTERNAL;
    }
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    outer1Executor.reset(
        new (std::nothrow) ScatterMesh(dispatcher_, commOuter->Rank(), commOuter->RankSize()));
    CHK_SMART_PTR_NULL(outer1Executor);
    outer1Executor->CloseBarrier();

    /* 内层topo:all_reduce */
    /* 外层所有rank均参与内层的broadcast计算，所以此处对rank不作限制，但是每个rank需找到自己所在的内层通信域 */
    u32 commIndex = commOuter->Rank();
    std::vector<Slice> slice;
    CHK_RET(GetRankSliceSize(dataType, count, commOuter->RankSize(), slice));

    CHK_PRT_RET(slice.empty(), HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]got slice is empty"),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(commIndex >= currComm->commInner.size(),
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]commIndex[%u] >=(tag[%s])comm_inner.size[%llu]",
            commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    u64 curSize = count * SIZE_TABLE[dataType];
    if (UseInterServerNHRAlgo(algType_)) {
        HCCL_DEBUG("broadcast mesh: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
            curSize, deviceNumPerAggregation_, commOuter->RankSize());
        if (curSize / deviceNumPerAggregation_ <= NHR_BCAST_SMALL_SIZE) {
            innerExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
        } else {
            innerExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
        }
        HCCL_INFO("broadcast mesh: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        innerExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
        HCCL_INFO("broadcast mesh: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        const u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
        if (ShouldUseBinaryBroadcastOfNB(curSize / deviceNumPerAggregation_, innerRankSize, userRankSize_,
            deviceNumPerAggregation_)) {
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
    if (deviceType_ == DevType::DEV_TYPE_910B) {
        outer2Executor.reset(
            new (std::nothrow) AllGatherMeshAtomic(dispatcher_, streamInfo->ringStreams,
            streamInfo->ringSignal, streamInfo->ringSignalAux, commOuter->Rank(), commOuter->RankSize(),
            commOuter->UserRank()));
    } else {
        outer2Executor.reset(
            new (std::nothrow) AllGatherMesh(dispatcher_, streamInfo->ringStreams, streamInfo->ringSignal,
            streamInfo->ringSignalAux, commOuter->Rank(), commOuter->RankSize(),
            commOuter->UserRank()));
    }
    CHK_SMART_PTR_NULL(outer2Executor);

    /* 节点内执行器 stage0 */
    u32 rootRank = 0;
    HcclResult ret = commOuter->GetRankByUserRank(root, rootRank);

    CHK_PRT_RET(ret == HCCL_E_PARA,
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]invalid root[%u] to get userrank", root), ret);

    if (ret == HCCL_SUCCESS) {
        CHK_RET(outer1Executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op, rootRank, slice));

        u32 rankSize = commOuter->RankSize();
        CHK_RET(outer1Executor->RegisterProfiler(
            (0 << PROF_RINGINDEX_OFFSET_OF_PLANEID)+(rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
            commOuter->Rank(), PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(commOuter->RunExecutor(outer1Executor));
    }
    HCCL_INFO("broadcast meshhd stage0 run success");
    u64 hdCount = slice[commOuter->Rank()].size / perDataSize;
    /* 节点间执行器 stage1 */
    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
    std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
    u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
    CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
        subUserrankRoot, userRank_, root),
        HCCL_E_INTERNAL);

    u32 subRoot = 0;
    CHK_RET(commInner->GetRankByUserRank(subUserrankRoot, subRoot));

    // 增加偏移参数
    CHK_RET(innerExecutor->Prepare(inputMem, outputMem, outputMem, hdCount, dataType, stream, op, subRoot,
        std::vector<Slice>(0), slice[commOuter->Rank()].offset));

    u32 rankSize = currComm->commInner[commIndex]->RankSize();
    CHK_RET(innerExecutor->RegisterProfiler((0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(currComm->commInner[commIndex]->RunExecutor(innerExecutor));
    HCCL_INFO("broadcast meshhd stage1 run success");

    /* 节点内执行器 stage2 */
    {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
            for (u32 streamIndex = 0; streamIndex < streamInfo->ringStreams.size(); streamIndex++) {
                CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                    streamInfo->ringStreams[streamIndex].ptr(), stream.ptr()));
            }
        }
        CHK_RET(outer2Executor->Prepare(outputMem, outputMem, outputMem, count, dataType, stream, op,
                                        OUTER_BRIDGE_RANK_ID, slice));

        u32 rankSize = commOuter->RankSize();
        CHK_RET(outer2Executor->RegisterProfiler((0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commOuter->Rank(),
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(commOuter->RunExecutor(outer2Executor));
    }

    HCCL_INFO("broadcast meshhd stage2 run success");
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadCast4pRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    std::unique_ptr<ExecutorBase> outer1Executor;
    std::unique_ptr<ExecutorBase> innerExecutor;
    std::unique_ptr<ExecutorBase> outer2Executor;

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[BroadCastOperator][BroadCast4pRingExecutor]outer comm(tag[%s]) is empty",
        tag.c_str()), HCCL_E_INTERNAL);

    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    /* 外层:reduce_scatter */
    // 构造ring algorithm对应的reduce-scatter实例
    outer1Executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
    CHK_SMART_PTR_NULL(outer1Executor);

    /* 内层topo:all_reduce */
    /* 外层所有rank均参与内层的broadcast计算，所以此处对rank不作限制，但是每个rank需找到自己所在的内层通信域 */
    u32 commIndex = commOuter->Rank();
    std::vector<Slice> slice;
    CHK_RET(GetRankSliceSize(dataType, count, commOuter->RankSize(), slice));

    CHK_PRT_RET(slice.empty(), HCCL_ERROR("[BroadCastOperator][BroadCast4pRingExecutor]got slice is empty"),
        HCCL_E_INTERNAL);

    // 判断内层环是否OK
    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet, HCCL_ERROR("[BroadCastOperator][BroadCast4pRingExecutor]commIndex[%u] >=(tag[%s])", \
        "comm_inner.size[%llu]", commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    u64 curSize = count * SIZE_TABLE[dataType];
    if (UseInterServerNHRAlgo(algType_)) {
        HCCL_DEBUG("broadcast 4p ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
            curSize, deviceNumPerAggregation_, commOuter->RankSize());
        if (curSize / deviceNumPerAggregation_ <= NHR_BCAST_SMALL_SIZE) {
            innerExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
        } else {
            innerExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
        }
        HCCL_INFO("broadcast 4p ring: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        innerExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
        HCCL_INFO("broadcast 4p ring: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        const u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
        if (ShouldUseBinaryBroadcastOfNB(curSize / deviceNumPerAggregation_, innerRankSize, userRankSize_,
            deviceNumPerAggregation_)) {
            innerExecutor.reset(new (std::nothrow) BroadcastNBBinary(dispatcher_));
        } else {
            innerExecutor.reset(new (std::nothrow) BroadcastNB(dispatcher_));
        }
        HCCL_INFO("broadcast 4p ring: using nonuniform-bruck algo inter-server.");
    } else {
        innerExecutor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
        HCCL_INFO("broadcast 4p ring: using Recursive halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    /* 外层topo:all_gather */
    outer2Executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
    CHK_SMART_PTR_NULL(outer2Executor);

    /* 节点内执行器 stage0 */
    u32 rootRank = 0;
    HcclResult ret = commOuter->GetRankByUserRank(root, rootRank);
    CHK_PRT_RET(ret == HCCL_E_PARA, HCCL_ERROR("[BroadCastOperator][BroadCast4pRingExecutor]invalid root rank[%u]"\
        " to get user rank", root), ret);

    if (ret == HCCL_SUCCESS) {
        CHK_RET(outer1Executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op, rootRank, slice));

        CHK_RET(commOuter->RunExecutor(outer1Executor));
    }
    HCCL_INFO("broadcast 4pringhd stage0 run success");

    u64 hdCount = slice[commOuter->Rank()].size / perDataSize;

    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
    std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
    u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
    CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR("[BroadCastOperator][BroadCast4pRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, userRank_, root), HCCL_E_INTERNAL);
    u32 subRoot = 0;
    CHK_RET(commInner->GetRankByUserRank(subUserrankRoot, subRoot));

    CHK_RET(innerExecutor->Prepare(inputMem, inputMem, outputMem, hdCount, dataType, stream, op,
                                    subRoot, std::vector<Slice>(0), slice[commOuter->Rank()].offset));

    CHK_RET(currComm->commInner[commIndex]->RunExecutor(innerExecutor));

    HCCL_INFO("broadcast 4pringhd stage1 run success");

    /* 节点内执行器 stage2 */
    {
        CHK_RET(outer2Executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op,
                                        OUTER_BRIDGE_RANK_ID, slice));

        CHK_RET(commOuter->RunExecutor(outer2Executor));
    }
    HCCL_INFO("broadcast 4pringhd stage1 run success");

    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadcastPlusBroadcast(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.size() == 0, HCCL_ERROR("commOuter size is zero"), HCCL_E_PARA);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    bool inSameServer = false;
    CHK_RET(IsUserRankInSameServer(userRank_, root, inSameServer));

    bool rootIsDevPhyZero = [this] (u32 rootRank) {
        for (u32 i = 0; i < rankInfoList_.size(); i++) {
            if (rankInfoList_[i].userRank == rootRank) {
                return rankInfoList_[i].devicePhyId == 0;
            }
        }
        return false;
    } (root);
    // 第一步，如果root不在dev 0上，先将数据bcast到设备0上，在进行server间bcast，设备0调度网卡更快
    if (inSameServer && !rootIsDevPhyZero) {
        u32 rootRank = 0;
        CHK_RET(commOuter->GetRankByUserRank(root, rootRank));
        std::unique_ptr<ExecutorBase> bCastRingInNode(new (std::nothrow) BroadcastRing(dispatcher_));
        CHK_SMART_PTR_NULL(bCastRingInNode);
        CHK_RET(bCastRingInNode->Prepare(inputMem, outputMem, inputMem, count, dataType, stream, op, rootRank));
        CHK_RET(commOuter->RunExecutor(bCastRingInNode));
    }
    // 第二步，进行server间bcast
    if (devicePhyId_ == 0) {
        std::unique_ptr<ExecutorBase> broadcastExecutor = nullptr;
        u64 curSize = count * SIZE_TABLE[dataType];
        if (UseInterServerRingAlgo(algType_)) {
            broadcastExecutor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
            HCCL_INFO("broadcast ring: using ring algo inter-server.");
        } else if (UseInterServerNHRAlgo(algType_)) {
            HCCL_DEBUG("broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, deviceNumPerAggregation_, commOuter->RankSize());
            if (curSize / deviceNumPerAggregation_ <= NHR_BCAST_SMALL_SIZE) {
                broadcastExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
            } else {
                broadcastExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            broadcastExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            const u32 innerRankSize = currComm->commInner[COMM_INDEX_0]->RankSize();
            if (ShouldUseBinaryBroadcastOfNB(curSize / deviceNumPerAggregation_, innerRankSize, userRankSize_,
                deviceNumPerAggregation_)) {
                broadcastExecutor.reset(new (std::nothrow) BroadcastNBBinary(dispatcher_));
            } else {
                broadcastExecutor.reset(new (std::nothrow) BroadcastNB(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nonuniform-bruck algo inter-server.");
        } else {
            broadcastExecutor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("broadcast recursive hd: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(broadcastExecutor);
        u32 innerRootUserRank;
        // 获取root所在的server的device0的userRank
        CHK_RET(GetUserRankByDevIDInServerOfUserRankIn(root, 0, innerRootUserRank));

        CHK_PRT_RET(currComm->commInner.size() == 0, HCCL_ERROR("commInner size is zero"), HCCL_E_PARA);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[COMM_INDEX_0];
        CHK_SMART_PTR_NULL(commInner);
        u32 planeRoot = 0;
        CHK_RET(commInner->GetRankByUserRank(innerRootUserRank, planeRoot));
        CHK_RET(broadcastExecutor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op, planeRoot));

        CHK_RET(commInner->RunExecutor(broadcastExecutor));
    }
    // 第三步，执行server内broadcast（从设备0到设备1）
    std::unique_ptr<ExecutorBase> bcastExecutor(new (std::nothrow) BroadcastRing(dispatcher_));
    CHK_SMART_PTR_NULL(bcastExecutor);
    u32 outerRootUserRank;
    // 获取本rank所在server上device0的UserRank
    CHK_RET(GetUserRankByDevIDInServerOfUserRankIn(userRank_, 0, outerRootUserRank));
    u32 rootRank = 0;
    CHK_RET(commOuter->GetRankByUserRank(outerRootUserRank, rootRank));
    CHK_RET(bcastExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, op, rootRank));
    CHK_RET(commOuter->RunExecutor(bcastExecutor));

    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadCastDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem,
                                                          DeviceMem &outputMem, u64 count, HcclDataType dataType,
                                                          HcclReduceOp op, u32 root, Stream &stream,
                                                          const HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[BroadCastOperator][BroadCastDoubleRingExecutor] The BroadCastDoubleRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    std::vector<Slice>              dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> mulRingSlice;  // 数据基于该rank上环0的偏移
    // step1: 节点内的scatter
    u32       ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    bool bRet = currComm->commOuter.size() < ringNum;
    CHK_PRT_RET(bRet,
                HCCL_ERROR("[BroadCastOperator][BroadCastDoubleRingExecutor]bcast double ring comm(tag:[%s]) ",
                           "size is[%llu],not[%u]", tag.c_str(), currComm->commOuter.size(), ringNum),
                HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    // 按ranksize得到内存切分slice数
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    // 将根节点数据切分成sliceNum份
    CHK_RET(ExecutorBase::PrepareSliceData(count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 外层:scatter */
    // 将每slice再切分成2份，按各ring的dev顺序排列
    // 构造ring algorithm对应的reduce-scatter实例
    mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);
    CHK_PRT_RET(mulRingSlice.size() != ringNum,
                HCCL_ERROR("[BroadCastOperator][BroadCastDoubleRingExecutor]"
                           "ringNum[%u] !=mulRingSlice size[%llu]",
                           ringNum, mulRingSlice.size()),
                HCCL_E_INTERNAL);
    if (opInfo != nullptr) {
        HcomCollOpInfo opInfoByBroadcastDMAreduce = *opInfo;
        opInfoByBroadcastDMAreduce.outputAddr     = nullptr;
        CHK_RET(MultiRingScatter(tag, inputMem, outputMem, count, dataType, mulRingSlice, root, stream,
                                 &opInfoByBroadcastDMAreduce));
    } else {
        CHK_RET(MultiRingScatter(tag, inputMem, outputMem, count, dataType, mulRingSlice, root, stream, opInfo));
    }

    HCCL_INFO("Broadcast double ring stage0 run success");

    CHK_SMART_PTR_NULL(currComm->commLevel2[0]);
    u32 level2RankSize = currComm->commLevel2[0]->RankSize();

    u64 hdCount = 0;
    u64 hdSize  = 0;

    if (level2RankSize <= 1) {
        // step2: 节点间的broadcast
        u32 segmentIdx;
        u32 commIndex;
        CHK_RET(hcclImpl_->PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, currComm->commOuter, mulRingSlice, tag));

        hdCount = hdSize / perDataSize;

        HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, tag.c_str(),
                   currComm->commInner.size());

        bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet,
                    HCCL_ERROR("[HcclImpl][BroadCastDoubleRingExecutor]commIndex[%u] >= "
                               "(tag:[%s])comm_inner.size[%llu]",
                               commIndex, tag.c_str(), currComm->commInner.size()),
                    HCCL_E_INTERNAL);

        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        std::unique_ptr<ExecutorBase> innerExecutor;
        u64 curSize = count * SIZE_TABLE[dataType];
        if (UseInterServerNHRAlgo(algType_)) {
            HCCL_DEBUG("broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, deviceNumPerAggregation_, currComm->commOuter[COMM_INDEX_0]->RankSize());
            if (curSize / deviceNumPerAggregation_ <= NHR_BCAST_SMALL_SIZE) {
                innerExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
            } else {
                innerExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            const u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
            if (ShouldUseBinaryBroadcastOfNB(curSize / deviceNumPerAggregation_, innerRankSize, userRankSize_,
                deviceNumPerAggregation_)) {
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

        u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
        CHK_PRT_RET(
            subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[HcclImpl][BroadCastDoubleRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                       subUserrankRoot, userRank_, root),
            HCCL_E_INTERNAL);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        u32                        planeRoot = 0;
        CHK_RET(commInner->GetRankByUserRank(subUserrankRoot, planeRoot));

        u32 ranksize = commInner->RankSize();
        // 节点间的hd 使用环0来记录
        CHK_RET(innerExecutor->Prepare(inputMem, inputMem, outputMem, hdCount, dataType, stream, op, planeRoot,
                                       std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));

        CHK_RET(innerExecutor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(),
                                                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(commInner->RunExecutor(innerExecutor));

        HCCL_INFO("Broadcast double ring stage1 run success");

        // step3: 节点内的allgatherring
        if (opInfo != nullptr) {
            HcomCollOpInfo opInfoByBroadcastDMAreduce = *opInfo;
            opInfoByBroadcastDMAreduce.inputAddr      = nullptr;
            CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, hdCount, dataType, mulRingSlice, stream, PROF_STAGE_2,
                                       0, &opInfoByBroadcastDMAreduce));
        } else {
            CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, hdCount, dataType, mulRingSlice, stream, PROF_STAGE_2,
                                       0, opInfo));
        }
        HCCL_INFO("Broadcast double ring stage2 run success");
    } else {
        // step2: 节点间的scatter
        /* count数据准备 */
        std::vector<Slice> level1dataSegsSlice; // 数据分成inner ranksize份，每份的起始偏移和大小
        u32                commIndex = currComm->commOuter[COMM_INDEX_0]->Rank();
        bRet                         = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet,
                    HCCL_ERROR("[BroadCastOperator][BroadCastDoubleRingExecutor]commIndex[%u] >= "
                               "(tag:[%s])comm_inner.size[%llu]",
                               commIndex, tag.c_str(), currComm->commInner.size()),
                    HCCL_E_INTERNAL);

        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);

        // level 1通信数据量
        u64 level1count = count / sliceNum;

        // 按ranksize得到内存切分slice数
        u32 level1sliceNum = currComm->commInner[commIndex]->RankSize();
        // 将根节点数据切分成level1sliceNum份
        CHK_RET(ExecutorBase::PrepareSliceData(level1count, perDataSize, level1sliceNum, 0, level1dataSegsSlice));

        u64 level1segmentIdx = currComm->commInner[commIndex]->Rank();

        DeviceMem level1InputMem
            = inputMem.range(level1dataSegsSlice[level1segmentIdx].offset, (level1count * perDataSize));
        DeviceMem level1OutputMem
            = outputMem.range(level1dataSegsSlice[level1segmentIdx].offset, (level1count * perDataSize));

        std::unique_ptr<ExecutorBase> level1Executor;
        level1Executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
        CHK_SMART_PTR_NULL(level1Executor);
        CHK_RET(level1Executor->Prepare(level1InputMem, level1InputMem, level1OutputMem, level1count, dataType, stream,
                                        op, OUTER_BRIDGE_RANK_ID, level1dataSegsSlice,
                                        level1dataSegsSlice[level1segmentIdx].offset));
        CHK_RET(level1Executor->RegisterProfiler((level1sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                                                     + currComm->commInner[commIndex]->Rank(),
                                                 PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(currComm->commInner[commIndex]->RunExecutor(level1Executor));
        HCCL_INFO("Broadcast double ring [superpod] level1 run success");

        // step3: 超节点间的broadcast
        u64                             level2hdSize;
        u32                             level2segmentIdx;
        u32                             level2commIndex;
        std::vector<std::vector<Slice>> multiRingSlice;
        multiRingSlice.push_back(level1dataSegsSlice);
        CHK_RET(hcclImpl_->PrepareInnerCommInfo(level2segmentIdx, level2commIndex, level2hdSize, currComm->commOuter,
                                                multiRingSlice, tag));

        u64 level2hdCount = level2hdSize / perDataSize;

        HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commLevel2.size():%llu", level2commIndex, tag.c_str(),
                   currComm->commLevel2.size());

        bRet = level2commIndex >= currComm->commLevel2.size();
        CHK_PRT_RET(bRet,
                    HCCL_ERROR("[BroadCastOperator][BroadCastDoubleRingExecutor]level2commIndex[%u] >= "
                               "(tag:[%s])comm_inner.size[%llu]",
                               level2commIndex, tag.c_str(), currComm->commLevel2.size()),
                    HCCL_E_INTERNAL);

        CHK_SMART_PTR_NULL(currComm->commLevel2[0]);
        std::unique_ptr<ExecutorBase> level2Executor;
        if (UseLevel2RingAlgo(algType_)) {
            level2Executor.reset(new (std::nothrow) BroadcastRing(dispatcher_));
            HCCL_INFO("broadcast ring: using ring algo inter-server.");
        } else {
            level2Executor.reset(new (std::nothrow) BcastRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("broadcast ring: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level2Executor);

        u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
        CHK_PRT_RET(
            subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[BroadCastOperator][BroadCastDoubleRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],"
                       "root[%u]",
                       subUserrankRoot, userRank_, root),
            HCCL_E_INTERNAL);
        std::unique_ptr<CommBase> &commlevel2 = currComm->commLevel2[0];
        u32                        planeRoot  = 0;
        CHK_RET(commlevel2->GetRankByUserRank(subUserrankRoot, planeRoot));

        u32 ranksize = commlevel2->RankSize();
        // 节点间的hd 使用环0来记录
        CHK_RET(level2Executor->Prepare(inputMem, inputMem, outputMem, level2hdCount, dataType, stream, op, planeRoot,
                                        std::vector<Slice>(0), level1dataSegsSlice[level2segmentIdx].offset));

        CHK_RET(level2Executor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commlevel2->Rank(),
                                                 PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(commlevel2->RunExecutor(level2Executor));

        // step4: 节点间的allgather
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("allgather ring: using halving-doubling algo inter-server.");
        }

        CHK_SMART_PTR_NULL(innerExecutor);
        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
        CHK_RET(innerExecutor->Prepare(outputMem, outputMem, inputMem, level2hdCount, dataType, stream,
                                       HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, std::vector<Slice>(COMM_INDEX_0),
                                       0));

        u32 rankSize = commInner->RankSize();
        CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(),
                                                PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(commInner->RunExecutor(innerExecutor));

        // step5: 节点内的allgatherring
        u64 level0count = level2hdCount * rankSize;
        if (opInfo != nullptr) {
            HcomCollOpInfo opInfoByBroadcastDMAreduce = *opInfo;
            opInfoByBroadcastDMAreduce.inputAddr      = nullptr;
            CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, level0count, dataType, mulRingSlice, stream,
                                       PROF_STAGE_2, 0, &opInfoByBroadcastDMAreduce));
        } else {
            CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, level0count, dataType, mulRingSlice, stream,
                                       PROF_STAGE_2, 0, opInfo));
        }
        HCCL_INFO("Broadcast[superpod] double ring stage5 run success");
    }

    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadCastRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, const HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[BroadCastOperator][BroadCastRingExecutor] The BroadCastRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > mulRingSlice; // 数据基于该rank上环0的偏移
    // step1: 节点内的scatter
    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? OUTER_PLANE_NUM_IN_8PRING :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE;
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    bool bRet = currComm->commOuter.size() < ringNum;
    CHK_PRT_RET(bRet, HCCL_ERROR("[BroadCastOperator][BroadCastRingExecutor](tag:[%s]) size is[%llu],not[%u]",
        tag.c_str(), currComm->commOuter.size(), ringNum), HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    // 按ranksize得到内存切分slice数
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    // 将根节点数据切分成sliceNum份
    CHK_RET(ExecutorBase::PrepareSliceData(count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 外层:scatter */
    // 将每slice再切分成4份，按各ring的dev顺序排列
    if (ringNum == OUTER_PLANE_NUM_IN_8PRING) {
        // 构造ring algorithm对应的reduce-scatter实例
        mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);
        CHK_PRT_RET(mulRingSlice.size() != ringNum, HCCL_ERROR("[BroadCastOperator][BroadCastRingExecutor]ringNum[%u] "\
            "!=mulRingSlice size[%llu]", ringNum, mulRingSlice.size()), HCCL_E_INTERNAL);
    } else {
        mulRingSlice.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }

    if (opInfo != nullptr) {
        HcomCollOpInfo opInfoByBroadcastDMAreduce = *opInfo;
        opInfoByBroadcastDMAreduce.outputAddr     = nullptr;
        CHK_RET(MultiRingScatter(tag, inputMem, outputMem, count, dataType, mulRingSlice, root, stream,
                                 &opInfoByBroadcastDMAreduce));
    } else {
        CHK_RET(MultiRingScatter(tag, inputMem, outputMem, count, dataType, mulRingSlice, root, stream, opInfo));
    }
    HCCL_INFO("broadcast 8PringHD stage0 run success");

    // step2: 节点间的broadcast
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(hcclImpl_->PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, currComm->commOuter, mulRingSlice, tag));

    u64 hdCount = hdSize / perDataSize;
    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList_.size() != DEVICE_EIGHT;
    std::vector<u32>::iterator iterNic = std::find(nicList_.begin(), nicList_.end(), devicePhyId_);
    bool innRunRet = isMultiNic && (iterNic == nicList_.end());
    if (!innRunRet) { // 满足以下条件, 不做server间通信: 1. 8P ring的拓扑 2. 网口不满配 3. 当前device不出网口
        HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, tag.c_str(),
            currComm->commInner.size());

        bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet, HCCL_ERROR("[BroadCastOperator][BroadCastRingExecutor]commIndex[%u] >= "\
            "(tag:[%s])comm_inner.size[%llu]", commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        std::unique_ptr<ExecutorBase> innerExecutor;
        u64 curSize = count * SIZE_TABLE[dataType];
        if (UseInterServerNHRAlgo(algType_)) {
            HCCL_DEBUG("broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
                curSize, deviceNumPerAggregation_, currComm->commOuter[COMM_INDEX_0]->RankSize());
            if (curSize / deviceNumPerAggregation_ <= NHR_BCAST_SMALL_SIZE) {
                innerExecutor.reset(new (std::nothrow) BroadcastNHROneshot(dispatcher_));
            } else {
                innerExecutor.reset(new (std::nothrow) BroadcastNHR(dispatcher_));
            }
            HCCL_INFO("broadcast ring: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) BroadcastNHRV1(dispatcher_));
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            const u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
        if (ShouldUseBinaryBroadcastOfNB(curSize / deviceNumPerAggregation_, innerRankSize, userRankSize_,
            deviceNumPerAggregation_)) {
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

        u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
        CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[BroadCastOperator][BroadCastRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                subUserrankRoot, userRank_, root), HCCL_E_INTERNAL);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        u32 planeRoot = 0;
        CHK_RET(commInner->GetRankByUserRank(subUserrankRoot, planeRoot));

        u32 ranksize = commInner->RankSize();
        // 节点间的hd 使用环0来记录
        CHK_RET(innerExecutor->Prepare(inputMem, inputMem, outputMem, hdCount, dataType, stream,
            op, planeRoot, std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));

        CHK_RET(innerExecutor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(), \
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(commInner->RunExecutor(innerExecutor));
    }
    HCCL_INFO("broadcast 8PringHD stage1 run success");

    // step3: 节点内的allgatherring
    if (opInfo != nullptr) {
        HcomCollOpInfo opInfoByBroadcastDMAreduce = *opInfo;
        opInfoByBroadcastDMAreduce.inputAddr      = nullptr;
        CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, hdCount, dataType, mulRingSlice, stream, PROF_STAGE_2, 0,
                                   &opInfoByBroadcastDMAreduce));
    } else {
        CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, hdCount, dataType, mulRingSlice, stream, PROF_STAGE_2, 0,
                                   opInfo));
    }

    HCCL_INFO("broadcast 8PringHD stage2 run success");
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::BroadcastStarExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    std::unique_ptr<ExecutorBase> BcastStarExecutor;
    BcastStarExecutor.reset(new (std::nothrow) BroadcastStar(dispatcher_, userRank_));
    CHK_SMART_PTR_NULL(BcastStarExecutor);

    std::vector<u32> nicRankList{0, 1};
    CHK_RET(BcastStarExecutor->Prepare(inputMem, outputMem, inputMem, count, dataType, stream, op, root,
        std::vector<Slice>(0), 0, nicRankList));

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.size() == 0, HCCL_ERROR("commOuter size is zero"), HCCL_E_PARA);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    CHK_RET(commOuter->RunExecutor(BcastStarExecutor));
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::GetUserRankByDevIDInServerOfUserRankIn(u32 userRankIn, s32 devPyID, u32& userRankOut)
{
    bool foundServerIdx = false;
    u32 serverIdx = 0;
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].userRank == userRankIn) {
            serverIdx = rankInfoList_[i].serverIdx;
            foundServerIdx = true;
            break;
        }
    }
    if (foundServerIdx) {
        for (u32 i = 0; i < rankInfoList_.size(); i++) {
            if (rankInfoList_[i].serverIdx == serverIdx && rankInfoList_[i].devicePhyId == devPyID) {
                userRankOut = rankInfoList_[i].userRank;
                return HCCL_SUCCESS;
            }
        }
    }

    return HCCL_E_PARA;
}

HcclResult BroadCastOperator::IsUserRankInSameServer(u32 userRankA, u32 userRankB, bool &inSameServer)
{
    u32 serverIdxA = 0;
    bool foundServerIdxA = false;
    u32 serverIdxB = 0;
    bool foundServerIdxB = false;

    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        if (userRankA == rankInfoList_[i].userRank) {
            serverIdxA = rankInfoList_[i].serverIdx;
            foundServerIdxA = true;
        }
        if (userRankB == rankInfoList_[i].userRank) {
            serverIdxB = rankInfoList_[i].serverIdx;
            foundServerIdxB = true;
        }
        if (foundServerIdxA && foundServerIdxB) {
            inSameServer = (serverIdxA == serverIdxB);
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_PARA;
}

HcclResult BroadCastOperator::GetRankSliceSize(HcclDataType dataType, const u64 count, const u32 rankSize,
    std::vector<Slice> &sliceList)
{
    if (rankSize <= 0) {
        HCCL_ERROR("[Get][RankSliceSize]errNo[0x%016llx] rankSize[%u] is invalid", HCCL_ERROR_CODE(HCCL_E_PARA),
            rankSize);
        return HCCL_E_PARA;
    }

    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    u64 align = (count * perDataSize) / rankSize; // 按128字节对齐整除均分
    if ((count % rankSize) > 0) {
        align += 1;
    }

    u64 sliceSize = ExecutorBase::RoundUpWithDivisor(align, HCCL_MIN_SLICE_ALIGN);
    u64 residueSize = count * perDataSize;

    for (u32 i = 0; i < rankSize; i++) {
        Slice slice;
        slice.size = sliceSize < residueSize ? sliceSize : residueSize;
        slice.offset = (slice.size == 0) ? 0 : (i * sliceSize);
        residueSize -= slice.size;

        // 将cout转换为字节数
        sliceList.push_back(slice);
    }

    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;
    if (Is310P3Common()) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (Is310PDevice() && topoType_ == TopoType::TOPO_TYPE_2P_MESH) {
        ret = SelectAlgfor310P(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_73) {
        ret = SelectAlgfor91073(param, algName);
    } else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag;
    } else {
        if (UseInterServerHDAlgo(algType_)) {
            u32 part1Size = 2 * (moduleNum_ - (1 << static_cast<u32>(log2(moduleNum_))));
            u32 rootId = param.root / deviceNumPerAggregation_;
            std::string appendTag = std::to_string((rootId >= part1Size) || ((rootId % 2) == 0));
            newTag = newTag + '_' + appendTag;
            if (param.opBaseAtraceInfo != nullptr) {
                CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
            }
        } else if (Is310P3Common()) {
            newTag = tag + algName;
        } else {
            AlgTypeLevel1 algType1 = GetLevel1AlgType(algType_);
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
            newTag = tag + level1Iter->second + algName;
        }
    }
    HCCL_INFO("[SelectAlg] broadcast newTag is [%s]", newTag.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastSelector][SelectAlg]tag[%s], broadcast failed, return[%d]", tag.c_str(), ret), ret);
    return ret;
}

HcclResult BroadCastOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "BroadCastCommFor310P";
    HCCL_INFO("[SelectAlgfor310P3] broadcast SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor310P(const OpParam& param, std::string& algName)
{
    algName = "BroadcastPlusBroadcast";
    HCCL_INFO("[SelectAlgfor310P] broadcast SelectAlgfor310P is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910A] broadcast SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910B] broadcast SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor91073(const OpParam& param, std::string& algName)
{
    if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        algName = "BroadCastRingExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        algName = "BroadCastDoubleRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor91073] broadcast SelectAlgfor91073 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_BROADCAST, Broadcast, BroadCastOperator);

}